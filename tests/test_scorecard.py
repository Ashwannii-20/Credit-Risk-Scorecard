import json

import numpy as np
import pandas as pd
import pytest

from scorecard.binning import bin_numeric, fit_binning, woe_transform
from scorecard.data import (CANDIDATE_FEATURES, LEAKAGE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                            PROJECT_ROOT, TARGET, add_target, load_raw, make_splits)
from scorecard.model import Scaling, fit_with_sign_check, score_points, scorecard_table
from scorecard.validation import psi, traffic_light

OUT = PROJECT_ROOT / "outputs"


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def built():
    df = add_target(load_raw())
    s = make_splits(df)
    tr = s["train"]
    b = fit_binning(tr, tr[TARGET], list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES))
    feats = ["initial_rating", "sector", "interest_coverage"]
    m, _ = fit_with_sign_check(woe_transform(tr, b, feats), tr[TARGET], feats)
    card = scorecard_table(m, b, Scaling())
    return s, b, m, card


# ------------------------------------------------------------------ leakage
def test_no_leakage_columns_in_candidates():
    assert not set(CANDIDATE_FEATURES) & set(LEAKAGE_COLUMNS)
    for c in ("pd_annual", "lgd", "el", "unexpected_loss", "rwa", "survival_months",
              "recovery_rate", "loss_given_default", "default_date", "defaulted"):
        assert c in LEAKAGE_COLUMNS


def test_final_model_and_scorecard_have_no_leakage(built):
    _, _, m, card = built
    assert not set(m.features) & set(LEAKAGE_COLUMNS)
    assert not set(card["feature"]) & set(LEAKAGE_COLUMNS)


def test_saved_outputs_have_no_leakage():
    if not (OUT / "scorecard.csv").exists():
        pytest.skip("run scripts/run_scorecard.py first")
    card = pd.read_csv(OUT / "scorecard.csv")
    summary = json.loads((OUT / "summary.json").read_text())
    assert not set(card["feature"]) & set(LEAKAGE_COLUMNS)
    assert not set(summary["selected_features"]) & set(LEAKAGE_COLUMNS)


def test_target_window_and_excluded_cohorts():
    df = add_target(load_raw())
    bad = df[df[TARGET] == 1]
    assert ((bad["default_date"] - bad["origination_date"]).dt.days <= 366).all()
    s = make_splits(df)
    for k in ("train", "holdout"):
        assert s[k]["cohort_year"].between(2015, 2018).all()
    assert s["oot"]["cohort_year"].between(2021, 2023).all()
    assert not set(s["train"]["loan_id"]) & set(s["holdout"]["loan_id"])


# ------------------------------------------------------------------ points
def test_points_monotonic_in_risk(built):
    """Within every feature, a bin with a higher train bad rate never gets more points."""
    _, _, _, card = built
    for f, g in card.groupby("feature"):
        g = g.sort_values("bad_rate_train")
        assert (np.diff(g["points"].to_numpy()) <= 0).all(), f


def test_numeric_bins_monotonic_bad_rate():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=20000), name="x")
    y = pd.Series((rng.random(20000) < 1 / (1 + np.exp(3 + x))).astype(int))
    t = bin_numeric(x, y).table
    d = np.diff(t["bad_rate"].to_numpy())
    assert (d <= 0).all() or (d >= 0).all()


def test_scaling_anchor_and_pdo():
    sc = Scaling(600, 50, 20)
    assert sc.score_from_log_odds_bad(np.log(1 / 50)) == pytest.approx(600)
    assert sc.score_from_log_odds_bad(np.log(1 / 100)) == pytest.approx(620)  # doubling odds = +PDO


def test_points_sum_matches_model_score(built):
    s, b, m, card = built
    d = s["holdout"]
    pts = score_points(d, card, b)
    exact = Scaling().score_from_log_odds_bad(m.log_odds_bad(woe_transform(d, b, m.features)))
    assert np.abs(pts.to_numpy() - exact).max() <= len(m.features) * 0.5 + 1e-9  # rounding only


# ------------------------------------------------------------------ PSI
def test_psi_identical_is_zero():
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)
    assert psi([0.2, 0.3, 0.5], [20, 30, 50]) == pytest.approx(0.0)  # shares vs counts


def test_psi_known_value_and_symmetry():
    e, a = np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.10, 0.20, 0.30, 0.40])
    expected = np.sum((a - e) * np.log(a / e))
    assert psi(e, a) == pytest.approx(expected)
    assert psi(e, a) == pytest.approx(psi(a, e))
    assert psi(e, a) > 0


def test_psi_handles_empty_bins_and_shape():
    assert np.isfinite(psi([50, 50, 0], [40, 40, 20]))
    with pytest.raises(ValueError):
        psi([1, 2], [1, 2, 3])


def test_traffic_light_thresholds():
    assert traffic_light(0.05) == "green"
    assert traffic_light(0.10) == "amber"
    assert traffic_light(0.2499) == "amber"
    assert traffic_light(0.25) == "red"
