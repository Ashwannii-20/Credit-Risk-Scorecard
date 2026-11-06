"""End-to-end run: data -> binning -> selection -> model -> scorecard -> validation -> outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .binning import fit_binning, iv_table, woe_transform
from .data import (CATEGORICAL_FEATURES, DEFAULT_DATA_PATH, LEAKAGE_COLUMNS, NUMERIC_FEATURES,
                   PROJECT_ROOT, TARGET, add_target, cohort_summary, load_raw, make_splits)
from .model import Scaling, fit_with_sign_check, score_points, scorecard_table, select_features
from .validation import (band_default_rates, count_reversals, discrimination_table,
                         is_monotonic_decreasing, psi, psi_table, score_band_edges, traffic_light)

TOP_N_CSI = 5
MIN_BAD_PER_BAND = 15


def run(data_path: Path = DEFAULT_DATA_PATH, out_dir: Path = PROJECT_ROOT / "outputs") -> dict:
    out_dir = Path(out_dir)
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # ---- data, target, exclusions, splits
    df = add_target(load_raw(data_path))
    cohort_summary(df).to_csv(out_dir / "cohort_default_rates.csv", index=False)
    splits = make_splits(df)
    train, holdout, oot = splits["train"], splits["holdout"], splits["oot"]
    dev = pd.concat([train, holdout])

    candidates = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    assert not set(candidates) & set(LEAKAGE_COLUMNS), "leakage column in candidate features"

    # ---- binning on train only
    binnings = fit_binning(train, train[TARGET], list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES))
    iv = iv_table(binnings)
    iv.to_csv(out_dir / "iv_table.csv", index=False)
    pd.concat([b.table for b in binnings.values()]).to_csv(out_dir / "woe_bins.csv", index=False)

    # ---- selection
    woe_train = woe_transform(train, binnings, candidates)
    selected, sel_log = select_features(iv, woe_train, iv_min=0.02, corr_max=0.6)
    model, sign_log = fit_with_sign_check(woe_train, train[TARGET], selected, p_max=0.05)
    sel_log = sel_log.merge(sign_log.rename(columns={"reason": "model_reason"}), on="feature", how="left")
    sel_log.loc[sel_log["model_reason"].notna(), "decision"] = "dropped"
    sel_log["reason"] = sel_log["reason"].where(sel_log["model_reason"].isna(), sel_log["model_reason"])
    sel_log.drop(columns="model_reason").to_csv(out_dir / "feature_selection.csv", index=False)
    model.summary_table().to_csv(out_dir / "model_coefficients.csv", index=False)
    woe_train[model.features].corr().round(3).to_csv(out_dir / "woe_correlation_final.csv")

    # ---- scorecard
    scaling = Scaling(base_score=600, base_odds=50, pdo=20)
    card = scorecard_table(model, binnings, scaling)
    card.to_csv(out_dir / "scorecard.csv", index=False)

    scored = {}
    for name, d in (("train", train), ("holdout", holdout), ("oot", oot)):
        s = score_points(d, card, binnings)
        pd_hat = model.predict_pd(woe_transform(d, binnings, model.features))
        scored[name] = pd.DataFrame({"loan_id": d["loan_id"], "cohort_year": d["cohort_year"],
                                     "sample": name, "score": s.to_numpy(), "pd_model": pd_hat,
                                     TARGET: d[TARGET].to_numpy()})
    scores_all = pd.concat(scored.values(), ignore_index=True)
    scores_all.to_csv(out_dir / "scored_loans.csv", index=False)

    # ---- discrimination
    disc = discrimination_table({k: (v[TARGET].to_numpy(), v["score"].to_numpy()) for k, v in scored.items()})
    disc["mean_pd_model"] = [scored[k]["pd_model"].mean() for k in disc["sample"]]
    disc.to_csv(out_dir / "discrimination.csv", index=False)

    # ---- challenger: initial_rating only (is the rest of the card adding anything out of sample?)
    from .model import WoeLogit
    from .validation import auc_gini, ks_stat
    bench = WoeLogit(["initial_rating"]).fit(woe_train, train[TARGET])
    brows = []
    for name, d in (("train", train), ("holdout", holdout), ("oot", oot)):
        p_full = scored[name]["pd_model"].to_numpy()
        p_b = bench.predict_pd(woe_transform(d, binnings, ["initial_rating"]))
        for label, pr in (("final scorecard (model PD)", p_full), ("initial_rating only", p_b)):
            auc, gini = auc_gini(d[TARGET], pr)
            brows.append({"sample": name, "model": label, "auc": auc, "gini": gini, "ks": ks_stat(d[TARGET], pr)})
    pd.DataFrame(brows).to_csv(out_dir / "benchmark_rating_only.csv", index=False)

    # ---- attribute back-test: does each bin's risk ranking hold in holdout and OOT?
    bt = []
    for f in model.features:
        for name, d in (("train", train), ("holdout", holdout), ("oot", oot)):
            idx = binnings[f].assign(d[f])
            g = d.groupby(idx.to_numpy())[TARGET].agg(n="size", n_bad="sum", bad_rate="mean")
            g = g.reindex(range(len(binnings[f].labels)))
            for i, r in g.iterrows():
                bt.append({"feature": f, "bin": binnings[f].labels[i], "sample": name,
                           "n": r["n"], "n_bad": r["n_bad"], "bad_rate": r["bad_rate"]})
    pd.DataFrame(bt).to_csv(out_dir / "attribute_backtest.csv", index=False)

    # ---- score bands (train deciles), monotonicity
    edges = score_band_edges(scored["train"]["score"], 10, scored["train"][TARGET], min_bad=MIN_BAD_PER_BAND)
    decile_edges = score_band_edges(scored["train"]["score"], 10)
    raw = []
    for name, v in scored.items():
        b = band_default_rates(v["score"], v[TARGET], decile_edges)
        b.insert(0, "sample", name)
        raw.append(b)
    pd.concat(raw).to_csv(out_dir / "score_decile_default_rates_unmerged.csv", index=False)
    band_rows, mono_rows = [], []
    for name, v in scored.items():
        b = band_default_rates(v["score"], v[TARGET], edges)
        b.insert(0, "sample", name)
        band_rows.append(b)
        mono_rows.append({"sample": name, "monotonic_decreasing": is_monotonic_decreasing(b["bad_rate"]),
                          "n_reversals": count_reversals(b["bad_rate"])})
    bands = pd.concat(band_rows, ignore_index=True)
    bands.to_csv(out_dir / "score_band_default_rates.csv", index=False)
    mono = pd.DataFrame(mono_rows)
    mono.to_csv(out_dir / "score_band_monotonicity.csv", index=False)

    # ---- stability: PSI of score bands, CSI of top features (dev = train+holdout vs OOT)
    from .validation import assign_band, band_labels
    dev_scores = pd.concat([scored["train"], scored["holdout"]])["score"]
    psi_detail = psi_table(assign_band(dev_scores, edges), assign_band(scored["oot"]["score"], edges),
                           band_labels(edges))
    psi_detail.to_csv(out_dir / "psi_score_detail.csv", index=False)
    score_psi = psi(psi_detail["dev_n"], psi_detail["oot_n"])

    stab = [{"variable": "score", "type": "PSI", "value": score_psi, "rag": traffic_light(score_psi),
             "in_model": True}]
    # CSI for every model feature plus the highest-IV candidates (TOP_N_CSI by IV overall)
    top = list(dict.fromkeys(list(model.features) + list(iv["feature"][:TOP_N_CSI])))
    csi_detail = []
    for f in top:
        t = psi_table(binnings[f].assign(dev[f]).to_numpy(), binnings[f].assign(oot[f]).to_numpy(),
                      binnings[f].labels)
        t.insert(0, "feature", f)
        csi_detail.append(t)
        v = psi(t["dev_n"], t["oot_n"])
        stab.append({"variable": f, "type": "CSI", "value": v, "rag": traffic_light(v),
                     "in_model": f in model.features})
    stability = pd.DataFrame(stab)
    stability.to_csv(out_dir / "stability_psi_csi.csv", index=False)
    pd.concat(csi_detail).to_csv(out_dir / "csi_feature_detail.csv", index=False)

    # ---- charts
    plots.score_distribution(
        {"development": (dev[TARGET].to_numpy(), dev_scores.to_numpy()),
         "oot": (scored["oot"][TARGET].to_numpy(), scored["oot"]["score"].to_numpy())},
        chart_dir / "score_distribution.png")
    plots.roc_ks({k: (v[TARGET].to_numpy(), v["score"].to_numpy()) for k, v in scored.items()}, disc,
                 chart_dir / "roc_ks.png")
    plots.band_default_rate(bands, chart_dir / "band_default_rate.png")

    summary = {
        "n_train": len(train), "n_holdout": len(holdout), "n_oot": len(oot),
        "excluded_2019_2020": int(df["cohort_year"].isin([2019, 2020]).sum()),
        "selected_features": model.features,
        "scaling": {"base_score": 600, "base_odds": 50, "pdo": 20,
                    "factor": scaling.factor, "offset": scaling.offset},
        "discrimination": disc.set_index("sample")[["auc", "gini", "ks", "bad_rate"]].to_dict("index"),
        "score_psi": score_psi,
        "band_edges": edges,
        "decile_edges_unmerged": decile_edges,
        "score_range": [int(scores_all["score"].min()), int(scores_all["score"].max())],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs")
    a = p.parse_args()
    s = run(a.data, a.out)
    print(json.dumps({k: s[k] for k in ("selected_features", "discrimination", "score_psi")}, indent=2,
                     default=float))


if __name__ == "__main__":
    main()
