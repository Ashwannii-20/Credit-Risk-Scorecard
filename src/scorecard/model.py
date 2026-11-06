"""Feature selection, logistic regression on WOE, and points scaling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .binning import FeatureBinning

# ---------------------------------------------------------------- selection

def select_features(iv: pd.DataFrame, woe_train: pd.DataFrame, iv_min: float = 0.02,
                    corr_max: float = 0.6) -> tuple[list[str], pd.DataFrame]:
    """Keep features with IV > iv_min, then drop the lower-IV member of any pair with
    |Pearson corr of WOE| > corr_max. Returns (selected, decision log)."""
    log = []
    ranked = iv.sort_values("iv", ascending=False)
    kept: list[str] = []
    corr = woe_train.corr().abs()
    for _, r in ranked.iterrows():
        f = r["feature"]
        if r["iv"] <= iv_min:
            log.append({"feature": f, "iv": r["iv"], "decision": "dropped", "reason": f"IV <= {iv_min}"})
            continue
        clash = [k for k in kept if corr.loc[f, k] > corr_max]
        if clash:
            log.append({"feature": f, "iv": r["iv"], "decision": "dropped",
                        "reason": f"abs corr > {corr_max} with {clash[0]} ({corr.loc[f, clash[0]]:.2f})"})
            continue
        kept.append(f)
        log.append({"feature": f, "iv": r["iv"], "decision": "kept", "reason": ""})
    return kept, pd.DataFrame(log)


# ---------------------------------------------------------------- model

@dataclass
class WoeLogit:
    features: list[str]
    result: sm.regression.linear_model.RegressionResultsWrapper | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoeLogit":
        self.result = sm.Logit(y.astype(float), sm.add_constant(X[self.features], has_constant="add")).fit(disp=0)
        return self

    @property
    def intercept(self) -> float:
        return float(self.result.params["const"])

    @property
    def coefs(self) -> pd.Series:
        return self.result.params.drop("const")

    def log_odds_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + X[self.features].to_numpy() @ self.coefs.to_numpy()

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds_bad(X)))

    def summary_table(self) -> pd.DataFrame:
        r = self.result
        return pd.DataFrame({"coef": r.params, "std_err": r.bse, "z": r.tvalues, "p_value": r.pvalues}) \
            .rename_axis("term").reset_index()


def fit_with_sign_check(X: pd.DataFrame, y: pd.Series, features: list[str], p_max: float = 0.05):
    """Fit, then iteratively remove features with the wrong sign (WOE=ln(G/B) => coefficient
    must be negative for log-odds of bad) or p-value > p_max. Returns (model, removal log)."""
    feats, log = list(features), []
    while True:
        m = WoeLogit(feats).fit(X, y)
        c, p = m.coefs, m.result.pvalues.drop("const")
        wrong = c[c >= 0]
        if len(wrong):
            f = wrong.index[0]
            log.append({"feature": f, "reason": f"positive coefficient {c[f]:.3f} (wrong sign)"})
        elif (p > p_max).any():
            f = p.idxmax()
            log.append({"feature": f, "reason": f"p-value {p[f]:.3f} > {p_max}"})
        else:
            return m, pd.DataFrame(log, columns=["feature", "reason"])
        feats.remove(f)


# ---------------------------------------------------------------- scaling

@dataclass(frozen=True)
class Scaling:
    base_score: float = 600.0
    base_odds: float = 50.0  # good:bad
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_from_log_odds_bad(self, log_odds_bad):
        """score = offset + factor * ln(odds good:bad) = offset - factor * log_odds_bad."""
        return self.offset - self.factor * np.asarray(log_odds_bad)


def scorecard_table(model: WoeLogit, binnings: dict[str, FeatureBinning], scaling: Scaling) -> pd.DataFrame:
    """Points per bin. Intercept and offset are spread evenly across features so every
    attribute carries its share and the total equals the scaled score (before rounding)."""
    n = len(model.features)
    rows = []
    for f in model.features:
        b = model.coefs[f]
        t = binnings[f].table
        pts = -(b * t["woe"] + model.intercept / n) * scaling.factor + scaling.offset / n
        for (_, r), p in zip(t.iterrows(), pts):
            rows.append({"feature": f, "bin": r["bin"], "woe": r["woe"], "coef": b,
                         "points_raw": p, "points": int(round(p)),
                         "n_train": r["n"], "bad_rate_train": r["bad_rate"]})
    return pd.DataFrame(rows)


def score_points(df: pd.DataFrame, card: pd.DataFrame, binnings: dict[str, FeatureBinning]) -> pd.Series:
    """Integer score = sum of rounded attribute points (what a scorecard user would compute)."""
    total = pd.Series(0, index=df.index, dtype=int)
    for f, grp in card.groupby("feature", sort=False):
        pts = grp["points"].to_numpy()
        total += pts[binnings[f].assign(df[f]).to_numpy()]
    return total
