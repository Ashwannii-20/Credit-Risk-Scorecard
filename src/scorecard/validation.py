"""Discrimination, calibration-by-band and stability metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PSI_AMBER, PSI_RED = 0.10, 0.25


def auc_gini(y, risk) -> tuple[float, float]:
    """risk: higher = riskier (e.g. PD, or -score)."""
    auc = roc_auc_score(y, risk)
    return auc, 2 * auc - 1


def ks_stat(y, risk) -> float:
    fpr, tpr, _ = roc_curve(y, risk)
    return float(np.max(tpr - fpr))


def discrimination_table(samples: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for name, (y, score) in samples.items():
        auc, gini = auc_gini(y, -np.asarray(score))
        rows.append({"sample": name, "n": len(y), "n_bad": int(np.sum(y)), "bad_rate": float(np.mean(y)),
                     "auc": auc, "gini": gini, "ks": ks_stat(y, -np.asarray(score))})
    return pd.DataFrame(rows)


def psi(expected, actual, eps: float = 1e-4) -> float:
    """Population Stability Index between two distributions given as counts or shares over
    the same bins: sum((a - e) * ln(a / e)). Empty bins are floored at eps."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape:
        raise ValueError("expected and actual must have the same number of bins")
    e = np.clip(e / e.sum(), eps, None)
    a = np.clip(a / a.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def traffic_light(value: float, amber: float = PSI_AMBER, red: float = PSI_RED) -> str:
    if value < amber:
        return "green"
    return "amber" if value < red else "red"


def psi_table(expected_bins: pd.Series, actual_bins: pd.Series, labels: list) -> pd.DataFrame:
    """Per-bin contribution table. *_bins are bin indices 0..k-1."""
    k = len(labels)
    e = np.bincount(expected_bins, minlength=k).astype(float)
    a = np.bincount(actual_bins, minlength=k).astype(float)
    ep, ap = e / e.sum(), a / a.sum()
    contrib = (np.clip(ap, 1e-4, None) - np.clip(ep, 1e-4, None)) * np.log(np.clip(ap, 1e-4, None) / np.clip(ep, 1e-4, None))
    return pd.DataFrame({"bin": labels, "dev_n": e.astype(int), "dev_pct": ep,
                         "oot_n": a.astype(int), "oot_pct": ap, "psi_contrib": contrib})


def score_band_edges(train_scores: pd.Series, n_bands: int = 10, train_y: pd.Series | None = None,
                     min_bad: int = 0) -> list[float]:
    """Interior cut points from train score quantiles (rounded, de-duplicated).

    If train_y and min_bad are given, adjacent bands are merged (smallest band first) until every
    band holds at least min_bad train defaults, so band default rates are not driven by 1-5
    events. Merging uses train only; holdout/OOT monotonicity remains an independent test.
    """
    q = np.quantile(train_scores, np.linspace(0, 1, n_bands + 1)[1:-1])
    edges = sorted(set(int(round(v)) for v in q))
    if train_y is None or min_bad <= 0:
        return edges
    y = np.asarray(train_y)
    while len(edges) > 1:
        bads = np.bincount(assign_band(train_scores, edges), weights=y, minlength=len(edges) + 1)
        low = np.where(bads < min_bad)[0]
        if not len(low):
            break
        i = low[np.argmin(bads[low])]
        if i == 0:
            del edges[0]
        elif i == len(edges):
            del edges[-1]
        else:  # merge with the neighbour holding fewer defaults
            del edges[i - 1 if bads[i - 1] <= bads[i + 1] else i]
    return edges


def band_labels(edges: list[float]) -> list[str]:
    b = [None, *edges, None]
    out = []
    for lo, hi in zip(b[:-1], b[1:]):
        out.append(f"< {hi}" if lo is None else (f">= {lo}" if hi is None else f"{lo}-{hi - 1}"))
    return out


def assign_band(scores, edges) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), np.asarray(scores), side="right")


def band_default_rates(scores: pd.Series, y: pd.Series, edges: list[float]) -> pd.DataFrame:
    labels = band_labels(edges)
    idx = assign_band(scores, edges)
    df = pd.DataFrame({"band": idx, "y": np.asarray(y)})
    g = df.groupby("band")["y"].agg(n="size", n_bad="sum").reindex(range(len(labels)), fill_value=0)
    g["bad_rate"] = g["n_bad"] / g["n"].replace(0, np.nan)
    g["pct_n"] = g["n"] / g["n"].sum()
    g.insert(0, "score_band", labels)
    return g.reset_index(drop=True)


def is_monotonic_decreasing(rates) -> bool:
    r = pd.Series(rates).dropna().to_numpy()
    return bool(np.all(np.diff(r) <= 0))


def count_reversals(rates) -> int:
    r = pd.Series(rates).dropna().to_numpy()
    return int(np.sum(np.diff(r) > 0))
