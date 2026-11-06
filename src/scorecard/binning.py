"""WOE / IV binning.

Convention: WOE = ln(%good / %bad). Higher WOE = lower risk. A small additive smoothing
(0.5 events) keeps WOE finite in sparse bins.

* Numerics: fine classing on quantiles, then adjacent bins are merged until the bad rate is
  monotonic in the chosen direction and every bin meets a minimum population share and
  minimum bad count (coarse classing). Direction is the one giving the higher IV.
* Categoricals: levels are ordered by bad rate, rare levels absorbed into neighbours, and
  adjacent levels merged (smallest chi-square difference first) down to a maximum number
  of groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMOOTH = 0.5


def _woe_iv(good: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = (good + SMOOTH) / (good.sum() + SMOOTH * len(good))
    b = (bad + SMOOTH) / (bad.sum() + SMOOTH * len(bad))
    woe = np.log(g / b)
    return woe, (g - b) * woe


@dataclass
class FeatureBinning:
    name: str
    kind: str  # "numeric" | "categorical"
    edges: list[float] | None = None  # numeric: interior cut points, bins are (-inf,e1],(e1,e2],...
    groups: list[list[str]] | None = None  # categorical: list of level groups
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    @property
    def labels(self) -> list[str]:
        return list(self.table["bin"])

    def assign(self, x: pd.Series) -> pd.Series:
        """Map raw values to bin index (0..k-1)."""
        if self.kind == "numeric":
            idx = np.searchsorted(np.asarray(self.edges, dtype=float), x.to_numpy(dtype=float), side="left")
            return pd.Series(idx, index=x.index)
        lookup = {lvl: i for i, grp in enumerate(self.groups) for lvl in grp}
        # unseen levels go to the highest-risk group (lowest WOE) as a conservative default
        worst = int(self.table["woe"].idxmin())
        return x.astype(str).map(lookup).fillna(worst).astype(int)

    def transform(self, x: pd.Series) -> pd.Series:
        woe = self.table["woe"].to_numpy()
        return pd.Series(woe[self.assign(x).to_numpy()], index=x.index, name=self.name)


def _numeric_labels(edges: list[float]) -> list[str]:
    bounds = [-np.inf, *edges, np.inf]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if np.isinf(lo):
            out.append(f"<= {hi:g}")
        elif np.isinf(hi):
            out.append(f"> {lo:g}")
        else:
            out.append(f"({lo:g}, {hi:g}]")
    return out


def _summary(name, labels, good, bad) -> pd.DataFrame:
    woe, iv = _woe_iv(good, bad)
    n = good + bad
    return pd.DataFrame(
        {
            "feature": name,
            "bin": labels,
            "n": n.astype(int),
            "pct_n": n / n.sum(),
            "n_bad": bad.astype(int),
            "bad_rate": bad / np.maximum(n, 1),
            "woe": woe,
            "iv": iv,
        }
    )


def _merge_numeric(edges: list[float], good: np.ndarray, bad: np.ndarray, increasing: bool,
                   min_share: float, min_bad: int, max_bins: int):
    edges, good, bad = list(edges), good.astype(float).copy(), bad.astype(float).copy()

    def merge(i):  # merge bin i with bin i+1
        nonlocal edges, good, bad
        good[i] += good[i + 1]
        bad[i] += bad[i + 1]
        good, bad = np.delete(good, i + 1), np.delete(bad, i + 1)
        del edges[i]

    total = good.sum() + bad.sum()
    changed = True
    while changed and len(good) > 1:
        changed = False
        rate = bad / (good + bad)
        # 1) monotonicity violations
        diffs = np.diff(rate)
        viol = np.where(diffs < 0)[0] if increasing else np.where(diffs > 0)[0]
        if len(viol):
            i = viol[np.argmin(np.abs(diffs[viol]))]  # mildest violation first
            merge(i)
            changed = True
            continue
        # 2) size / event constraints and max bins
        n = good + bad
        small = np.where((n / total < min_share) | (bad < min_bad))[0]
        if len(small) or len(good) > max_bins:
            if len(small):
                i = small[np.argmin(n[small])]
            else:  # merge the adjacent pair with closest bad rates
                i = int(np.argmin(np.abs(diffs)))
                merge(i)
                changed = True
                continue
            if i == 0:
                merge(0)
            elif i == len(good) - 1:
                merge(i - 1)
            else:  # merge with neighbour whose bad rate is closest
                left, right = abs(rate[i] - rate[i - 1]), abs(rate[i + 1] - rate[i])
                merge(i - 1 if left <= right else i)
            changed = True
    return edges, good, bad


def bin_numeric(x: pd.Series, y: pd.Series, n_fine: int = 20, min_share: float = 0.05,
                min_bad: int = 10, max_bins: int = 6) -> FeatureBinning:
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_fine + 1)[1:-1]))
    edges = [float(e) for e in qs if e < x.max()]
    idx = np.searchsorted(edges, x.to_numpy(dtype=float), side="left")
    k = len(edges) + 1
    bad = np.bincount(idx, weights=y.to_numpy(), minlength=k)
    good = np.bincount(idx, minlength=k) - bad
    best = None
    for increasing in (True, False):
        e, g, b = _merge_numeric(edges, good, bad, increasing, min_share, min_bad, max_bins)
        iv = _woe_iv(g, b)[1].sum()
        if best is None or iv > best[0]:
            best = (iv, e, g, b)
    _, e, g, b = best
    fb = FeatureBinning(name=x.name, kind="numeric", edges=e)
    fb.table = _summary(x.name, _numeric_labels(e), g, b)
    return fb


def bin_categorical(x: pd.Series, y: pd.Series, min_share: float = 0.05, min_bad: int = 10,
                    max_groups: int = 5) -> FeatureBinning:
    x = x.astype(str)
    stats = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["sum", "count"])
    stats["rate"] = stats["sum"] / stats["count"]
    stats = stats.sort_values("rate")
    groups = [[lvl] for lvl in stats.index]
    bad = stats["sum"].to_numpy(dtype=float)
    n = stats["count"].to_numpy(dtype=float)
    total = n.sum()

    def merge(i):
        nonlocal groups, bad, n
        groups[i] = groups[i] + groups[i + 1]
        del groups[i + 1]
        bad[i] += bad[i + 1]
        n[i] += n[i + 1]
        bad, n = np.delete(bad, i + 1), np.delete(n, i + 1)

    def chi2(i):  # 2x2 chi-square between adjacent groups i, i+1
        obs = np.array([[bad[i], n[i] - bad[i]], [bad[i + 1], n[i + 1] - bad[i + 1]]])
        exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
        return float(((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum())

    while len(groups) > 1:
        small = np.where((n / total < min_share) | (bad < min_bad))[0]
        if len(small):
            i = small[np.argmin(n[small])]
            if i == 0:
                merge(0)
            elif i == len(groups) - 1:
                merge(i - 1)
            else:
                merge(i - 1 if chi2(i - 1) <= chi2(i) else i)
            continue
        if len(groups) > max_groups:
            merge(int(np.argmin([chi2(i) for i in range(len(groups) - 1)])))
            continue
        break
    fb = FeatureBinning(name=x.name, kind="categorical", groups=groups)
    fb.table = _summary(x.name, [", ".join(g) for g in groups], n - bad, bad)
    return fb


def fit_binning(df: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str],
                **kwargs) -> dict[str, FeatureBinning]:
    out = {c: bin_numeric(df[c].rename(c), y, **kwargs) for c in numeric}
    cat_kwargs = {k: v for k, v in kwargs.items() if k in ("min_share", "min_bad")}
    out.update({c: bin_categorical(df[c].rename(c), y, **cat_kwargs) for c in categorical})
    return out


def woe_transform(df: pd.DataFrame, binnings: dict[str, FeatureBinning], features: list[str]) -> pd.DataFrame:
    return pd.DataFrame({f: binnings[f].transform(df[f]) for f in features}, index=df.index)


def iv_table(binnings: dict[str, FeatureBinning]) -> pd.DataFrame:
    rows = [{"feature": k, "type": b.kind, "n_bins": len(b.table), "iv": b.iv} for k, b in binnings.items()]
    t = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    t["strength"] = pd.cut(t["iv"], [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
                           labels=["unpredictive", "weak", "medium", "strong", "suspicious"]).astype(str)
    return t
