"""Validation charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

COLORS = {"train": "#1f4e79", "holdout": "#2e86c1", "oot": "#d35400"}
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def score_distribution(scores: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    allv = np.concatenate([s for _, s in scores.values()])
    bins = np.arange(np.floor(allv.min() / 10) * 10, np.ceil(allv.max() / 10) * 10 + 10, 10)
    y_dev, s_dev = scores["development"]
    ax = axes[0]
    ax.hist(s_dev[y_dev == 0], bins=bins, density=True, alpha=0.6, label="Good", color="#2e86c1")
    ax.hist(s_dev[y_dev == 1], bins=bins, density=True, alpha=0.6, label="Bad (12m default)", color="#c0392b")
    ax.set(title="Development 2015-2018: score by outcome", xlabel="Score", ylabel="Density")
    ax.legend()
    ax = axes[1]
    for name, color in (("development", "#1f4e79"), ("oot", "#d35400")):
        _, s = scores[name]
        ax.hist(s, bins=bins, density=True, histtype="step", linewidth=2, color=color,
                label={"development": "Development 2015-2018", "oot": "Out-of-time 2021-2023"}[name])
    ax.set(title="Population stability: development vs OOT", xlabel="Score", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def roc_ks(samples: dict[str, tuple[np.ndarray, np.ndarray]], metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for name, (y, s) in samples.items():
        fpr, tpr, _ = roc_curve(y, -s)
        m = metrics.set_index("sample").loc[name]
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name}: AUC {m.auc:.3f} / Gini {m.gini:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(title="ROC curve", xlabel="False positive rate (goods)", ylabel="True positive rate (bads)")
    ax.legend(loc="lower right")
    ax = axes[1]
    y, s = samples["oot"]
    y, s = np.asarray(y), np.asarray(s)
    # evaluate at distinct score cut-offs (ties handled the same way as the KS in the tables)
    cum_good, cum_bad, thr = roc_curve(y, -s)
    x = np.array([np.mean(s <= -t) if np.isfinite(t) else 0.0 for t in thr])
    k = int(np.argmax(cum_bad - cum_good))
    ax.plot(x, cum_bad, color="#c0392b", lw=2, label="Cumulative % bads")
    ax.plot(x, cum_good, color="#2e86c1", lw=2, label="Cumulative % goods")
    ax.vlines(x[k], cum_good[k], cum_bad[k], color="black", lw=1.5, linestyles=":")
    ax.annotate(f"KS = {cum_bad[k] - cum_good[k]:.3f}\nscore <= {-thr[k]:.0f}",
                xy=(x[k], (cum_bad[k] + cum_good[k]) / 2), xytext=(x[k] + 0.08, 0.35),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set(title="KS chart, out-of-time 2021-2023", xlabel="Population share (sorted lowest score first)",
           ylabel="Cumulative share")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def band_default_rate(bands: pd.DataFrame, path: Path) -> None:
    """bands: long table with columns sample, score_band, bad_rate."""
    labels = list(dict.fromkeys(bands["score_band"]))
    samples = list(dict.fromkeys(bands["sample"]))
    x = np.arange(len(labels))
    w = 0.8 / len(samples)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, smp in enumerate(samples):
        r = bands[bands["sample"] == smp].set_index("score_band").reindex(labels)["bad_rate"] * 100
        ax.bar(x + (i - (len(samples) - 1) / 2) * w, r, width=w, color=COLORS[smp], label=smp)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set(title="12-month default rate by score band (train score deciles merged to >= 15 train defaults per band)",
           xlabel="Score band", ylabel="12m default rate (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
