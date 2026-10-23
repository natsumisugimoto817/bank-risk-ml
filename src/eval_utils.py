"""Evaluation helpers for binary classifiers: a metrics table and standard plots.

Works for any project with a 0/1 target (credit default, fraud, ...).

Usage in a notebook:
    from eval_utils import metrics_table, plot_eval
    metrics_table(y_te, p_te)                       # p_te = {"EBM": proba, "XGBoost": proba, ...}
    plot_eval(y_te, p_te, colors=COLORS, pos_label="default")
    plot_eval(y_te, p_te, panels=["roc", "calibration"])   # only some plots

Each plot also exists on its own (plot_roc, plot_pr, plot_ks, plot_calibration) and takes an `ax`,
so you can put it into your own figure.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

NEG_COLOR = "#a3a3a3"   # target = 0 (grey)
POS_COLOR = "#e07b72"   # target = 1 (soft red)
GREY = "#8a8985"
DEFAULT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c2185b", "#8d6e63"]


def _as_dict(preds):
    # allow a single array as well as {name: array}
    return preds if isinstance(preds, dict) else {"model": preds}


def _colors(names, colors):
    colors = dict(colors or {})
    for i, n in enumerate(names):
        colors.setdefault(n, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    return colors


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def ks_stat(y, p):
    """KS = max(TPR - FPR) over all thresholds."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics_table(y, preds):
    """One row per model: AUC, Gini, KS, PR-AUC, Brier, plus the baselines to compare with.

    pr_auc_random = positive rate (what a random model gets)
    brier_baseline = p * (1 - p) (predicting the positive rate for everyone)
    """
    y = np.asarray(y)
    base = y.mean()
    rows = {}
    for m, p in _as_dict(preds).items():
        auc = roc_auc_score(y, p)
        rows[m] = {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_stat(y, p),
                   "PR-AUC": average_precision_score(y, p), "Brier": brier_score_loss(y, p)}
    out = pd.DataFrame(rows).T
    out["pr_auc_random"] = base
    out["brier_baseline"] = base * (1 - base)
    return out


# ----------------------------------------------------------------------------
# single plots (each draws on one ax)
# ----------------------------------------------------------------------------
def plot_roc(ax, y, preds, colors=None):
    """ROC: true positive rate vs false positive rate. Diagonal = random. AUC = area under."""
    preds = _as_dict(preds)
    colors = _colors(list(preds), colors)
    for m, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr * 100, tpr * 100, color=colors[m], label=f"{m}  (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 100], [0, 100], color=GREY, ls="--", lw=1)
    ax.set(title="ROC curve", xlabel="False positive rate (%)", ylabel="True positive rate (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)


def plot_pr(ax, y, preds, colors=None, pos_label="positive"):
    """Precision vs recall for every threshold. Dashed = random model (precision = positive rate)."""
    preds = _as_dict(preds)
    colors = _colors(list(preds), colors)
    base = np.mean(y)
    for m, p in preds.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec * 100, prec * 100, color=colors[m], label=f"{m}  (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(base * 100, color=GREY, ls="--", lw=1)
    ax.text(50, base * 100 - 1, f"random model = {base:.1%} ({pos_label} rate)", ha="center", va="top",
            fontsize=8, color=GREY)
    ax.set(title="Precision-recall curve", xlabel=f"Recall: share of {pos_label} caught (%)",
           ylabel=f"Precision: share of flagged that are {pos_label} (%)", ylim=(0, 100))
    ax.legend(frameon=False, loc="upper right", fontsize=8)


def plot_ks(ax, y, p, model_name="model", pos_label="positive", neg_label="negative"):
    """Go down the list from highest to lowest score.
    Red = share of all 1s found so far, grey = share of all 0s found so far. KS = biggest gap."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p))
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
    cum_neg = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    share = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    gap = cum_pos - cum_neg
    i = int(np.argmax(gap))

    ax.plot(share * 100, cum_pos * 100, color=POS_COLOR, label=f"1 ({pos_label}) found")
    ax.plot(share * 100, cum_neg * 100, color=NEG_COLOR, label=f"0 ({neg_label}) found")
    ax.vlines(share[i] * 100, cum_neg[i] * 100, cum_pos[i] * 100, color="black", lw=1.5)
    ax.annotate(f"KS = {gap[i]:.2f}\nat top {share[i] * 100:.0f}%",
                (share[i] * 100, (cum_pos[i] + cum_neg[i]) / 2 * 100),
                xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    ax.plot([0, 100], [0, 100], color=GREY, ls="--", lw=1)
    ax.set(title=f"KS — {model_name}", xlabel="Checked, highest score first (%)",
           ylabel="Share of the class found (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)


def plot_calibration(ax, y, preds, colors=None, n_bins=10, show_brier=True):
    """Split by predicted probability into n_bins groups (equal size).
    x = average predicted, y = actual rate. On the diagonal = probabilities are right.
    The box compares Brier with p * (1 - p) (predicting the average for everyone)."""
    preds = _as_dict(preds)
    colors = _colors(list(preds), colors)
    base = np.mean(y)
    lines = [f"baseline p(1-p) = {base * (1 - base):.4f}"]
    hi = 0
    for m, p in preds.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred * 100, frac * 100, color=colors[m], marker="o", markersize=4, label=m)
        hi = max(hi, frac.max(), mean_pred.max())
        lines.append(f"{m}: Brier {brier_score_loss(y, p):.4f}")
    lim = hi * 100 * 1.1
    ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1)
    ax.set(title="Calibration", xlabel="Predicted rate (%)", ylabel="Actual rate (%)",
           xlim=(0, lim), ylim=(0, lim))
    if show_brier:
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.4"})
    ax.legend(frameon=False, loc="lower right", fontsize=8)


# ----------------------------------------------------------------------------
# all in one row
# ----------------------------------------------------------------------------
PANELS = ("roc", "pr", "ks", "calibration")


def plot_eval(y, preds, colors=None, panels=PANELS, ks_model=None, pos_label="positive",
              neg_label="negative", n_bins=10, ncols=None, title=None):
    """Draw the chosen plots side by side.

    y:         true labels (0 / 1)
    preds:     {model name: predicted probability} (or a single array)
    colors:    {model name: color}; missing ones get default colors
    panels:    any of "roc", "pr", "ks", "calibration", in the order you want
    ks_model:  model used for the KS plot (default: the first one)
    pos_label / neg_label: names used in labels, e.g. "default" / "paid", "fraud" / "normal"
    ncols:     plots per row (default: all in one row)
    """
    y = np.asarray(y)
    preds = _as_dict(preds)
    names = list(preds)
    colors = _colors(names, colors)
    ks_model = ks_model or names[0]

    panels = list(panels)
    ncols = ncols or len(panels)
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    for name, ax in zip(panels, axes):
        if name == "roc":
            plot_roc(ax, y, preds, colors)
        elif name == "pr":
            plot_pr(ax, y, preds, colors, pos_label=pos_label)
        elif name == "ks":
            plot_ks(ax, y, preds[ks_model], model_name=ks_model, pos_label=pos_label, neg_label=neg_label)
        elif name == "calibration":
            plot_calibration(ax, y, preds, colors, n_bins=n_bins)
        else:
            raise ValueError(f"unknown panel: {name}. Use any of {PANELS}")

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()
