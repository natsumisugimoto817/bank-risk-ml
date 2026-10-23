"""Small helpers for EDA: data quality checks and distribution plots.

Usage in a notebook:
    from eda_utils import (check_duplicates, missing_table, missing_vs_target, psi, psi_table,
                           plot_boxplots, plot_numeric_by_target, plot_categorical_by_target,
                           plot_target_rate_by_bin, corr_matrix, high_corr_pairs, plot_corr_matrix)
"""
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# colors for target = 0 / 1 (used in every plot)
NEG_COLOR = "#a3a3a3"   # target = 0 (grey)
POS_COLOR = "#e07b72"   # target = 1 (soft red)
BOX_COLOR = "#7f9fbf"   # boxplots (no target)


# ----------------------------------------------------------------------------
# data quality
# ----------------------------------------------------------------------------
def check_duplicates(df, subset=None, key=None, verbose=True):
    """Count duplicate rows.

    subset: columns used to judge "same row" (default: all columns)
    key:    ID column, e.g. "loan_id". If given, also count IDs that appear more than once.
    Returns a dict so the numbers can be reused later.
    """
    n_dup = int(df.duplicated(subset=subset).sum())
    out = {"rows": len(df), "duplicate_rows": n_dup, "duplicate_share": n_dup / len(df) if len(df) else 0.0}

    if key is not None:
        out["duplicate_keys"] = int(df[key].duplicated().sum())

    if verbose:
        print(f"rows: {out['rows']:,}   duplicate rows: {n_dup:,} ({out['duplicate_share']:.2%})")
        if key is not None:
            print(f"duplicate {key}: {out['duplicate_keys']:,}")
    return out


def _is_missing(s, missing_tokens):
    # NaN, plus text values we use as "missing" (e.g. "Missing", "")
    miss = s.isna()
    if missing_tokens and not pd.api.types.is_numeric_dtype(s):
        miss = miss | s.astype(str).str.strip().isin(missing_tokens)
    return miss


def missing_table(df, cols=None, missing_tokens=("Missing", ""), only_missing=False):
    """Missing rate per column, highest first.

    missing_tokens: text values that also mean "missing" (we filled categorical NaN with "Missing").
    only_missing:   show only columns that have at least one missing value.
    """
    cols = list(df.columns) if cols is None else list(cols)
    rows = []
    for c in cols:
        miss = _is_missing(df[c], missing_tokens)
        rows.append({"column": c, "dtype": str(df[c].dtype),
                     "n_missing": int(miss.sum()), "missing_%": miss.mean() * 100})
    out = pd.DataFrame(rows).set_index("column").sort_values("missing_%", ascending=False)
    if only_missing:
        out = out[out["n_missing"] > 0]
    return out


def missing_vs_target(df, target, cols=None, missing_tokens=("Missing", "")):
    """Compare the target rate when a column is missing vs present.

    A big gap means "missing" itself carries information (keep a missing flag, don't just fill).
    """
    cols = [c for c in (df.columns if cols is None else cols) if c != target]
    rows = []
    for c in cols:
        miss = _is_missing(df[c], missing_tokens)
        if not miss.any():
            continue
        rows.append({"column": c, "missing_%": miss.mean() * 100,
                     "target_if_missing": df.loc[miss, target].mean(),
                     "target_if_present": df.loc[~miss, target].mean()})
    if not rows:
        return pd.DataFrame(columns=["missing_%", "target_if_missing", "target_if_present", "gap"])
    out = pd.DataFrame(rows).set_index("column")
    out["gap"] = out["target_if_missing"] - out["target_if_present"]
    return out.sort_values("missing_%", ascending=False)


# ----------------------------------------------------------------------------
# PSI
# ----------------------------------------------------------------------------
def _quantile_edges(x, bins):
    # inner bin edges from quantiles; duplicates removed (discrete features have repeated values)
    inner = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(inner)


def _bin_share(x, inner_edges, eps):
    # side="left": a value equal to an edge goes to the lower bin. This keeps a big pile of
    # one value (e.g. 90% of rows have pub_rec = 0) in its own bin instead of mixing it with 1, 2, ...
    idx = np.searchsorted(inner_edges, x, side="left")
    counts = np.bincount(idx, minlength=len(inner_edges) + 1)
    share = counts / counts.sum()
    return np.where(share == 0, eps, share)   # empty bin -> tiny value, so log() works


def psi(expected, actual, bins=10, eps=1e-6, include_missing=True, edges=None):
    """Population Stability Index between two samples of one numeric variable.

    expected: reference sample (e.g. train), actual: new sample (e.g. test / this month)
    bins:     number of quantile bins, made from `expected`
    eps:      value used for empty bins (instead of 0)
    include_missing: treat NaN as its own bin (missing share can also drift)
    edges:    inner bin edges to use instead of quantiles of `expected`

    Rule of thumb: < 0.1 stable, 0.1-0.25 some shift, > 0.25 big shift.
    """
    e = pd.Series(expected, dtype=float)
    a = pd.Series(actual, dtype=float)
    e_ok, a_ok = e.dropna().to_numpy(), a.dropna().to_numpy()
    if len(e_ok) == 0 or len(a_ok) == 0:
        return np.nan
    if edges is None:
        edges = _quantile_edges(e_ok, bins)

    e_share = _bin_share(e_ok, edges, eps) * (len(e_ok) / len(e))
    a_share = _bin_share(a_ok, edges, eps) * (len(a_ok) / len(a))
    if include_missing:
        e_share = np.append(e_share, max(e.isna().mean(), eps))
        a_share = np.append(a_share, max(a.isna().mean(), eps))
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def psi_table(ref, new, cols, bins=10, eps=1e-6, include_missing=True):
    """PSI for many columns at once (e.g. ref = train, new = test), highest first."""
    rows = []
    for c in cols:
        v = psi(ref[c], new[c], bins=bins, eps=eps, include_missing=include_missing)
        rows.append({"column": c, "psi": v})
    out = pd.DataFrame(rows).set_index("column").sort_values("psi", ascending=False)
    out["level"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "some shift", "big shift"])
    return out


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def _clip_range(clip):
    # 0.01 -> (0.01, 0.99); (0.01, 0.95) stays; None / 0 -> no clipping
    if not clip:
        return None
    if isinstance(clip, (int, float)):
        return (clip, 1 - clip)
    return tuple(clip)


def _clip_label(clip):
    if clip is None:
        return "no clipping"
    lo, hi = clip
    return f"clipped at p{lo * 100:g}-p{hi * 100:g}"


def _grid(n, ncols, width=3.2, height=2.6):
    # make a grid of n small plots, ncols per row, and hide the unused ones
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _numeric_cols(df, cols, exclude=()):
    cols = df.columns if cols is None else cols
    return [c for c in cols if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _categorical_cols(df, cols, exclude=()):
    cols = df.columns if cols is None else cols
    return [c for c in cols if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]


def _is_skewed(s, threshold=2.0):
    # skewed, non-negative, and not a small count (counts like pub_rec = 0, 1, 2 don't need log)
    return s.min() >= 0 and s.nunique() > 20 and s.skew() > threshold


def plot_boxplots(df, cols=None, target=None, ncols=5, log="auto", showfliers=True, exclude=(),
                  title="Boxplots (numeric features)"):
    """One boxplot per numeric column, ncols per row.

    target:  if given, draw two boxes per feature (0 = grey, 1 = red) to compare the classes.
    log:     "auto" = log-like axis (symlog) only for skewed, non-negative columns;
             True = all columns; False = never. Skewed money columns otherwise look like
             "everything is an outlier".
    exclude: columns to skip, e.g. ID columns.
    """
    excl = list(exclude) + ([target] if target is not None else [])
    cols = _numeric_cols(df, cols, exclude=excl)
    fig, axes = _grid(len(cols), ncols, height=2.4 if target is None else 2.6)
    flier = {"marker": ".", "markersize": 2, "alpha": 0.3}
    for c, ax in zip(cols, axes):
        s = df[c]
        if target is None:
            data, colors, labels = [s.dropna()], [BOX_COLOR], None
        else:
            y = df[target]
            data, colors, labels = [s[y == 0].dropna(), s[y == 1].dropna()], [NEG_COLOR, POS_COLOR], ["0", "1"]
        bp = ax.boxplot(data, vert=False, widths=0.5, showfliers=showfliers, patch_artist=True,
                        medianprops={"color": "black"}, flierprops=flier)
        for box, col in zip(bp["boxes"], colors):
            box.set_facecolor(col)
            box.set_alpha(0.7)
        if labels:
            ax.set_yticks([1, 2], labels, fontsize=8)
        else:
            ax.set_yticks([])

        use_log = (log is True) or (log == "auto" and _is_skewed(s.dropna()))
        name = c
        if use_log:
            if (s.dropna() > 0).all():
                ax.set_xscale("log")
            else:
                # zeros present: symlog is linear near 0 and log further out
                ax.set_xscale("symlog", linthresh=1 if s.median() >= 1 else 0.01)
                ax.set_xlim(left=0)
            name += " (log scale)"
        ax.set_title(name, fontsize=10)
    suffix = f"  by {target} (grey = 0, red = 1)" if target is not None else ""
    fig.suptitle(title + suffix, y=1.0)
    plt.tight_layout()
    plt.show()


def plot_numeric_by_target(df, target, cols=None, ncols=5, bins=30, clip=0.01, density=True,
                           show_iv=True, iv_bins=10, eps=1e-6):
    """Histogram of each numeric column, split by target (0 = grey, 1 = red).

    clip:    cut the long tails before plotting. 0.01 = 1st-99th percentile,
             (0.01, 0.95) = different lower / upper cut, None = no cut. Shown in the title.
    density: True = compare shapes (each class sums to 1). The positive class is usually small,
             so with raw counts it would be almost invisible.
    show_iv: PSI between the 0-group and the 1-group, shown in each title.
             (Same formula as PSI; in credit scoring this is called Information Value, IV.)
             Calculated on the raw values (before clipping) with quantile bins; empty bins -> eps.
             Rule of thumb for IV: < 0.02 not useful, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong,
             > 0.5 suspiciously strong (check for leakage).
    """
    clip = _clip_range(clip)
    cols = _numeric_cols(df, cols, exclude=[target])
    fig, axes = _grid(len(cols), ncols)
    y = df[target]
    for c, ax in zip(cols, axes):
        raw = df[c]
        s = raw
        if clip is not None:
            s = raw.clip(raw.quantile(clip[0]), raw.quantile(clip[1]))
        s0, s1 = s[y == 0].dropna(), s[y == 1].dropna()
        edges = np.histogram_bin_edges(pd.concat([s0, s1]), bins=bins)
        # grey = filled bars, red = outline on top, so both stay readable where they overlap
        ax.hist(s0, bins=edges, density=density, color=NEG_COLOR, alpha=0.6)
        ax.hist(s1, bins=edges, density=density, color=POS_COLOR, histtype="step", linewidth=1.8)

        name = c
        if show_iv:
            # same bins for both groups, made from all rows
            inner = _quantile_edges(raw.dropna().to_numpy(), iv_bins)
            iv = psi(raw[y == 0], raw[y == 1], eps=eps, edges=inner)
            name += f"  |  IV {iv:.2f}"
        ax.set_title(name, fontsize=10)
        ax.set_yticks([])
    fig.legend(handles=[Patch(color=NEG_COLOR, label=f"{target} = 0"), Patch(color=POS_COLOR, label=f"{target} = 1")],
               loc="upper right", ncol=2, frameon=False)
    title = f"Numeric features by {target}  ({_clip_label(clip)}" + (", density" if density else "") + ")"
    fig.suptitle(title, y=1.0)
    plt.tight_layout()
    plt.show()


def plot_categorical_by_target(df, target, cols=None, ncols=5, top_n=8, show_rate=True):
    """Bar chart per categorical column, split by target (0 = grey, 1 = red).

    Bars show the share of each class in each category (each class sums to 100%),
    so the small positive class is easy to compare.
    top_n:     keep the most common categories, put the rest into "Other".
    show_rate: write the target rate (% of 1) above each category.
    """
    cols = _categorical_cols(df, cols, exclude=[target])
    fig, axes = _grid(len(cols), ncols, height=3.0)
    y = df[target]
    for c, ax in zip(cols, axes):
        s = df[c].astype(str)
        top = s.value_counts().index[:top_n]
        s = s.where(s.isin(top), "Other")
        order = s.value_counts().index

        share0 = s[y == 0].value_counts(normalize=True).reindex(order, fill_value=0) * 100
        share1 = s[y == 1].value_counts(normalize=True).reindex(order, fill_value=0) * 100
        x = np.arange(len(order))
        ax.bar(x - 0.2, share0.values, width=0.4, color=NEG_COLOR)
        ax.bar(x + 0.2, share1.values, width=0.4, color=POS_COLOR)
        ax.set_xticks(x, [str(o)[:12] for o in order], rotation=45, ha="right", fontsize=8)
        ax.set_title(c, fontsize=10)
        ax.set_ylabel("% of class", fontsize=8)

        if show_rate:
            rate = y.groupby(s).mean().reindex(order) * 100
            top_y = np.maximum(share0.values, share1.values)
            for xi, (r, h) in enumerate(zip(rate.values, top_y)):
                ax.text(xi, h, f"{r:.0f}%", ha="center", va="bottom", fontsize=7, color=POS_COLOR)
            ax.set_ylim(0, top_y.max() * 1.2)   # room for the labels
    fig.legend(handles=[Patch(color=NEG_COLOR, label=f"{target} = 0"), Patch(color=POS_COLOR, label=f"{target} = 1")],
               loc="upper right", ncol=2, frameon=False)
    fig.suptitle(f"Categorical features by {target}" + ("  (red number = % of 1 in category)" if show_rate else ""), y=1.0)
    plt.tight_layout()
    plt.show()


def _short_num(v, _pos=None):
    # 25000 -> "25k", 1500000 -> "1.5M", 0.2 -> "0.2"
    if abs(v) >= 1e6:
        return f"{v / 1e6:g}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _wilson(k, n, z=1.96):
    # 95% interval for a rate k / n (works better than +-1.96*se when n is small or rate is near 0)
    n = np.asarray(n, dtype=float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def plot_target_rate_by_bin(df, target, cols=None, ncols=5, bins=10, ci=True, log="auto", min_count=50):
    """P(target = 1 | X): target rate in each bin of X, one small plot per numeric column.

    bins: number of quantile bins (each bin has about the same number of rows).
          Columns with few values (counts like pub_rec, up to 50 whole-number values) use one bin per value.
    min_count: for those few-value columns, rare high values are merged into the last value
          with at least min_count rows (so the last point means "this value or more").
    ci:   shaded 95% interval (Wilson). Wide band = few rows in that bin, don't over-read it.
    log:  "auto" = log x-axis for skewed money columns, True / False to force.
    Dashed line = overall target rate. Points above it = riskier than average.
    If the column has missing values, the target rate for "missing" is shown in the title.
    """
    cols = _numeric_cols(df, cols, exclude=[target])
    fig, axes = _grid(len(cols), ncols)
    y = df[target]
    overall = y.mean() * 100
    for c, ax in zip(cols, axes):
        s = df[c]
        ok = s.notna()
        x, yy = s[ok], y[ok]

        name = c
        n_unique = x.nunique()
        is_count = n_unique <= 50 and bool((x % 1 == 0).all())   # whole numbers like pub_rec, inq_last_6mths
        if n_unique <= max(bins, 20) or is_count:
            # one bin per value; merge rare high values (e.g. pub_rec = 4, 5, ...) into the top value
            vc = x.value_counts()
            kept = vc.index[vc >= min_count]
            if len(kept) and x.max() > kept.max():
                x = x.clip(upper=kept.max())
                name += f" (top = {kept.max():g}+)"
            group = x
        else:
            inner = _quantile_edges(x.to_numpy(), bins)
            group = pd.Series(np.searchsorted(inner, x.to_numpy(), side="left"), index=x.index)
        g = pd.DataFrame({"x": x, "y": yy, "g": group}).groupby("g")
        stat = g.agg(x_mid=("x", "median"), n=("y", "size"), k=("y", "sum")).sort_values("x_mid")
        rate = stat["k"] / stat["n"] * 100

        if ci:
            lo, hi = _wilson(stat["k"].to_numpy(), stat["n"].to_numpy())
            ax.fill_between(stat["x_mid"], lo * 100, hi * 100, color=POS_COLOR, alpha=0.2, lw=0)
        ax.plot(stat["x_mid"], rate, color=POS_COLOR, marker="o", markersize=3)
        ax.axhline(overall, color=NEG_COLOR, ls="--", lw=1)

        use_log = (log is True) or (log == "auto" and _is_skewed(x))
        if use_log and (stat["x_mid"] > 0).all():
            ax.set_xscale("log")
            # ticks at 1, 2, 5 x 10^k with short labels (20k, 50k, 100k)
            ax.xaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_short_num))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            name += " (log x)"
        if (~ok).any():
            name += f"  |  missing: {y[~ok].mean() * 100:.0f}%"
        ax.set_title(name, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(f"{target} rate (%)", fontsize=8)
    fig.suptitle(f"{target} rate by bin of each feature  (dashed = overall {overall:.1f}%, band = 95% CI)", y=1.0)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# correlation
# ----------------------------------------------------------------------------
def corr_matrix(df, cols=None, method="spearman"):
    """Correlation matrix of numeric columns.

    method: "spearman" (rank-based, default) is safer for skewed money columns and outliers;
            "pearson" measures straight-line relation only.
    Categorical columns are skipped.
    """
    cols = _numeric_cols(df, cols)
    return df[cols].corr(method=method)


def high_corr_pairs(df, cols=None, threshold=0.8, method="spearman"):
    """Pairs of numeric columns with |correlation| >= threshold, strongest first.

    Strongly correlated features carry similar information: their importance gets split,
    and in linear models their coefficients become unstable. Consider grouping them or keeping one.
    """
    corr = corr_matrix(df, cols, method)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))   # each pair once
    pairs = upper.stack().rename("corr").reset_index()
    pairs.columns = ["feature_1", "feature_2", "corr"]
    pairs["abs_corr"] = pairs["corr"].abs()
    out = pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False)
    if out.empty:
        print(f"no pairs with |{method} corr| >= {threshold}")
    return out.drop(columns="abs_corr").reset_index(drop=True)


def plot_corr_matrix(df, cols=None, method="spearman", threshold=0.8, annot="auto", triangle=True,
                     figsize=None, title=None):
    """Heatmap of the correlation matrix (red = positive, blue = negative).

    threshold: cells with |corr| >= threshold get a black frame, so the high pairs stand out.
    annot:     write the numbers in the cells. "auto" = only when there are 20 columns or fewer.
    triangle:  show only the lower triangle (the matrix is symmetric).
    """
    corr = corr_matrix(df, cols, method)
    n = len(corr)
    show_num = (n <= 20) if annot == "auto" else bool(annot)
    data = corr.to_numpy().copy()
    if triangle:
        data[np.triu(np.ones((n, n), dtype=bool), k=1)] = np.nan

    size = figsize or (max(6, 0.5 * n + 2), max(5, 0.45 * n + 1.5))
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), corr.columns, fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label=f"{method} correlation")

    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if np.isnan(v) or i == j:
                continue
            if show_num:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) >= 0.6 else "black")
            if threshold is not None and abs(v) >= threshold:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.5))

    ax.set_title(title or f"{method.capitalize()} correlation (framed: |corr| >= {threshold})")
    plt.tight_layout()
    plt.show()
