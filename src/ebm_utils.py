"""Plots for Explainable Boosting Machines (interpret's ExplainableBoostingClassifier).

Usage in a notebook:
    from ebm_utils import plot_shapes, plot_interactions, plot_shape_vs_rate

    plot_shapes(ebm, features=imp_ebm.index[:6], X_ref=X_tr)        # main effects
    plot_interactions(ebm, top_n=3, X_ref=X_tr, hide_uncertain=True)  # pairwise terms, noise greyed out
    plot_shape_vs_rate(ebm, X_tr, y_tr, features=imp_ebm.index[:6])  # EBM vs one-variable view
    plot_feature_report(ebm, X_tr, y_tr, "loan_amnt", target_name="default")   # one feature, 3 views

    # same y-axis for every feature (compare the size of effects)
    ylim = shape_ylim(ebm, X_ref=X_tr)
    plot_feature_report(ebm, X_tr, y_tr, "fico", target_name="default", ylim=ylim)

Shape values are contributions to the log-odds. EBM centers every term, so 0 = average:
above 0 = riskier than average, below 0 = safer.
"""
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

EBM_COLOR = "#2a78d6"
RATE_COLOR = "#e07b72"   # one-variable view (same soft red as eda_utils)
GREY = "#8a8985"


# ----------------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------------
def _term_index(ebm, term):
    names = list(ebm.term_names_)
    if term not in names:
        raise ValueError(f"'{term}' is not a term of this EBM. Terms: {names}")
    return names.index(term)


def main_terms(ebm, order=None):
    """Names of single-feature terms (no interactions), most important first.

    order: list of names to sort by (e.g. permutation importance index). Default: EBM importance.
    """
    singles = [t for t in ebm.term_names_ if " & " not in t]
    if order is not None:
        return [t for t in order if t in singles]
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_)
    return list(imp[singles].sort_values(ascending=False).index)


def interaction_terms(ebm):
    """Names of pairwise terms, most important first (EBM importance)."""
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_)
    pairs = [t for t in ebm.term_names_ if " & " in t]
    return list(imp[pairs].sort_values(ascending=False).index)


def shape_data(ebm, feature):
    """Learned shape of one feature.

    Returns a dict:
      kind:    "continuous" or "categorical"
      x:       bin left edges (continuous) or category names (categorical)
      scores:  contribution to log-odds for each bin / category
      lower, upper: uncertainty band (from EBM's internal bagging)
      missing: contribution when the value is missing (EBM keeps it, but the default plots hide it)
    """
    i = _term_index(ebm, feature)
    d = ebm.explain_global().data(i)
    scores = np.asarray(d["scores"], dtype=float)
    lower = np.asarray(d.get("lower_bounds", scores), dtype=float)
    upper = np.asarray(d.get("upper_bounds", scores), dtype=float)
    names = d["names"]
    f_idx = list(ebm.feature_names_in_).index(feature)
    is_cont = ebm.feature_types_in_[f_idx] == "continuous"
    x = np.asarray(names, dtype=float)[:-1] if is_cont else list(names)   # continuous: edges, one more than scores
    missing = float(ebm.term_scores_[i][0])                               # first slot = missing value
    return {"kind": "continuous" if is_cont else "categorical", "x": x, "scores": scores,
            "lower": lower, "upper": upper, "missing": missing}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _grid(n, ncols, width=4.2, height=3.2):
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _view_range(feature, x, X_ref, clip):
    # x-range to show: X_ref percentiles if given, else all bins
    if X_ref is not None and clip:
        lo, hi = np.nanpercentile(X_ref[feature].astype(float), [clip[0] * 100, clip[1] * 100])
        return lo, hi
    return np.nanmin(x), np.nanmax(x)


def _short_num(v, _pos=None):
    if abs(v) >= 1e6:
        return f"{v / 1e6:g}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


# ----------------------------------------------------------------------------
# main effects
# ----------------------------------------------------------------------------
def shape_ylim(ebm, features=None, X_ref=None, clip=(0.01, 0.99), pad=0.1):
    """Common (low, high) log-odds range covering the shapes of these features (default: all single terms).

    Only the part of each shape inside the shown x-range counts (same clip as the plots),
    so one extreme bin at the far end doesn't stretch every plot. Pass the result as ylim=.
    """
    feats = main_terms(ebm, order=features) if features is not None else main_terms(ebm)
    lows, highs = [0.0], [0.0]
    for f in feats:
        d = shape_data(ebm, f)
        if d["kind"] == "continuous":
            x_lo, x_hi = _view_range(f, d["x"], X_ref, clip)
            vis = (d["x"] >= x_lo) & (d["x"] <= x_hi)
            if not vis.any():
                continue
            lows.append(d["lower"][vis].min())
            highs.append(d["upper"][vis].max())
        else:
            lows.append(d["lower"].min())
            highs.append(d["upper"].max())
    lo, hi = min(lows), max(highs)
    span = hi - lo + 1e-9
    return lo - pad * span, hi + pad * span


def plot_shape(ebm, feature, ax=None, X_ref=None, clip=(0.01, 0.99), color=EBM_COLOR, show_missing=True,
               ylim=None):
    """Shape function of one feature (step line + uncertainty band, or bars for categories).

    X_ref: data used to choose the x-range (e.g. X_tr). clip=(0.01, 0.99) shows the 1st-99th percentile,
           so a few extreme customers don't squash the plot. None = show every bin.
    ylim:  fixed (low, high) for the log-odds axis, e.g. from shape_ylim(), so features can be compared.
           For categorical features the log-odds axis is horizontal, so it is applied to x.
    """
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    d = shape_data(ebm, feature)
    title = feature
    if d["kind"] == "continuous":
        x_lo, x_hi = _view_range(feature, d["x"], X_ref, clip)
        vis = (d["x"] >= x_lo) & (d["x"] <= x_hi)
        ax.fill_between(d["x"], d["lower"], d["upper"], step="post", color=color, alpha=0.18, lw=0)
        ax.step(d["x"], d["scores"], where="post", color=color, lw=2)
        ax.set_xlim(x_lo, x_hi)
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif vis.any():
            lo, hi = d["lower"][vis].min(), d["upper"][vis].max()
            pad = 0.1 * (hi - lo + 1e-9)
            ax.set_ylim(min(lo, 0) - pad, max(hi, 0) + pad)
        ax.axhline(0, color=GREY, lw=1)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_short_num))
    else:
        order = np.argsort(d["scores"])
        names = np.asarray(d["x"])[order]
        s = d["scores"][order]
        err = [(d["scores"] - d["lower"])[order], (d["upper"] - d["scores"])[order]]
        ax.barh(names, s, color=color, height=0.6, xerr=err, error_kw={"ecolor": GREY, "lw": 1})
        ax.axvline(0, color=GREY, lw=1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
        if ylim is not None:
            ax.set_xlim(*ylim)
    if show_missing and abs(d["missing"]) > 1e-12:
        title += f"  |  missing: {d['missing']:+.2f}"
    ax.set_title(title, fontsize=10)
    if show:
        plt.tight_layout()
        plt.show()


def plot_shapes(ebm, features=None, top_n=6, ncols=3, X_ref=None, clip=(0.01, 0.99), sharey=True,
                ylim=None, title=None):
    """Shape functions of several features in a grid.

    features: names in the order to show (e.g. imp_ebm.index[:6] from permutation importance).
              Interaction names are skipped. Default: top_n by EBM importance.
    sharey:   same log-odds range for every plot, so the size of effects can be compared
              (False = each plot uses its own range, better to see the shape of small effects).
    ylim:     fixed range to use instead (e.g. shape_ylim(ebm, X_ref=X_tr) to match all features).
    """
    feats = main_terms(ebm, order=features)[:top_n] if features is not None else main_terms(ebm)[:top_n]
    if ylim is None and sharey:
        ylim = shape_ylim(ebm, feats, X_ref=X_ref, clip=clip)
    fig, axes = _grid(len(feats), ncols)
    for f, ax in zip(feats, axes):
        plot_shape(ebm, f, ax=ax, X_ref=X_ref, clip=clip, ylim=ylim)
    note = "  [same y-axis for all]" if ylim is not None else ""
    fig.suptitle((title or "EBM shape functions (y = contribution to log-odds; above 0 = riskier)") + note, y=1.0)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# interactions
# ----------------------------------------------------------------------------
def _axis_info(names, n_cells):
    # continuous side: names are edges (n_cells + 1); categorical side: names are labels (n_cells)
    if len(names) == n_cells + 1:
        return "continuous", np.asarray(names, dtype=float)
    return "categorical", list(names)


def _cell_index(values, kind, axis_x):
    # which heatmap cell each row falls into (continuous: by edges, categorical: by label)
    if kind == "continuous":
        v = pd.Series(values, dtype=float).to_numpy()
        idx = np.searchsorted(axis_x, v, side="right") - 1
        idx = np.clip(idx, 0, len(axis_x) - 2)
        idx[np.isnan(v)] = -1
        return idx
    lookup = {str(c): k for k, c in enumerate(axis_x)}
    return pd.Series(values).astype(str).map(lookup).fillna(-1).astype(int).to_numpy()


def interaction_counts(ebm, term, X):
    """Number of rows of X in each cell of a pairwise term (same grid as the heatmap)."""
    i = _term_index(ebm, term)
    d = ebm.explain_global().data(i)
    z = np.asarray(d["scores"], dtype=float)
    left, right = term.split(" & ")
    kind_l, left_x = _axis_info(d["left_names"], z.shape[0])
    kind_r, right_x = _axis_info(d["right_names"], z.shape[1])
    li = _cell_index(X[left], kind_l, left_x)
    ri = _cell_index(X[right], kind_r, right_x)
    ok = (li >= 0) & (ri >= 0)
    counts = np.zeros(z.shape, dtype=int)
    np.add.at(counts, (li[ok], ri[ok]), 1)
    return counts


def interaction_std(ebm, term):
    """Std of each interaction cell across EBM's internal bags (same grid as the heatmap)."""
    i = _term_index(ebm, term)
    sd = np.asarray(ebm.standard_deviations_[i], dtype=float)
    return sd[1:-1, 1:-1]   # drop the "missing" and "unknown" slots on both axes


def plot_interaction(ebm, term, ax=None, X_ref=None, clip=(0.01, 0.99), vmax=None, min_count=None,
                     hide_uncertain=False, z_thr=2.0):
    """Heatmap of one pairwise term. Red = adds risk, blue = reduces risk (log-odds).

    This is the extra effect of the pair on top of the two single shapes.
    min_count: with X_ref, areas with fewer rows than this are greyed out (too few customers
               to trust the colour). Rows are counted on blocks of about 10 x 10 over the grid,
               because single EBM cells are very small. The title shows the share of rows greyed out.
    hide_uncertain: grey out cells where |effect| < z_thr * std across EBM's bags,
               i.e. the colour can't be told apart from 0 (noise). Needs outer_bags > 1 (default 14).
    """
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    i = _term_index(ebm, term)
    d = ebm.explain_global().data(i)
    z = np.asarray(d["scores"], dtype=float)          # shape: (left cells, right cells)
    left, right = term.split(" & ")
    kind_l, left_x = _axis_info(d["left_names"], z.shape[0])
    kind_r, right_x = _axis_info(d["right_names"], z.shape[1])
    v = vmax or np.nanmax(np.abs(z)) or 1.0

    # x = right feature, y = left feature
    xs = right_x if kind_r == "continuous" else np.arange(z.shape[1] + 1) - 0.5
    ys = left_x if kind_l == "continuous" else np.arange(z.shape[0] + 1) - 0.5
    sparse_note = ""
    if min_count and X_ref is not None:
        counts = interaction_counts(ebm, term, X_ref)
        # EBM cells are very fine, so judge support on blocks (about 10 x 10 over the grid)
        bl = max(1, math.ceil(counts.shape[0] / 10))
        br = max(1, math.ceil(counts.shape[1] / 10))
        rl = np.arange(counts.shape[0]) // bl
        rr = np.arange(counts.shape[1]) // br
        block = np.zeros((rl.max() + 1, rr.max() + 1), dtype=int)
        np.add.at(block, (rl[:, None].repeat(counts.shape[1], 1), rr[None, :].repeat(counts.shape[0], 0)), counts)
        sparse = block[rl][:, rr] < min_count
        z = np.where(sparse, np.nan, z)
        share = counts[sparse].sum() / max(counts.sum(), 1)
        sparse_note = f"\ngrey: blocks with < {min_count} rows ({share:.1%} of rows)"
        ax.set_facecolor("#d9d9d9")
        v = vmax or np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
    if hide_uncertain:
        sd = interaction_std(ebm, term)
        if sd.shape == z.shape:
            noisy = np.abs(np.nan_to_num(z)) < z_thr * sd
            share = noisy.mean()
            z = np.where(noisy, np.nan, z)
            sparse_note += f"\ngrey: |effect| < {z_thr:g} x std ({share:.0%} of cells)"
            ax.set_facecolor("#d9d9d9")
    mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(z), cmap="RdBu_r", vmin=-v, vmax=v, shading="flat")
    plt.colorbar(mesh, ax=ax, shrink=0.8, label="log-odds")

    if kind_r == "continuous":
        ax.set_xlim(*_view_range(right, right_x, X_ref, clip))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_short_num))
    else:
        ax.set_xticks(range(z.shape[1]), right_x, rotation=45, ha="right", fontsize=8)
    if kind_l == "continuous":
        ax.set_ylim(*_view_range(left, left_x, X_ref, clip))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_short_num))
    else:
        ax.set_yticks(range(z.shape[0]), left_x, fontsize=8)
    ax.set(xlabel=right, ylabel=left)
    ax.set_title(term + sparse_note, fontsize=10)
    ax.grid(False)
    if show:
        plt.tight_layout()
        plt.show()


def plot_interactions(ebm, terms=None, top_n=3, ncols=3, X_ref=None, clip=(0.01, 0.99), min_count=None,
                      hide_uncertain=False, z_thr=2.0, title=None):
    """Heatmaps of several pairwise terms (default: top_n by EBM importance).
    min_count / hide_uncertain: grey out unreliable areas (see plot_interaction)."""
    terms = list(terms) if terms is not None else interaction_terms(ebm)[:top_n]
    if not terms:
        print("this EBM has no interaction terms")
        return
    fig, axes = _grid(len(terms), ncols, width=4.8, height=3.8)
    for t, ax in zip(terms, axes):
        plot_interaction(ebm, t, ax=ax, X_ref=X_ref, clip=clip, min_count=min_count,
                         hide_uncertain=hide_uncertain, z_thr=z_thr)
    fig.suptitle(title or "EBM pairwise interactions (extra effect on top of the single shapes)", y=1.0)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# EBM shape vs one-variable view
# ----------------------------------------------------------------------------
def _logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _rate_by_bin(x, y, bins, min_count):
    # returns DataFrame with x_mid, n, rate. Whole-number columns with few values: one bin per value.
    x = pd.Series(x, dtype=float)
    y = pd.Series(np.asarray(y), index=x.index)
    ok = x.notna()
    x, y = x[ok], y[ok]
    n_unique = x.nunique()
    if n_unique <= max(bins, 20) or (n_unique <= 50 and bool((x % 1 == 0).all())):
        vc = x.value_counts()
        kept = vc.index[vc >= min_count]
        if len(kept):
            x = x.clip(upper=kept.max())
        g = x
    else:
        inner = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
        g = pd.Series(np.searchsorted(inner, x.to_numpy(), side="left"), index=x.index)
    out = pd.DataFrame({"x": x, "y": y, "g": g}).groupby("g").agg(x_mid=("x", "median"), n=("y", "size"),
                                                                    rate=("y", "mean"))
    return out.sort_values("x_mid")


def plot_shape_vs_rate(ebm, X, y, features=None, top_n=6, ncols=3, bins=10, min_count=50,
                       clip=(0.01, 0.99), ylim=None, title=None):
    """Compare two views of each feature, both on the log-odds scale (0 = average):

      blue line  = EBM shape: effect of the feature AFTER the other features are taken into account
      red points = one-variable view: logit(rate in bin) - logit(overall rate), from the data only

    Same shape  -> the feature works on its own.
    Red strong, blue flat -> other (correlated) features already carry this information.
    Red flat, blue strong -> the feature matters only together with the others.

    X, y: usually the training data (X_tr, y_tr). Categorical features are compared per category.
    ylim: fixed log-odds range for every plot (e.g. shape_ylim(ebm, X_ref=X_tr)). Red points outside
          the range are cut off; the title then says how many.
    """
    y = np.asarray(y)
    overall = _logit(y.mean())
    feats = main_terms(ebm, order=features)[:top_n] if features is not None else main_terms(ebm)[:top_n]
    fig, axes = _grid(len(feats), ncols)
    for f, ax in zip(feats, axes):
        d = shape_data(ebm, f)
        if d["kind"] == "continuous":
            plot_shape(ebm, f, ax=ax, X_ref=X, clip=clip, show_missing=False, ylim=ylim)
            r = _rate_by_bin(X[f], y, bins, min_count)
            rl = _logit(r["rate"]) - overall
            ax.plot(r["x_mid"], rl, color=RATE_COLOR, marker="o", markersize=4, lw=1.2, ls="--")
            if ylim is None:
                ax.set_ylim(min(ax.get_ylim()[0], rl.min() * 1.1), max(ax.get_ylim()[1], rl.max() * 1.1))
            else:
                n_out = int(((rl < ylim[0]) | (rl > ylim[1])).sum())
                if n_out:
                    ax.set_title(f"{f}  ({n_out} red point(s) off scale)", fontsize=10)
        else:
            cats = list(d["x"])
            s = pd.Series(np.asarray(y), index=X.index).groupby(X[f].astype(str)).mean()
            uni = _logit(s.reindex(cats).to_numpy(dtype=float)) - overall
            pos = np.arange(len(cats))
            ax.bar(pos - 0.2, d["scores"], width=0.4, color=EBM_COLOR)
            ax.bar(pos + 0.2, uni, width=0.4, color=RATE_COLOR, alpha=0.8)
            ax.set_xticks(pos, [str(c)[:12] for c in cats], rotation=45, ha="right", fontsize=8)
            ax.axhline(0, color=GREY, lw=1)
            ax.set_title(f, fontsize=10)
            if ylim is not None:
                ax.set_ylim(*ylim)
    handles = [Line2D([], [], color=EBM_COLOR, lw=2, label="EBM shape (other features accounted for)"),
               Line2D([], [], color=RATE_COLOR, lw=1.2, ls="--", marker="o", label="one-variable view (data only)")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title or "EBM shape vs one-variable view (log-odds, 0 = average)", y=1.0, x=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# one feature, three views
# ----------------------------------------------------------------------------
NEG_COLOR = "#a3a3a3"   # target = 0 (grey)
POS_COLOR = "#e07b72"   # target = 1 (soft red)
HIST_COLOR = "#d9d9d9"  # background histogram


def _clip_pair(clip):
    if not clip:
        return None
    return (clip, 1 - clip) if isinstance(clip, (int, float)) else tuple(clip)


def plot_feature_report(ebm, X, y, feature, bins=10, hist_bins=30, clip=0.01, density_of="both", min_count=50,
                        target_name="target", ylim=None):
    """Three plots side by side for one feature.

    1. P(X | Y): histogram of the feature for Y = 0 (grey) and Y = 1 (red outline), density.
    2. EBM shape + data behind it (right axis), with the SAME bins as plot 1:
       density_of: "both"     = grey bar = all rows, red part = rows with Y = 1 (default)
                   "all"      = all rows only
                   "positive" = rows with Y = 1 only
       The shape is less reliable where bars are low, especially where there are few Y = 1 rows.
    3. EBM shape + P(Y | X): red points = target rate per bin, on the same log-odds scale
       as the shape (0 = average). The right axis shows the same points as a rate in %.

    hist_bins: number of equal-width bins for plots 1 and 2 (over the shown range).
    bins:      number of quantile bins for plot 3 (each point has about the same number of rows).
               Whole-number columns with few values (counts) use one bin per value everywhere.
    clip: cut the tails for the view, e.g. 0.01 = 1st-99th percentile, (0.01, 0.95), or None
    X, y: usually the training data (X_tr, y_tr)
    ylim: fixed log-odds range for plots 2 and 3, so reports of different features use the same scale.
          Make it once with ylim = shape_ylim(ebm, X_ref=X_tr) and pass it to every call.
    """
    y = pd.Series(np.asarray(y), index=X.index)
    d = shape_data(ebm, feature)
    clip = _clip_pair(clip)
    x_all = X[feature]
    overall = _logit(y.mean())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

    if d["kind"] == "continuous":
        x = x_all.astype(float)
        lo, hi = (x.quantile(clip[0]), x.quantile(clip[1])) if clip else (x.min(), x.max())
        r = _rate_by_bin(x, y.to_numpy(), bins, min_count)
        n_unique = x.nunique()
        is_count = n_unique <= max(bins, 20) or (n_unique <= 50 and bool((x.dropna() % 1 == 0).all()))
        if is_count:
            hi = max(hi, r["x_mid"].max())                   # keep the merged top value ("k+") visible
            lo, hi = np.floor(lo), np.ceil(hi)
            edges = np.arange(lo - 0.5, hi + 1.5)            # one bin per whole number
        else:
            edges = np.histogram_bin_edges(x.dropna().clip(lo, hi), bins=hist_bins, range=(lo, hi))
        xc = x.clip(edges[0], edges[-1])

        # 1. P(X | Y)
        ax = axes[0]
        ax.hist(xc[y == 0].dropna(), bins=edges, density=True, color=NEG_COLOR, alpha=0.6, label=f"{target_name} = 0")
        ax.hist(xc[y == 1].dropna(), bins=edges, density=True, color=POS_COLOR, histtype="step", lw=1.8,
                label=f"{target_name} = 1")
        ax.set(title=f"1. P({feature} | {target_name})", xlabel=feature)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8)

        # 2. shape + data behind (same bins as plot 1)
        ax = axes[1]
        ax2 = ax.twinx()
        n_all, _ = np.histogram(xc.dropna(), bins=edges)
        n_pos, _ = np.histogram(xc[y == 1].dropna(), bins=edges)
        left, width = edges[:-1], np.diff(edges)
        if density_of in ("both", "all"):
            ax2.bar(left, n_all, width=width, align="edge", color=HIST_COLOR, edgecolor="white", lw=0.5)
        if density_of in ("both", "positive"):
            ax2.bar(left, n_pos, width=width, align="edge", color=POS_COLOR, alpha=0.35, edgecolor="white", lw=0.5)
        label = {"both": f"rows (grey = all, red = {target_name} = 1)", "all": "rows (all)",
                 "positive": f"rows ({target_name} = 1)"}[density_of]
        ax2.set_ylabel(label, fontsize=8, color=GREY)
        ax2.tick_params(axis="y", colors=GREY, labelsize=8)
        ax2.grid(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        plot_shape(ebm, feature, ax=ax, X_ref=X, clip=clip, show_missing=True, ylim=ylim)
        ax.set_xlim(edges[0], edges[-1])
        if is_count:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_title(f"2. EBM shape + rows per bin\n{ax.get_title()}", fontsize=10)
        ax.set_ylabel("log-odds", fontsize=8)

        # 3. shape + P(Y | X)
        ax = axes[2]
        plot_shape(ebm, feature, ax=ax, X_ref=X, clip=clip, show_missing=False, ylim=ylim)
        rate_lo = _logit(r["rate"].to_numpy()) - overall
        ax.plot(r["x_mid"], rate_lo, color=POS_COLOR, marker="o", markersize=4, lw=1.2, ls="--")
        if ylim is None:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(min(y0, rate_lo.min() * 1.1), max(y1, rate_lo.max() * 1.1))
        ax.set_xlim(edges[0], edges[-1])
        if is_count:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylabel("log-odds (0 = average)", fontsize=8)
    else:
        cats = [str(c) for c in d["x"]]
        xs = x_all.astype(str)

        # 1. P(X | Y)
        ax = axes[0]
        s0 = xs[y == 0].value_counts(normalize=True).reindex(cats, fill_value=0) * 100
        s1 = xs[y == 1].value_counts(normalize=True).reindex(cats, fill_value=0) * 100
        pos = np.arange(len(cats))
        ax.bar(pos - 0.2, s0, width=0.4, color=NEG_COLOR, label=f"{target_name} = 0")
        ax.bar(pos + 0.2, s1, width=0.4, color=POS_COLOR, label=f"{target_name} = 1")
        ax.set_xticks(pos, [c[:12] for c in cats], rotation=45, ha="right", fontsize=8)
        ax.set(title=f"1. P({feature} | {target_name})", ylabel="% of class")
        ax.legend(frameon=False, fontsize=8)

        # 2. shape + rows per category
        ax = axes[1]
        ax2 = ax.twinx()
        if density_of in ("both", "all"):
            ax2.bar(pos, xs.value_counts().reindex(cats, fill_value=0), width=0.8, color=HIST_COLOR)
        if density_of in ("both", "positive"):
            ax2.bar(pos, xs[y == 1].value_counts().reindex(cats, fill_value=0), width=0.8, color=POS_COLOR, alpha=0.35)
        ax2.set_ylabel("rows" + (f" (grey = all, red = {target_name} = 1)" if density_of == "both" else ""),
                       fontsize=8, color=GREY)
        ax2.tick_params(axis="y", colors=GREY, labelsize=8)
        ax2.grid(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(pos, d["scores"], color=EBM_COLOR, marker="o", lw=0, markersize=7)
        ax.vlines(pos, d["lower"], d["upper"], color=EBM_COLOR, lw=2)
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xticks(pos, [c[:12] for c in cats], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"2. EBM shape + rows per category\n{feature}", fontsize=10)
        ax.set_ylabel("log-odds", fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # 3. shape + P(Y | X)
        ax = axes[2]
        rate = y.groupby(xs).mean().reindex(cats)
        ax.bar(pos - 0.2, d["scores"], width=0.4, color=EBM_COLOR, label="EBM shape")
        ax.bar(pos + 0.2, _logit(rate.to_numpy(dtype=float)) - overall, width=0.4, color=POS_COLOR,
               label="one-variable view")
        ax.axhline(0, color=GREY, lw=1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xticks(pos, [c[:12] for c in cats], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("log-odds (0 = average)", fontsize=8)

    # right axis of panel 3: same scale shown as a rate in %
    ax = axes[2]
    to_rate = lambda v: 100 / (1 + np.exp(-(np.asarray(v) + overall)))
    to_lo = lambda p: _logit(np.asarray(p) / 100) - overall
    sec = ax.secondary_yaxis("right", functions=(to_rate, to_lo))
    sec.set_ylabel(f"{target_name} rate (%)", fontsize=8, color=POS_COLOR)
    sec.tick_params(labelsize=8, colors=POS_COLOR)
    ax.set_title(f"3. EBM shape (blue) vs P({target_name} | {feature}) (red)", fontsize=10)

    fig.suptitle(f"{feature}", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
