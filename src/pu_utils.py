"""Positive-Unlabeled (PU) learning with score-based weights.

Setting: some positives are labeled (e.g. confirmed fraud), the rest of the data is "unlabeled":
it is mostly negatives, but some undetected positives are hidden in it.
Treating all unlabeled rows as clean negatives teaches the model that hidden positives are normal.

Method (Elkan & Noto, 2008), two steps:
  1. Train a first model g(x) = P(labeled | x) on labeled-vs-unlabeled, with cross-fitting
     (each row is scored by a model that did not see it).
  2. Estimate c = P(labeled | positive) as the average g(x) over labeled positives.
     For each unlabeled row, the chance that it is a hidden positive is
         w(x) = (1 - c) / c * g(x) / (1 - g(x))
     Then train the final model with sample weights:
       - "duplicate" (Elkan & Noto): each unlabeled row is used twice,
         as a positive with weight w(x) and as a negative with weight 1 - w(x)
       - "downweight": each unlabeled row is a negative with weight 1 - w(x)
         (simpler: "this row looks risky, so don't punish the model much for scoring it high")

Usage:
    from pu_utils import (hide_labels, PUWeightedClassifier, c_from_prior, pu_report,
                          c_sensitivity, plot_c_sensitivity)
    s = hide_labels(y_true, label_freq=0.5)                     # simulation only
    c = c_from_prior(s_tr, prior=0.02)                          # assumed true fraud rate
    pu = PUWeightedClassifier(make_model, method="duplicate", c=c).fit(X_tr, s_tr)
    p = pu.predict_proba(X_te)[:, 1]

    # how much does the assumed c matter? (simulation: true labels known)
    res = c_sensitivity(make_model, X_tr, s_tr, X_te, y_te, s_te)
    plot_c_sensitivity(res, true_c=0.5)
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ----------------------------------------------------------------------------
# simulation: hide some positive labels
# ----------------------------------------------------------------------------
def hide_labels(y, label_freq=0.5, random_state=42, select_prob=None):
    """Make PU labels s from true labels y (only for experiments where y is known).

    label_freq:  share of positives that stay labeled (c). The rest become "unlabeled" (s = 0).
    select_prob: optional array, chance for each positive to be labeled (not random detection,
                 e.g. large frauds are found more often). Overrides label_freq.
                 Use it to test what happens when the "labeled at random" assumption is wrong.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(int)
    prob = np.full(len(y), label_freq, dtype=float) if select_prob is None else np.asarray(select_prob, float)
    s = (y == 1) & (rng.uniform(size=len(y)) < prob)
    return s.astype(int)


# ----------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------
def estimate_c(g_labeled):
    """c = P(labeled | positive) = average first-model score over labeled positives (Elkan & Noto's e1).

    Warning: this is only accurate when positives and negatives are well separated.
    With overlapping classes (typical in credit and fraud) it UNDER-estimates c a lot,
    which makes the weights too large. Prefer c_from_prior() when you have a business estimate.
    """
    return float(np.mean(g_labeled))


def c_from_prior(s, prior):
    """c from an assumed true positive rate: c = (share of labeled rows) / prior.

    prior: business estimate of the real positive rate, e.g. "about 2% of accounts are fraud,
           but only 1% are confirmed" -> prior = 0.02, c = 0.5. Try a few values (sensitivity).
    """
    return float(np.mean(np.asarray(s)) / prior)


def pu_weights(g, c, eps=1e-6):
    """Chance that an unlabeled row is a hidden positive: w = (1 - c) / c * g / (1 - g), clipped to [0, 1]."""
    g = np.clip(np.asarray(g, dtype=float), eps, 1 - eps)
    w = (1 - c) / c * g / (1 - g)
    return np.clip(w, 0.0, 1.0)


def make_weighted_set(X, s, w, method="duplicate"):
    """Training set for step 2.

    X: features, s: PU labels (1 = labeled positive, 0 = unlabeled), w: hidden-positive weight per row
    method: "duplicate"  -> unlabeled rows twice: (label 1, weight w) and (label 0, weight 1 - w)
            "downweight" -> unlabeled rows once:  (label 0, weight 1 - w)
    Returns X_new, y_new, sample_weight.
    """
    s = np.asarray(s).astype(int)
    w = np.asarray(w, dtype=float)
    pos, unl = s == 1, s == 0
    X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)
    take = (lambda m: X[m]) if not isinstance(X, pd.DataFrame) else (lambda m: X.loc[m])

    parts_X = [take(pos), take(unl)]
    parts_y = [np.ones(pos.sum()), np.zeros(unl.sum())]
    parts_w = [np.ones(pos.sum()), 1 - w[unl]]
    if method == "duplicate":
        parts_X.append(take(unl))
        parts_y.append(np.ones(unl.sum()))
        parts_w.append(w[unl])
    elif method != "downweight":
        raise ValueError("method must be 'duplicate' or 'downweight'")

    X_new = pd.concat(parts_X, ignore_index=True) if isinstance(X, pd.DataFrame) else np.vstack(parts_X)
    return X_new, np.concatenate(parts_y).astype(int), np.concatenate(parts_w)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class PUWeightedClassifier:
    """Two-step PU classifier.

    base_model: a model (or a function returning a new model) with fit(X, y, sample_weight=...)
                and predict_proba, e.g. LightGBM, XGBoost, HistGradientBoosting, EBM.
    method:     "duplicate" (Elkan & Noto) or "downweight"
    n_folds:    folds for cross-fitting the first model (out-of-fold scores, no self-scoring)
    c:          share of positives that are labeled, if you know or assume it (e.g. from c_from_prior).
                None = estimate it from the data (often too low, see estimate_c).
    final_model: model for step 2 (default: same type as base_model)
    fit_kwargs: extra keyword arguments passed to every fit (e.g. categorical settings)

    After fit:
      g_    out-of-fold first-model scores, P(labeled | x)
      c_    estimated share of positives that are labeled
      w_    hidden-positive weight per training row (0 for labeled rows)
    """

    def __init__(self, base_model, method="duplicate", n_folds=5, c=None, final_model=None, random_state=42,
                 fit_kwargs=None):
        self.base_model = base_model
        self.method = method
        self.c = c
        self.n_folds = n_folds
        self.final_model = final_model
        self.random_state = random_state
        self.fit_kwargs = fit_kwargs or {}

    def _new(self, m):
        return m() if callable(m) and not hasattr(m, "fit") else clone(m)

    def first_stage(self, X, s):
        """Step 1 only: out-of-fold P(labeled | x) for every row (cross-fitting)."""
        s = np.asarray(s).astype(int)
        X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)
        take = (lambda idx: X.iloc[idx]) if isinstance(X, pd.DataFrame) else (lambda idx: X[idx])
        g = np.zeros(len(s))
        folds = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for tr, va in folds.split(np.zeros(len(s)), s):
            m = self._new(self.base_model)
            m.fit(take(tr), s[tr], **self.fit_kwargs)
            g[va] = m.predict_proba(take(va))[:, 1]
        return g

    def fit(self, X, s, g=None):
        """g: step-1 scores from first_stage(), to reuse them (e.g. when trying several c)."""
        s = np.asarray(s).astype(int)
        X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)

        # step 1: out-of-fold P(labeled | x)
        g = self.first_stage(X, s) if g is None else np.asarray(g, dtype=float)
        self.g_ = g
        self.c_ = self.c if self.c is not None else estimate_c(g[s == 1])
        self.w_ = np.where(s == 0, pu_weights(g, self.c_), 0.0)

        # step 2: weighted final model
        X_new, y_new, sw = make_weighted_set(X, s, self.w_, self.method)
        self.model_ = self._new(self.final_model or self.base_model)
        self.model_.fit(X_new, y_new, sample_weight=sw, **self.fit_kwargs)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# ----------------------------------------------------------------------------
# evaluation (simulation, where the true labels are known)
# ----------------------------------------------------------------------------
def hidden_recall_at_k(y_true, s, score, k=0.01):
    """Among UNLABELED rows ranked by score, share of hidden positives found in the top k (fraction).

    This answers the business question: "if analysts review the top k of unlabeled accounts,
    how many of the undetected frauds do they find?"
    """
    y_true, s, score = np.asarray(y_true), np.asarray(s), np.asarray(score)
    unl = s == 0
    hidden = y_true[unl] == 1
    if hidden.sum() == 0:
        return np.nan
    n_top = max(1, int(round(unl.sum() * k)))
    top = np.argsort(-score[unl])[:n_top]
    return hidden[top].sum() / hidden.sum()


def pu_report(y_true, s, scores, ks=(0.01, 0.05)):
    """Compare models on the TRUE labels. scores: {name: predicted probability}.

    AUC / PR-AUC (true): ranking against the real positives, including hidden ones
    hidden recall @ k:   share of hidden positives found in the top k of unlabeled rows
    mean score:          average predicted probability vs the true positive rate (probability scale)
    """
    rows = {}
    for name, p in scores.items():
        row = {"AUC (true)": roc_auc_score(y_true, p), "PR-AUC (true)": average_precision_score(y_true, p)}
        for k in ks:
            row[f"hidden recall @ top {k:.0%}"] = hidden_recall_at_k(y_true, s, p, k)
        row["mean score"] = float(np.mean(p))
        rows[name] = row
    out = pd.DataFrame(rows).T
    out.attrs["true positive rate"] = float(np.mean(y_true))
    return out


# ----------------------------------------------------------------------------
# c sensitivity
# ----------------------------------------------------------------------------
def c_sensitivity(base_model, X_tr, s_tr, X_te, y_te, s_te, cs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                  method="duplicate", k=0.05, n_folds=5, random_state=42, fit_kwargs=None):
    """Train the PU model with several assumed values of c and score each on the TRUE test labels.

    Step 1 is trained once and reused, so only step 2 is repeated for each c.
    Also adds the naive model (unlabeled = negative) as the row c = NaN, for comparison.
    Returns a DataFrame with one row per c.
    """
    pu = PUWeightedClassifier(base_model, method=method, n_folds=n_folds, random_state=random_state,
                              fit_kwargs=fit_kwargs)
    g = pu.first_stage(X_tr, s_tr)
    rows = []
    for c in cs:
        model = PUWeightedClassifier(base_model, method=method, n_folds=n_folds, c=c,
                                     random_state=random_state, fit_kwargs=fit_kwargs).fit(X_tr, s_tr, g=g)
        p = model.predict_proba(X_te)[:, 1]
        rows.append({"c": c, "PR-AUC (true)": average_precision_score(y_te, p),
                     "AUC (true)": roc_auc_score(y_te, p),
                     f"hidden recall @ top {k:.0%}": hidden_recall_at_k(y_te, s_te, p, k),
                     "mean score": float(np.mean(p))})
    naive = pu._new(base_model)
    naive.fit(X_tr, np.asarray(s_tr).astype(int), **(fit_kwargs or {}))
    p = naive.predict_proba(X_te)[:, 1]
    rows.append({"c": np.nan, "PR-AUC (true)": average_precision_score(y_te, p),
                 "AUC (true)": roc_auc_score(y_te, p),
                 f"hidden recall @ top {k:.0%}": hidden_recall_at_k(y_te, s_te, p, k),
                 "mean score": float(np.mean(p))})
    out = pd.DataFrame(rows)
    out.attrs["true positive rate"] = float(np.mean(y_te))
    return out


def plot_c_sensitivity(res, true_c=None, metrics=None, title="How much does the assumed c matter?"):
    """One small plot per metric: PU result for each assumed c (line) vs the naive model (dashed).

    true_c: the real c, if known (simulation) -> vertical line.
    For "mean score", the dotted line is the true positive rate (where a well-scaled model should be).
    """
    import matplotlib.pyplot as plt

    pu = res[res["c"].notna()].sort_values("c")
    naive = res[res["c"].isna()].iloc[0]
    metrics = metrics or [m for m in res.columns if m != "c"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), squeeze=False)
    for m, ax in zip(metrics, axes.ravel()):
        ax.plot(pu["c"], pu[m], color="#2a78d6", marker="o", label="PU (assumed c)")
        ax.axhline(naive[m], color="#a3a3a3", ls="--", lw=1.5, label="naive (unlabeled = 0)")
        if m == "mean score" and "true positive rate" in res.attrs:
            ax.axhline(res.attrs["true positive rate"], color="#e07b72", ls=":", lw=1.5, label="true positive rate")
        if true_c is not None:
            ax.axvline(true_c, color="black", lw=1)
            ax.text(true_c, ax.get_ylim()[1], " true c", va="top", fontsize=8)
        ax.set(title=m, xlabel="assumed c (share of positives labeled)")
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()
