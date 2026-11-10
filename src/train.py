"""
train.py

Trains and evaluates persuasion models on two complementary tasks.

Task A -- comment-level delta prediction
    Every challenger comment, labeled 1 if the OP awarded it a confirmed delta
    (~2% positive). This is the realistic "which comments will change minds?"
    setting. Models compared:
        - Sentiment only (LR)                  tests whether tone alone predicts persuasion
        - Hand-crafted text features (LR)      interpretable baseline
        - Hand-crafted text features (HGB)     same features, non-linear model
        - TF-IDF (LR)                          word choice
        - TF-IDF + hand-crafted text (LR)      the deployable scorer used by predict.py
        - All features incl. context (HGB)     adds where/when the comment was posted
        - Stacked (HGB + out-of-fold TF-IDF)   gradient boosting over all features plus
                                               the TF-IDF model's out-of-fold score

Task B -- paired root replies (Tan et al., 2016)
    Root replies come in pairs on the same thread: one whose discussion won a
    delta and one similar one that did not. Pairwise accuracy (is the winner
    scored higher?) controls for topic and OP, isolating argument quality.

Methodology
    - Train/test split follows the corpus' official thread-level split, so no
      thread appears in both (comments in one thread share vocabulary and OP,
      and a random row split leaks that across the boundary).
    - Decision thresholds are tuned on out-of-fold predictions from
      GroupKFold over training threads, never on the test set.
    - Test metrics come with 95% bootstrap CIs, resampling whole threads.
    - Feature-group ablation: drop each group from the full HGB model, and
      also fit each group alone, to measure what each contributes.

Usage:
    python src/train.py [--folds 5] [--bootstrap 500] [--skip-ablation] [--figures-only]

Outputs:
    models/scorer.joblib                deployable TF-IDF + hand-crafted model (predict.py)
    models/test_predictions_comment_level.parquet   test-set scores (for --figures-only)
    reports/results/metrics_comment_level.csv       Task A metrics with CIs
    reports/results/metrics_paired.csv              Task B metrics
    reports/results/ablation.csv                    feature-group ablation
    reports/results/feature_coefficients.csv        standardized LR coefficients (hand-crafted)
    reports/figures/*.png               PR curves and ablation chart
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import ALL_FEATURE_COLS, FEATURE_GROUPS, TEXT_FEATURE_COLS  # noqa: E402

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
FIGURES_DIR = os.path.join("reports", "figures")
RESULTS_DIR = os.path.join("reports", "results")  # committed: small metric tables
RANDOM_STATE = 42

TFIDF_PARAMS = dict(max_features=50000, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
# Stronger regularization for the 50K-dimensional TF-IDF models: C=1 overfits
# (grouped validation PR-AUC 0.091 at C=1 vs 0.106 at C=0.1 for the combined model).
TFIDF_C = 0.1
SENTIMENT_COLS = FEATURE_GROUPS["sentiment"]


# --- Data ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    comments = pd.read_parquet(os.path.join(PROCESSED_DIR, "comments.parquet"))
    features = pd.read_parquet(os.path.join(PROCESSED_DIR, "features.parquet"))
    overlap = [c for c in features.columns if c in comments.columns and c != "comment_id"]
    df = comments.drop(columns=overlap).merge(features, on="comment_id", how="inner", validate="one_to_one")
    df["reply_text"] = df["reply_text"].fillna("")
    return df


# --- Models -------------------------------------------------------------------

def make_lr(C=1.0):
    return LogisticRegression(max_iter=2000, C=C, class_weight="balanced", random_state=RANDOM_STATE)


def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=RANDOM_STATE,
    )


def fit_predict_lr_dense(X_tr, y_tr, X_ap):
    scaler = StandardScaler().fit(X_tr)
    clf = make_lr().fit(scaler.transform(X_tr), y_tr)
    return clf.predict_proba(scaler.transform(X_ap))[:, 1], (scaler, clf)


def fit_predict_hgb(X_tr, y_tr, X_ap):
    clf = make_hgb().fit(X_tr, y_tr)
    return clf.predict_proba(X_ap)[:, 1], clf


def fit_predict_text_models(tr: pd.DataFrame, ap: pd.DataFrame, label: str):
    """
    Fit every Task A base model on `tr` and score `ap`.
    Returns ({model_name: probs}, fitted objects for the deployable scorer).
    """
    y_tr = tr[label].values
    preds = {}

    preds["Sentiment only (LR)"], _ = fit_predict_lr_dense(
        tr[SENTIMENT_COLS].values, y_tr, ap[SENTIMENT_COLS].values)
    preds["Hand-crafted text (LR)"], _ = fit_predict_lr_dense(
        tr[TEXT_FEATURE_COLS].values, y_tr, ap[TEXT_FEATURE_COLS].values)
    preds["Hand-crafted text (HGB)"], _ = fit_predict_hgb(
        tr[TEXT_FEATURE_COLS].values, y_tr, ap[TEXT_FEATURE_COLS].values)
    preds["All features + context (HGB)"], _ = fit_predict_hgb(
        tr[ALL_FEATURE_COLS].values, y_tr, ap[ALL_FEATURE_COLS].values)

    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    T_tr = tfidf.fit_transform(tr["reply_text"])
    T_ap = tfidf.transform(ap["reply_text"])

    tfidf_clf = make_lr(TFIDF_C).fit(T_tr, y_tr)
    preds["TF-IDF (LR)"] = tfidf_clf.predict_proba(T_ap)[:, 1]

    scaler = StandardScaler().fit(tr[TEXT_FEATURE_COLS].values)
    C_tr = sparse.hstack([T_tr, scaler.transform(tr[TEXT_FEATURE_COLS].values)]).tocsr()
    C_ap = sparse.hstack([T_ap, scaler.transform(ap[TEXT_FEATURE_COLS].values)]).tocsr()
    combined_clf = make_lr(TFIDF_C).fit(C_tr, y_tr)
    preds["TF-IDF + hand-crafted text (LR)"] = combined_clf.predict_proba(C_ap)[:, 1]

    fitted = {"tfidf": tfidf, "scaler": scaler, "clf": combined_clf}
    return preds, fitted


STACK_COLS = ALL_FEATURE_COLS + ["tfidf_score"]


def stacked_oof(train_df, tfidf_oof, folds, label):
    """Out-of-fold predictions for the stacked HGB (used only to tune its threshold)."""
    X = np.column_stack([train_df[ALL_FEATURE_COLS].values, tfidf_oof])
    y = train_df[label].values
    oof = np.zeros(len(train_df))
    for tr_idx, va_idx in folds:
        oof[va_idx], _ = fit_predict_hgb(X[tr_idx], y[tr_idx], X[va_idx])
    return oof


# --- Evaluation ---------------------------------------------------------------

def best_f1_threshold(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precisions[:-1], recalls[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[np.argmax(f1)])


def precision_at_k(y_true, y_score, frac):
    k = max(int(round(len(y_true) * frac)), 1)
    top = np.argsort(-y_score)[:k]
    return float(np.mean(np.asarray(y_true)[top]))


def bootstrap_ci(metric_fn, y_true, y_score, groups, n_boot, seed=RANDOM_STATE):
    """95% CI of metric_fn, resampling whole groups (threads) with replacement."""
    if n_boot <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    codes, uniques = pd.factorize(groups)
    members = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(metric_fn(y_true[idx], y_score[idx]))
    return tuple(np.percentile(stats, [2.5, 97.5]))


def evaluate(name, y_test, score_test, threshold, groups, n_boot):
    y_pred = (score_test >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    pr_auc = average_precision_score(y_test, score_test)
    lo, hi = bootstrap_ci(average_precision_score, y_test, score_test, groups, n_boot)
    return {
        "model": name,
        "pr_auc": pr_auc, "pr_auc_ci_low": lo, "pr_auc_ci_high": hi,
        "roc_auc": roc_auc_score(y_test, score_test),
        "precision_at_1pct": precision_at_k(y_test, score_test, 0.01),
        "precision_at_5pct": precision_at_k(y_test, score_test, 0.05),
        "threshold_oof": threshold, "precision": p, "recall": r, "f1": f1,
    }


def pairwise_accuracy(df, score_col, label_col="thread_success"):
    """Fraction of (winner, loser) root-reply pairs where the winner scores higher (ties = 0.5)."""
    exploded = df.assign(pair_id=df["pair_ids"].str.split(";")).explode("pair_id")
    wins = []
    for _, g in exploded.groupby("pair_id"):
        pos, neg = g.loc[g[label_col] == 1, score_col].values, g.loc[g[label_col] == 0, score_col].values
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            wins.append(np.mean((diff > 0) + 0.5 * (diff == 0)))
    return float(np.mean(wins)), len(wins), np.asarray(wins)


# --- Tasks --------------------------------------------------------------------

def run_comment_level(df, n_folds, n_boot):
    label = "delta_awarded"
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"\n### Task A: comment-level  train={len(train)} ({train[label].mean():.2%} pos)  "
          f"test={len(test)} ({test[label].mean():.2%} pos)")

    folds = list(GroupKFold(n_splits=n_folds).split(train, groups=train["conversation_id"]))
    oof = {}
    for i, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()
        preds, _ = fit_predict_text_models(train.iloc[tr_idx], train.iloc[va_idx], label)
        for name, p in preds.items():
            oof.setdefault(name, np.zeros(len(train)))[va_idx] = p
        print(f"  fold {i + 1}/{n_folds} done ({time.time() - t0:.0f}s)")

    print("  fitting on full training set...")
    test_preds, scorer_parts = fit_predict_text_models(train, test, label)

    # Stacked model: HGB over all features + the TF-IDF model's out-of-fold score.
    stack_name = "Stacked (HGB + OOF TF-IDF score)"
    oof[stack_name] = stacked_oof(train, oof["TF-IDF (LR)"], folds, label)
    X_stack_tr = np.column_stack([train[ALL_FEATURE_COLS].values, oof["TF-IDF (LR)"]])
    X_stack_te = np.column_stack([test[ALL_FEATURE_COLS].values, test_preds["TF-IDF (LR)"]])
    test_preds[stack_name], _ = fit_predict_hgb(X_stack_tr, train[label].values, X_stack_te)

    results = []
    for name, score in test_preds.items():
        threshold = best_f1_threshold(train[label].values, oof[name])
        results.append(evaluate(name, test[label].values, score, threshold,
                                test["conversation_id"].values, n_boot))
    results = pd.DataFrame(results).sort_values("pr_auc", ascending=False)

    deploy_name = "TF-IDF + hand-crafted text (LR)"
    scorer = {
        **scorer_parts,
        "feature_cols": TEXT_FEATURE_COLS,
        "threshold": best_f1_threshold(train[label].values, oof[deploy_name]),
        "base_rate": float(train[label].mean()),
        "test_metrics": results.set_index("model").loc[deploy_name].to_dict(),
    }
    return results, test, test_preds, scorer


def run_ablation(df):
    """Full HGB model, minus each feature group, and each group alone (Task A, test PR-AUC)."""
    label = "delta_awarded"
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    y_tr, y_te = train[label].values, test[label].values

    def score(cols):
        p, _ = fit_predict_hgb(train[cols].values, y_tr, test[cols].values)
        return average_precision_score(y_te, p)

    full = score(ALL_FEATURE_COLS)
    rows = []
    for group, cols in FEATURE_GROUPS.items():
        without = [c for c in ALL_FEATURE_COLS if c not in cols]
        rows.append({
            "group": group, "n_features": len(cols),
            "pr_auc_alone": score(cols),
            "pr_auc_without": score(without),
        })
        rows[-1]["drop_in_pr_auc"] = full - rows[-1]["pr_auc_without"]
        print(f"  {group:<12} alone={rows[-1]['pr_auc_alone']:.4f}  "
              f"without={rows[-1]['pr_auc_without']:.4f}  (full={full:.4f})")
    out = pd.DataFrame(rows).sort_values("drop_in_pr_auc", ascending=False)
    out["full_pr_auc"] = full
    return out


def run_paired(df, n_boot):
    """Task B: paired root replies. Train on paired train rows, report pairwise accuracy on test."""
    label = "thread_success"
    paired = df[df[label].notna()].copy()
    paired[label] = paired[label].astype(int)
    train, test = paired[paired["split"] == "train"], paired[paired["split"] == "test"].copy()
    print(f"\n### Task B: paired root replies  train={len(train)}  test={len(test)}")

    y_tr = train[label].values
    models = {
        "Sentiment only (LR)": lambda: fit_predict_lr_dense(
            train[SENTIMENT_COLS].values, y_tr, test[SENTIMENT_COLS].values)[0],
        "Hand-crafted text (LR)": lambda: fit_predict_lr_dense(
            train[TEXT_FEATURE_COLS].values, y_tr, test[TEXT_FEATURE_COLS].values)[0],
        "Interplay only (LR)": lambda: fit_predict_lr_dense(
            train[FEATURE_GROUPS["interplay"]].values, y_tr, test[FEATURE_GROUPS["interplay"]].values)[0],
        "All features + context (HGB)": lambda: fit_predict_hgb(
            train[ALL_FEATURE_COLS].values, y_tr, test[ALL_FEATURE_COLS].values)[0],
    }

    def tfidf_combined():
        tfidf = TfidfVectorizer(**{**TFIDF_PARAMS, "min_df": 3})
        scaler = StandardScaler().fit(train[TEXT_FEATURE_COLS].values)
        X_tr = sparse.hstack([tfidf.fit_transform(train["reply_text"]),
                              scaler.transform(train[TEXT_FEATURE_COLS].values)]).tocsr()
        X_te = sparse.hstack([tfidf.transform(test["reply_text"]),
                              scaler.transform(test[TEXT_FEATURE_COLS].values)]).tocsr()
        clf = LogisticRegression(max_iter=2000, C=0.3, random_state=RANDOM_STATE).fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]

    models["TF-IDF + hand-crafted text (LR)"] = tfidf_combined

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, fn in models.items():
        test["score"] = fn()
        acc, n_pairs, wins = pairwise_accuracy(test, "score")
        boots = [rng.choice(wins, len(wins)).mean() for _ in range(max(n_boot, 1))]
        rows.append({
            "model": name, "pairwise_accuracy": acc,
            "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
            "roc_auc": roc_auc_score(test[label], test["score"]), "n_test_pairs": n_pairs,
        })
    return pd.DataFrame(rows).sort_values("pairwise_accuracy", ascending=False)


def lr_coefficients(df):
    """Standardized LR coefficients on the hand-crafted text features (Task A, full train)."""
    train = df[df["split"] == "train"]
    _, (scaler, clf) = fit_predict_lr_dense(
        train[TEXT_FEATURE_COLS].values, train["delta_awarded"].values, train[TEXT_FEATURE_COLS].values[:1])
    group_of = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    coefs = pd.DataFrame({"feature": TEXT_FEATURE_COLS, "coefficient": clf.coef_[0]})
    coefs["group"] = coefs["feature"].map(group_of)
    return coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index)


# --- Figures ------------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figures(test, test_preds, ablation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    y = test["delta_awarded"].values

    shown = ["Stacked (HGB + OOF TF-IDF score)", "TF-IDF + hand-crafted text (LR)",
             "Hand-crafted text (HGB)", "Sentiment only (LR)"]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style(ax)
    for name, color in zip(shown, SERIES):
        prec, rec, _ = precision_recall_curve(y, test_preds[name])
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name}  (PR-AUC {average_precision_score(y, test_preds[name]):.3f})")
    ax.axhline(y.mean(), color=INK_2, linewidth=1, linestyle="--")
    ax.text(0.01, y.mean() - 0.008, f"chance ({y.mean():.1%})", color=INK_2, fontsize=8, ha="left", va="top")
    ax.set_xlabel("Recall", color=INK)
    ax.set_ylabel("Precision", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, None)
    ax.set_title("Precision-recall on held-out threads (comment-level task)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "pr_curves.png"), dpi=150)
    plt.close(fig)

    if ablation is not None:
        ab = ablation.sort_values("drop_in_pr_auc")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.barh(ab["group"], ab["drop_in_pr_auc"], color=SERIES[0], height=0.6)
        ax.axvline(0, color=INK_2, linewidth=1)
        for i, v in enumerate(ab["drop_in_pr_auc"]):
            ax.text(max(v, 0) + 0.0008, i, f"{v:+.3f}", va="center", ha="left", color=INK, fontsize=8)
        ax.set_xlabel("Drop in test PR-AUC when the group is removed", color=INK)
        ax.set_title(f"Feature-group ablation (full model PR-AUC {ablation['full_pr_auc'].iloc[0]:.3f})",
                     color=INK, loc="left", fontsize=11)
        ax.tick_params(axis="y", colors=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "ablation.png"), dpi=150)
        plt.close(fig)


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw figures from saved test predictions and ablation.csv")
    args = parser.parse_args()

    preds_path = os.path.join(MODELS_DIR, "test_predictions_comment_level.parquet")
    ablation_path = os.path.join(RESULTS_DIR, "ablation.csv")
    if args.figures_only:
        preds = pd.read_parquet(preds_path)
        ablation = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else None
        save_figures(preds, {c: preds[c].values for c in preds.columns if c != "delta_awarded"}, ablation)
        print(f"Redrew figures in {FIGURES_DIR}/")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.set_option("display.width", 200)
    df = load_data()
    print(f"Loaded {len(df)} challenger comments.")

    results, test, test_preds, scorer = run_comment_level(df, args.folds, args.bootstrap)
    results.to_csv(os.path.join(RESULTS_DIR, "metrics_comment_level.csv"), index=False)
    joblib.dump(scorer, os.path.join(MODELS_DIR, "scorer.joblib"))
    pd.DataFrame({"delta_awarded": test["delta_awarded"].values, **test_preds}).to_parquet(preds_path, index=False)
    print("\n=== Task A: comment-level (test threads) ===")
    print(results.round(4).to_string(index=False))

    coefs = lr_coefficients(df)
    coefs.to_csv(os.path.join(RESULTS_DIR, "feature_coefficients.csv"), index=False)
    print("\n=== Standardized coefficients, hand-crafted text LR ===")
    print(coefs.round(3).to_string(index=False))

    ablation = None
    if not args.skip_ablation:
        print("\n=== Feature-group ablation (HGB, test PR-AUC) ===")
        ablation = run_ablation(df)
        ablation.to_csv(ablation_path, index=False)
    elif os.path.exists(ablation_path):
        ablation = pd.read_csv(ablation_path)

    paired = run_paired(df, args.bootstrap)
    paired.to_csv(os.path.join(RESULTS_DIR, "metrics_paired.csv"), index=False)
    print("\n=== Task B: paired root replies (test pairs) ===")
    print(paired.round(4).to_string(index=False))

    save_figures(test, test_preds, ablation)
    print(f"\nSaved models to {MODELS_DIR}/, results to {RESULTS_DIR}/, figures to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
