"""
train.py

Trains and evaluates three models for predicting CMV reply persuasiveness:
    1. Baseline: TF-IDF + LogisticRegression
    2. Hand-crafted features only: LogisticRegression
    3. Combined: TF-IDF + hand-crafted features + LogisticRegression

All three use LogisticRegression with class_weight='balanced' so the
~4.3% positive class is handled via loss reweighting rather than
resampling. (An earlier RandomForest version of model #2 was tried and
discarded — it achieved near-zero recall on the minority class despite
class_weight='balanced', since tree-vote balancing is much less reliable
than direct loss reweighting on this kind of imbalance.)

Reports precision/recall/F1 on the minority class plus PR-AUC and ROC-AUC
at both the default 0.5 threshold and an F1-optimal threshold found via
the precision-recall curve, since plain accuracy is meaningless on this
distribution and 0.5 is rarely the best cutoff for imbalanced data.

Usage:
    python src/train.py

Inputs:
    data/processed/cmv_pairs.csv      (op_text, reply_text, persuasive)
    data/processed/cmv_features.csv   (hand-crafted features, persuasive)

Output:
    models/baseline_tfidf.joblib
    models/handcrafted_features.joblib
    models/combined.joblib
    Printed evaluation report + saved metrics to models/metrics.csv
"""

import os
import pandas as pd
import numpy as np
from scipy import sparse
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

PAIRS_PATH = os.path.join("data", "processed", "cmv_pairs.csv")
FEATURES_PATH = os.path.join("data", "processed", "cmv_features.csv")
MODELS_DIR = "models"
RANDOM_STATE = 42

HANDCRAFTED_COLS = [
    "word_count", "sentence_count", "avg_sentence_length", "readability",
    "hedge_count", "certainty_count", "paragraph_count", "question_count",
    "has_quote", "evidence_count", "counterargument_count",
    "has_citation_like", "op_reply_vocab_overlap",
]


def load_data():
    pairs = pd.read_csv(PAIRS_PATH)
    features = pd.read_csv(FEATURES_PATH)

    # pairs and features were built in the same row order, so we can
    # align on position — but reset indices defensively either way.
    pairs = pairs.reset_index(drop=True)
    features = features.reset_index(drop=True)

    assert len(pairs) == len(features), (
        f"Row count mismatch: pairs={len(pairs)} vs features={len(features)}. "
        "Re-run data_loader.py and features.py to regenerate in sync."
    )

    df = pd.concat(
        [pairs[["reply_text", "persuasive"]], features[HANDCRAFTED_COLS]],
        axis=1,
    )
    df["reply_text"] = df["reply_text"].fillna("")
    return df


def find_best_threshold(y_true, y_proba):
    """
    Find the probability threshold that maximizes F1 on the persuasive
    class, using the precision-recall curve. Returns (best_threshold, best_f1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds, so drop the last precision/recall to align.
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12),
        0.0,
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate(name, y_true, y_pred, y_proba):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["not persuasive", "persuasive"], zero_division=0))
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Also report metrics at the F1-optimal threshold, since the default
    # 0.5 cutoff is often a poor choice on heavily imbalanced data.
    best_threshold, best_f1 = find_best_threshold(y_true, y_proba)
    y_pred_tuned = (y_proba >= best_threshold).astype(int)
    t_precision, t_recall, t_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_tuned, average="binary", zero_division=0
    )
    print(f"\n--- At F1-optimal threshold ({best_threshold:.3f}) ---")
    print(f"Precision: {t_precision:.4f}  Recall: {t_recall:.4f}  F1: {t_f1:.4f}")

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "best_threshold": best_threshold,
        "tuned_precision": t_precision,
        "tuned_recall": t_recall,
        "tuned_f1": t_f1,
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows. Class balance:\n{df['persuasive'].value_counts(normalize=True)}")

    X_text = df["reply_text"]
    X_handcrafted = df[HANDCRAFTED_COLS]
    y = df["persuasive"]

    # Single stratified split, reused (by index) across all three models
    # so results are directly comparable.
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = []

    
    # 1. Baseline: TF-IDF + Logistic Regression
   
    print("\nTraining baseline (TF-IDF + LogisticRegression)...")
    tfidf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5)),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    tfidf_pipeline.fit(X_text.loc[idx_train], y.loc[idx_train])
    y_pred = tfidf_pipeline.predict(X_text.loc[idx_test])
    y_proba = tfidf_pipeline.predict_proba(X_text.loc[idx_test])[:, 1]
    results.append(evaluate("Baseline (TF-IDF only)", y.loc[idx_test], y_pred, y_proba))
    joblib.dump(tfidf_pipeline, os.path.join(MODELS_DIR, "baseline_tfidf.joblib"))

   
    # 2. Hand-crafted features only
    # Using LogisticRegression here (not RandomForest) because balanced
    # class weighting is applied directly to the loss function, which
    # handles the ~4%/96% imbalance far more reliably than tree-based
    # voting does at default settings. An earlier RandomForest version
    # of this model achieved near-zero recall on the minority class
    # despite class_weight='balanced' — see README/notes for that ablation.
    
    print("\nTraining hand-crafted features model (LogisticRegression)...")
    handcrafted_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    handcrafted_pipeline.fit(X_handcrafted.loc[idx_train], y.loc[idx_train])
    y_pred = handcrafted_pipeline.predict(X_handcrafted.loc[idx_test])
    y_proba = handcrafted_pipeline.predict_proba(X_handcrafted.loc[idx_test])[:, 1]
    results.append(evaluate("Hand-crafted features only", y.loc[idx_test], y_pred, y_proba))
    joblib.dump(handcrafted_pipeline, os.path.join(MODELS_DIR, "handcrafted_features.joblib"))

    # Feature importances via coefficients, the key interpretability
    # payoff of this project. Coefficients are on standardized features,
    # so magnitude is directly comparable across features.
    coefs = pd.Series(
        handcrafted_pipeline.named_steps["clf"].coef_[0],
        index=HANDCRAFTED_COLS,
    ).sort_values(key=lambda s: s.abs(), ascending=False)
    print("\nFeature coefficients (hand-crafted model, standardized):")
    print(coefs)
    coefs.to_csv(os.path.join(MODELS_DIR, "feature_importances.csv"))

    
    # 3. Combined: TF-IDF + hand-crafted features
    
    print("\nTraining combined model (TF-IDF + hand-crafted)...")

    # Fit TF-IDF on train text, transform both splits, then hstack with
    # the (scaled) hand-crafted features.
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5)
    X_tfidf_train = tfidf.fit_transform(X_text.loc[idx_train])
    X_tfidf_test = tfidf.transform(X_text.loc[idx_test])

    scaler = StandardScaler()
    X_hand_train = scaler.fit_transform(X_handcrafted.loc[idx_train])
    X_hand_test = scaler.transform(X_handcrafted.loc[idx_test])

    X_combined_train = sparse.hstack([X_tfidf_train, X_hand_train]).tocsr()
    X_combined_test = sparse.hstack([X_tfidf_test, X_hand_test]).tocsr()

    combined_clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    combined_clf.fit(X_combined_train, y.loc[idx_train])
    y_pred = combined_clf.predict(X_combined_test)
    y_proba = combined_clf.predict_proba(X_combined_test)[:, 1]
    results.append(evaluate("Combined (TF-IDF + hand-crafted)", y.loc[idx_test], y_pred, y_proba))

    joblib.dump(
        {"tfidf": tfidf, "scaler": scaler, "clf": combined_clf},
        os.path.join(MODELS_DIR, "combined.joblib"),
    )

    
    # Summary
   
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODELS_DIR, "metrics.csv"), index=False)
    print("\n=== Summary (default 0.5 threshold) ===")
    print(results_df[["model", "precision", "recall", "f1", "pr_auc", "roc_auc"]].to_string(index=False))
    print("\n=== Summary (F1-optimal threshold per model) ===")
    print(results_df[["model", "best_threshold", "tuned_precision", "tuned_recall", "tuned_f1"]].to_string(index=False))
    print(f"\nSaved models to {MODELS_DIR}/, metrics to {MODELS_DIR}/metrics.csv")


if __name__ == "__main__":
    main()