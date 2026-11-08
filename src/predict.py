"""
predict.py

Score how persuasive a reply is likely to be against a given original post,
and explain which features and words drove the score.

Uses models/scorer.joblib (TF-IDF + hand-crafted text features, logistic
regression), produced by train.py. Only text is used, so it works on any
draft reply -- no thread metadata needed.

Usage:
    python src/predict.py --op "CMV: ..." --reply "I see your point, but..."
    python src/predict.py --op-file op.txt --reply-file reply.txt [--top 8]

Note: the model was trained with balanced class weights, so the score is a
ranking signal, not a calibrated probability. Compare it to the decision
threshold (tuned out-of-fold during training) rather than reading it as a %.
"""

import argparse
import os
import sys

import joblib
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import clean_text  # noqa: E402
from features import extract_text_features  # noqa: E402

DEFAULT_MODEL = os.path.join("models", "scorer.joblib")


def load_scorer(path=DEFAULT_MODEL):
    if not os.path.exists(path):
        sys.exit(f"No model at {path}. Run src/data_loader.py, src/features.py, src/train.py first.")
    return joblib.load(path)


def score(scorer, op_text: str, reply_text: str, top: int = 8) -> dict:
    op_text, reply_text = clean_text(op_text), clean_text(reply_text)
    cols = scorer["feature_cols"]
    feats = extract_text_features(op_text, reply_text)
    x_hand = scorer["scaler"].transform(np.array([[feats[c] for c in cols]]))
    x_tfidf = scorer["tfidf"].transform([reply_text])
    x = sparse.hstack([x_tfidf, x_hand]).tocsr()

    clf = scorer["clf"]
    prob = float(clf.predict_proba(x)[0, 1])
    coef = clf.coef_[0]
    n_terms = x_tfidf.shape[1]

    # Contribution of each input to the logit = coefficient * (scaled) value.
    hand_contrib = coef[n_terms:] * x_hand[0]
    vocab = scorer["tfidf"].get_feature_names_out()
    nz = x_tfidf.indices
    term_contrib = sorted(
        ((vocab[j], coef[j] * x_tfidf[0, j]) for j in nz), key=lambda t: t[1]
    )

    order = np.argsort(hand_contrib)
    return {
        "score": prob,
        "threshold": scorer["threshold"],
        "predicted_persuasive": prob >= scorer["threshold"],
        "features": feats,
        "top_positive_features": [(cols[i], hand_contrib[i]) for i in order[::-1][:top] if hand_contrib[i] > 0],
        "top_negative_features": [(cols[i], hand_contrib[i]) for i in order[:top] if hand_contrib[i] < 0],
        "top_positive_terms": [t for t in term_contrib[::-1][:top] if t[1] > 0],
        "top_negative_terms": [t for t in term_contrib[:top] if t[1] < 0],
    }


def _read(text, path):
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    return text or ""


def main():
    parser = argparse.ArgumentParser(description="Score a CMV reply's persuasiveness.")
    parser.add_argument("--op", help="original post text")
    parser.add_argument("--op-file", help="file containing the original post")
    parser.add_argument("--reply", help="reply text")
    parser.add_argument("--reply-file", help="file containing the reply")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    op_text, reply_text = _read(args.op, args.op_file), _read(args.reply, args.reply_file)
    if not reply_text.strip():
        parser.error("provide --reply or --reply-file")

    result = score(load_scorer(args.model), op_text, reply_text, args.top)
    verdict = "likely persuasive" if result["predicted_persuasive"] else "unlikely to persuade"
    print(f"\nScore: {result['score']:.3f}   (decision threshold {result['threshold']:.3f}) -> {verdict}\n")

    def show(title, items):
        if items:
            print(title)
            for name, val in items:
                print(f"  {val:+.3f}  {name}")

    show("Features pushing the score up:", result["top_positive_features"])
    show("Features pushing the score down:", result["top_negative_features"])
    show("Words/phrases pushing up:", result["top_positive_terms"])
    show("Words/phrases pushing down:", result["top_negative_terms"])


if __name__ == "__main__":
    main()
