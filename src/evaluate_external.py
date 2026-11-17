"""
evaluate_external.py

Out-of-domain check of the deployed scorer on Persuasive-Pairs
(Pauli et al., 2024; Hugging Face: APauli/Persuasive-Pairs).

Each item is a pair of short texts with the same content: an original and an
LLM rewrite made more or less persuasive. Three annotators rated which text
is more persuasive on a -3..+3 scale (positive = text2). Because content is
held fixed, this tests whether the CMV-trained scorer has learned anything
about persuasive *language* -- separate from length, topic, or conversation
context, which drive most of its CMV performance.

This dataset is used for evaluation only, never training: its label (which
paraphrase *sounds* more persuasive) differs from CMV's (did a real person
change their mind), and 447 pairs are rewrites of texts from the same CMV
corpus the scorer was trained on, so results are reported per source with
that subset flagged.

Baselines:
    - longer text wins       length is the scorer's strongest single feature
    - sentiment (VADER)      more positive text wins

Usage:
    python src/evaluate_external.py [--bootstrap 1000]

Output:
    reports/results/persuasive_pairs_eval.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_text_features  # noqa: E402
from predict import load_scorer, score  # noqa: E402

HF_DATASET = "APauli/Persuasive-Pairs"
RESULTS_PATH = os.path.join("reports", "results", "persuasive_pairs_eval.csv")
SCORE_COLS = ["score_1", "score_2", "score_3"]
RANDOM_STATE = 42


def load_pairs() -> pd.DataFrame:
    from datasets import load_dataset
    df = load_dataset(HF_DATASET, split="train").to_pandas()
    df["human_mean"] = df[SCORE_COLS].mean(axis=1)
    df = df[df["human_mean"] != 0].reset_index(drop=True)  # drop exact ties
    df["text2_wins"] = (df["human_mean"] > 0).astype(int)
    signs = np.sign(df[SCORE_COLS])
    df["unanimous"] = signs.nunique(axis=1) == 1
    return df


def pairwise(pred_text2_wins, truth, ties=None):
    """Accuracy of predicting the winner; ties (if given) count as half."""
    correct = (pred_text2_wins == truth).astype(float)
    if ties is not None:
        correct[ties] = 0.5
    return correct


def bootstrap_ci(correct, n_boot):
    rng = np.random.default_rng(RANDOM_STATE)
    boots = [rng.choice(correct, len(correct)).mean() for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scorer on Persuasive-Pairs.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    df = load_pairs()
    print(f"{len(df)} non-tied pairs from {df['source'].nunique()} sources.")
    scorer = load_scorer()

    s1, s2, sent1, sent2, n1, n2 = [], [], [], [], [], []
    for r in df.itertuples(index=False):
        for text, s_out, sent_out, n_out in ((r.text1, s1, sent1, n1), (r.text2, s2, sent2, n2)):
            # No OP is available, so OP-interplay features are computed against an empty post.
            s_out.append(score(scorer, "", text)["score"])
            f = extract_text_features("", text)
            sent_out.append(f["sentiment_compound"])
            n_out.append(f["log_word_count"])
    s1, s2, sent1, sent2, n1, n2 = map(np.array, (s1, s2, sent1, sent2, n1, n2))

    truth = df["text2_wins"].values
    methods = {
        "CMV scorer (TF-IDF + text features)": pairwise((s2 > s1).astype(int), truth, s1 == s2),
        "Longer text wins": pairwise((n2 > n1).astype(int), truth, n1 == n2),
        "More positive sentiment wins": pairwise((sent2 > sent1).astype(int), truth, sent1 == sent2),
    }

    subsets = {"All pairs": np.ones(len(df), bool),
               "Unanimous annotators": df["unanimous"].values}
    for src in sorted(df["source"].unique()):
        label = f"Source: {src}" + (" (same corpus as CMV training data)" if src == "Winning-Arguments" else "")
        subsets[label] = (df["source"] == src).values
    subsets["All sources except Winning-Arguments"] = (df["source"] != "Winning-Arguments").values

    rows = []
    for subset, mask in subsets.items():
        for method, correct in methods.items():
            lo, hi = bootstrap_ci(correct[mask], args.bootstrap)
            rows.append({"subset": subset, "method": method, "n_pairs": int(mask.sum()),
                         "pairwise_accuracy": correct[mask].mean(), "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    out.to_csv(RESULTS_PATH, index=False)

    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
