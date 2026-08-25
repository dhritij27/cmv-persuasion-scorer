"""
features.py

Extracts linguistic, structural, and argumentative features from CMV
(op_text, reply_text) pairs, for use in predicting persuasiveness.

Usage:
    python src/features.py

Input:
    data/processed/cmv_pairs.csv   (produced by data_loader.py)

Output:
    data/processed/cmv_features.csv
"""

import os
import re
import pandas as pd
import numpy as np
import textstat

INPUT_PATH = os.path.join("data", "processed", "cmv_pairs.csv")
OUTPUT_PATH = os.path.join("data", "processed", "cmv_features.csv")

# --- Word lists used for linguistic/argumentative features -----------------

HEDGE_WORDS = {
    "might", "may", "could", "perhaps", "possibly", "seem", "seems",
    "suggest", "suggests", "somewhat", "arguably", "likely", "unlikely",
    "appears", "appear", "presumably", "probably", "maybe",
}

CERTAINTY_WORDS = {
    "definitely", "certainly", "obviously", "clearly", "undoubtedly",
    "always", "never", "absolutely", "surely", "without question",
    "undeniably", "unquestionably",
}

EVIDENCE_MARKERS = {
    "study", "studies", "research", "data", "statistics", "evidence",
    "source", "according to", "survey", "report", "found that",
}

COUNTERARGUMENT_PHRASES = {
    "i understand", "i see your point", "you're right that", "you're right about",
    "granted", "that said", "however", "but", "on the other hand",
    "while it's true", "although", "even so", "nevertheless",
}


def _word_tokenize(text: str):
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def _count_phrase_hits(text_lower: str, phrases: set) -> int:
    """Count how many of the given words/phrases appear in the text."""
    count = 0
    for phrase in phrases:
        count += text_lower.count(phrase)
    return count


def extract_features(op_text: str, reply_text: str) -> dict:
    """
    Extract a single row of features for one (op_text, reply_text) pair.
    """
    reply_text = reply_text or ""
    op_text = op_text or ""

    reply_lower = reply_text.lower()
    words = _word_tokenize(reply_text)
    word_count = len(words)
    sentences = re.split(r"[.!?]+", reply_text)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    paragraphs = [p for p in reply_text.split("\n\n") if p.strip()]

    # --- Linguistic features ---
    avg_sentence_length = word_count / sentence_count
    try:
        readability = textstat.flesch_reading_ease(reply_text) if word_count > 0 else 0.0
    except Exception:
        readability = 0.0

    hedge_count = _count_phrase_hits(reply_lower, HEDGE_WORDS)
    certainty_count = _count_phrase_hits(reply_lower, CERTAINTY_WORDS)

    # --- Structural features ---
    paragraph_count = max(len(paragraphs), 1)
    question_count = reply_text.count("?")
    has_quote = int(">" in reply_text or '"' in reply_text)

    # --- Argumentative features ---
    evidence_count = _count_phrase_hits(reply_lower, EVIDENCE_MARKERS)
    counterargument_count = _count_phrase_hits(reply_lower, COUNTERARGUMENT_PHRASES)
    has_citation_like = int(bool(re.search(r"http[s]?://|www\.", reply_text)))

    # --- Style-matching feature: vocabulary overlap between OP and reply ---
    op_words = set(_word_tokenize(op_text))
    reply_words = set(words)
    if op_words and reply_words:
        overlap = len(op_words & reply_words) / len(op_words | reply_words)
    else:
        overlap = 0.0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "readability": readability,
        "hedge_count": hedge_count,
        "certainty_count": certainty_count,
        "paragraph_count": paragraph_count,
        "question_count": question_count,
        "has_quote": has_quote,
        "evidence_count": evidence_count,
        "counterargument_count": counterargument_count,
        "has_citation_like": has_citation_like,
        "op_reply_vocab_overlap": overlap,
    }


def main():
    print(f"Loading pairs from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows. Extracting features...")

    feature_rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        feats = extract_features(row.op_text, row.reply_text)
        feature_rows.append(feats)
        if (i + 1) % 20000 == 0:
            print(f"  processed {i + 1}/{len(df)} rows...")

    feature_df = pd.DataFrame(feature_rows)
    result = pd.concat(
        [df[["conversation_id", "persuasive"]].reset_index(drop=True), feature_df],
        axis=1,
    )

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved features to {OUTPUT_PATH}")
    print(result.describe())


if __name__ == "__main__":
    main()
    