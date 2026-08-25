"""
data_loader.py

Downloads the Change My View (CMV) "winning-args-corpus" via ConvoKit,
and converts it into a flat pandas DataFrame of (reply_text, original_post_text,
persuasive) rows suitable for feature extraction and modeling.

Usage:
    python src/data_loader.py

Output:
    data/processed/cmv_pairs.csv
"""

import os
import pandas as pd
from convokit import Corpus, download


RAW_CORPUS_NAME = "winning-args-corpus"
OUTPUT_PATH = os.path.join("data", "processed", "cmv_pairs.csv")


def load_corpus() -> Corpus:
    """Download (or load cached) the CMV winning-args-corpus."""
    print(f"Loading corpus: {RAW_CORPUS_NAME} (this may take a few minutes on first run)...")
    corpus = Corpus(filename=download(RAW_CORPUS_NAME))
    print("Corpus loaded.")
    print(corpus)
    return corpus


def build_pairs_dataframe(corpus: Corpus) -> pd.DataFrame:
    """
    Walk the corpus conversations and build a flat DataFrame where each row is:
        - conversation_id: the CMV thread this reply belongs to
        - op_text: the original post text (the view being argued about)
        - reply_text: a reply to that post
        - persuasive: 1 if this reply earned a delta (changed OP's mind), else 0

    ConvoKit's winning-args-corpus stores delta info in utterance metadata.
    We treat each root post as the OP text, and each direct/indirect reply
    utterance as a candidate persuasive/non-persuasive example.
    """
    rows = []

    for convo in corpus.iter_conversations():
        utterances = list(convo.iter_utterances())
        if not utterances:
            continue

        # The root utterance is the original post (the view under debate)
        root = convo.get_utterance(convo.id)
        op_text = root.text if root and root.text else ""

        if not op_text.strip():
            continue

        for utt in utterances:
            # Skip the OP's own post — we only want replies
            if utt.id == convo.id:
                continue
            if not utt.text or not utt.text.strip():
                continue

            # ConvoKit marks delta-awarded utterances in metadata; the exact
            # key can vary by corpus version, so we check a couple of
            # common conventions defensively.
            meta = utt.meta
            persuasive = int(
                bool(meta.get("success"))
                or bool(meta.get("delta"))
                or bool(meta.get("has_delta"))
            )

            rows.append(
                {
                    "conversation_id": convo.id,
                    "op_text": op_text,
                    "reply_text": utt.text,
                    "persuasive": persuasive,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Built {len(df)} (op, reply) pairs.")
    if not df.empty:
        print("Class balance:")
        print(df["persuasive"].value_counts(normalize=True))
    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    corpus = load_corpus()
    df = build_pairs_dataframe(corpus)

    if df.empty:
        print("Warning: no rows extracted. Check corpus metadata keys with "
              "corpus.print_summary_stats() or inspect utt.meta manually.")
        return

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved processed data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()