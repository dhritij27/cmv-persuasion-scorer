"""
features.py

Extracts interpretable features from CMV challenger comments, organized
into named groups so train.py can run group ablations (e.g. "does
sentiment add anything once the other features are known?").

Groups
------
- length:        size and shape of the reply
- readability:   reading ease, word length
- hedging:       hedges vs. certainty markers (rates per 100 words)
- argument:      evidence, examples, counterargument acknowledgment, links, quoting
- interaction:   pronoun use and questions (how the reply addresses the OP)
- formatting:    lists, emphasis, numbers
- sentiment:     VADER sentiment (tests whether *tone* predicts persuasion)
- interplay:     vocabulary overlap with the OP (Tan et al. 2016 "interplay" features)
- context:       position in the discussion (depth, timing, entry order) -- not
                 derived from the text, so excluded from the deployable scorer

Count-style features are expressed as rates per 100 words so they are not
just proxies for length (which is captured explicitly by the length group).

Usage:
    python src/features.py

Input:
    data/processed/comments.parquet, data/processed/ops.parquet  (from data_loader.py)

Output:
    data/processed/features.parquet  (comment_id + one column per feature)
"""

import os
import re
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd
import textstat
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROCESSED_DIR = os.path.join("data", "processed")
COMMENTS_PATH = os.path.join(PROCESSED_DIR, "comments.parquet")
OPS_PATH = os.path.join(PROCESSED_DIR, "ops.parquet")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "features.parquet")

# --- Lexicons ---------------------------------------------------------------

HEDGES = [
    "might", "may", "could", "perhaps", "possibly", "seem", "seems", "suggest",
    "suggests", "somewhat", "arguably", "likely", "unlikely", "appears", "appear",
    "presumably", "probably", "maybe", "i think", "i believe", "in my opinion",
    "it's possible", "generally", "often", "tend to", "tends to",
]
CERTAINTY = [
    "definitely", "certainly", "obviously", "clearly", "undoubtedly", "always",
    "never", "absolutely", "surely", "without question", "undeniably",
    "unquestionably", "of course", "no doubt", "everyone knows", "the fact is",
]
EVIDENCE = [
    "study", "studies", "research", "data", "statistics", "evidence", "source",
    "sources", "according to", "survey", "report", "found that", "percent",
    "experiment", "peer reviewed", "published",
]
EXAMPLES = ["for example", "for instance", "e.g.", "such as", "consider", "imagine"]
COUNTERARGUMENT = [
    "i understand", "i see your point", "you're right that", "you're right about",
    "you are right", "granted", "that said", "however", "on the other hand",
    "while it's true", "although", "even so", "nevertheless", "i agree",
    "fair point", "to be fair",
]
CONTRAST = ["but", "yet", "though", "instead", "rather"]
FIRST_SINGULAR = ["i", "me", "my", "mine", "myself", "i'm", "i've", "i'd"]
FIRST_PLURAL = ["we", "us", "our", "ours", "ourselves", "we're", "we've"]
SECOND_PERSON = ["you", "your", "yours", "yourself", "you're", "you've", "you'd"]

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)

FEATURE_GROUPS = {
    "length": ["log_word_count", "log_sentence_count", "avg_sentence_length", "paragraph_count"],
    "readability": ["readability", "avg_word_length", "type_token_ratio"],
    "hedging": ["hedge_rate", "certainty_rate"],
    "argument": [
        "evidence_rate", "example_rate", "counterargument_rate", "contrast_rate",
        "link_count", "quote_line_count", "quote_fraction",
    ],
    "interaction": ["first_singular_rate", "first_plural_rate", "second_person_rate", "question_rate"],
    "formatting": ["list_item_count", "emphasis_count", "number_rate"],
    "sentiment": ["sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_intensity"],
    "interplay": [
        "op_jaccard_all", "op_jaccard_content", "op_jaccard_stop",
        "op_content_coverage", "reply_content_novelty",
    ],
    "context": ["depth", "is_root_reply", "replies_to_op", "log_minutes_since_op", "log_root_rank"],
}
TEXT_FEATURE_COLS = [c for g, cols in FEATURE_GROUPS.items() if g != "context" for c in cols]
ALL_FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]

# --- Helpers ----------------------------------------------------------------

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")
LINK_RE = re.compile(r"https?://|www\.")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M)
EMPHASIS_RE = re.compile(r"\*\*[^*\n]+\*\*|\*[^*\n]+\*|__[^_\n]+__")
NUMBER_RE = re.compile(r"\b\d[\d,.]*%?")


def _phrase_regex(phrases):
    """One compiled, word-boundary-anchored regex matching any phrase."""
    alts = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    return re.compile(r"(?<![\w'])(?:" + "|".join(alts) + r")(?![\w'])")


LEXICON_RES = {
    name: _phrase_regex(words)
    for name, words in {
        "hedge": HEDGES, "certainty": CERTAINTY, "evidence": EVIDENCE,
        "example": EXAMPLES, "counterargument": COUNTERARGUMENT, "contrast": CONTRAST,
        "first_singular": FIRST_SINGULAR, "first_plural": FIRST_PLURAL,
        "second_person": SECOND_PERSON,
    }.items()
}


def tokenize(text: str):
    return WORD_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _vader():
    return SentimentIntensityAnalyzer()


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _strip_quotes(text: str) -> str:
    """Remove quoted lines ('> ...') so features describe the reply's own words."""
    return "\n".join(ln for ln in text.split("\n") if not ln.lstrip().startswith(">"))


# --- Feature extraction -----------------------------------------------------

def extract_text_features(op_text: str, reply_text: str) -> dict:
    """All non-context features for one (op_text, reply_text) pair."""
    reply_text = reply_text or ""
    op_text = op_text or ""

    lines = reply_text.split("\n")
    quote_lines = [ln for ln in lines if ln.lstrip().startswith(">")]
    own_text = _strip_quotes(reply_text)
    own_lower = own_text.lower()

    words = tokenize(own_text)
    n_words = len(words)
    per100 = 100.0 / max(n_words, 1)
    sentences = [s for s in SENTENCE_SPLIT_RE.split(own_text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    paragraphs = [p for p in re.split(r"\n\s*\n", own_text) if p.strip()]

    def rate(name):
        return len(LEXICON_RES[name].findall(own_lower)) * per100

    try:
        readability = float(textstat.flesch_reading_ease(own_text)) if n_words else 0.0
    except Exception:
        readability = 0.0
    readability = float(np.clip(readability, -100, 121))  # tame extreme values on tiny/odd texts

    sentiment = _vader().polarity_scores(own_text[:5000])  # cap for speed on very long replies

    op_words = set(tokenize(op_text))
    reply_words = set(words)
    op_content, reply_content = op_words - STOP_WORDS, reply_words - STOP_WORDS
    op_stop, reply_stop = op_words & STOP_WORDS, reply_words & STOP_WORDS

    total_chars = max(len(reply_text), 1)

    return {
        # length
        "log_word_count": np.log1p(n_words),
        "log_sentence_count": np.log1p(n_sentences),
        "avg_sentence_length": n_words / n_sentences,
        "paragraph_count": max(len(paragraphs), 1),
        # readability
        "readability": readability,
        "avg_word_length": (sum(map(len, words)) / n_words) if n_words else 0.0,
        "type_token_ratio": (len(reply_words) / n_words) if n_words else 0.0,
        # hedging
        "hedge_rate": rate("hedge"),
        "certainty_rate": rate("certainty"),
        # argument
        "evidence_rate": rate("evidence"),
        "example_rate": rate("example"),
        "counterargument_rate": rate("counterargument"),
        "contrast_rate": rate("contrast"),
        "link_count": len(LINK_RE.findall(reply_text)),
        "quote_line_count": len(quote_lines),
        "quote_fraction": sum(len(ln) for ln in quote_lines) / total_chars,
        # interaction
        "first_singular_rate": rate("first_singular"),
        "first_plural_rate": rate("first_plural"),
        "second_person_rate": rate("second_person"),
        "question_rate": own_text.count("?") * per100,
        # formatting
        "list_item_count": len(LIST_ITEM_RE.findall(own_text)),
        "emphasis_count": len(EMPHASIS_RE.findall(own_text)),
        "number_rate": len(NUMBER_RE.findall(own_text)) * per100,
        # sentiment
        "sentiment_compound": sentiment["compound"],
        "sentiment_pos": sentiment["pos"],
        "sentiment_neg": sentiment["neg"],
        "sentiment_intensity": abs(sentiment["compound"]),
        # interplay with OP
        "op_jaccard_all": _jaccard(op_words, reply_words),
        "op_jaccard_content": _jaccard(op_content, reply_content),
        "op_jaccard_stop": _jaccard(op_stop, reply_stop),
        "op_content_coverage": (len(op_content & reply_content) / len(op_content)) if op_content else 0.0,
        "reply_content_novelty": (len(reply_content - op_content) / len(reply_content)) if reply_content else 0.0,
    }


def context_features(depth, is_root_reply, replies_to_op, minutes_since_op, root_rank) -> dict:
    """Features describing where/when the comment was posted, not what it says."""
    return {
        "depth": float(depth),
        "is_root_reply": float(is_root_reply),
        "replies_to_op": float(replies_to_op),
        "log_minutes_since_op": np.log1p(max(float(minutes_since_op), 0.0)),
        # non-root comments have no entry rank; 0 is a distinct sentinel after log1p
        "log_root_rank": 0.0 if pd.isna(root_rank) else np.log1p(float(root_rank)),
    }


def _extract_row(args):
    op_text, reply_text = args
    return extract_text_features(op_text, reply_text)


def main():
    comments = pd.read_parquet(COMMENTS_PATH)
    ops = pd.read_parquet(OPS_PATH).set_index("conversation_id")["op_text"]
    print(f"Extracting features for {len(comments)} comments...")

    jobs = list(zip(comments["conversation_id"].map(ops), comments["reply_text"]))
    with Pool(processes=max(os.cpu_count() - 1, 1)) as pool:
        text_rows = pool.map(_extract_row, jobs, chunksize=500)

    text_df = pd.DataFrame(text_rows)
    ctx_df = pd.DataFrame([
        context_features(r.depth, r.is_root_reply, r.replies_to_op, r.minutes_since_op, r.root_rank)
        for r in comments.itertuples(index=False)
    ])
    out = pd.concat([comments[["comment_id"]], text_df, ctx_df], axis=1)[["comment_id"] + ALL_FEATURE_COLS]
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")
    print(out[ALL_FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
