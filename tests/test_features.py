import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from argument_transfer import (  # noqa: E402
    ROLES,
    TRANSFER_FEATURE_COLS,
    fit_effectiveness_model,
    fit_role_model,
    split_chunks,
    transfer_features,
)
from data_loader import clean_text  # noqa: E402
from features import (  # noqa: E402
    ALL_FEATURE_COLS,
    FEATURE_GROUPS,
    HANDCRAFTED_TEXT_COLS,
    TEXT_FEATURE_COLS,
    context_features,
    extract_text_features,
)
from evaluate_external import pairwise  # noqa: E402
from train import best_f1_threshold, pairwise_accuracy  # noqa: E402

OP = "CMV: Public transit should be free. Fares are a regressive tax on the poor."


def test_returns_every_handcrafted_text_feature():
    feats = extract_text_features(OP, "I think fares matter.")
    assert set(feats) == set(HANDCRAFTED_TEXT_COLS)
    assert all(np.isfinite(v) for v in feats.values())


def test_feature_groups_are_disjoint_and_complete():
    cols = [c for group in FEATURE_GROUPS.values() for c in group]
    assert len(cols) == len(set(cols)) == len(ALL_FEATURE_COLS)
    assert set(TEXT_FEATURE_COLS) == set(HANDCRAFTED_TEXT_COLS) | set(TRANSFER_FEATURE_COLS)
    assert not set(FEATURE_GROUPS["context"]) & set(TEXT_FEATURE_COLS)


def test_lexicons_match_whole_words_only():
    # "but" must not match inside "attribute"/"button"; "may" not inside "dismay".
    feats = extract_text_features(OP, "The attribute on the button caused dismay")
    assert feats["contrast_rate"] == 0
    assert feats["hedge_rate"] == 0
    feats = extract_text_features(OP, "It may work, but maybe not")
    assert feats["hedge_rate"] > 0 and feats["contrast_rate"] > 0


def test_rates_are_length_normalized():
    short = extract_text_features(OP, "Perhaps fares help.")
    long = extract_text_features(OP, " ".join(["Perhaps fares help."] * 10))
    assert short["hedge_rate"] == pytest.approx(long["hedge_rate"])


def test_quotes_detected_after_html_unescape_and_excluded_from_own_words():
    reply = clean_text("&gt; Fares are a regressive tax\n\nActually, studies show otherwise.")
    feats = extract_text_features(OP, reply)
    assert feats["quote_line_count"] == 1
    assert feats["evidence_rate"] > 0
    # quoted OP words should not inflate the reply's own vocabulary overlap
    no_quote = extract_text_features(OP, "Actually, studies show otherwise.")
    assert feats["op_jaccard_content"] == pytest.approx(no_quote["op_jaccard_content"])


def test_vocab_overlap_increases_when_reusing_op_words():
    low = extract_text_features(OP, "Bananas are yellow and delicious.")
    high = extract_text_features(OP, "Free public transit fares are not a regressive tax.")
    assert high["op_jaccard_content"] > low["op_jaccard_content"]
    assert high["op_content_coverage"] > low["op_content_coverage"]


def test_sentiment_signs():
    pos = extract_text_features(OP, "This is a wonderful, thoughtful, great point.")
    neg = extract_text_features(OP, "This is a terrible, stupid, awful argument.")
    assert pos["sentiment_compound"] > 0 > neg["sentiment_compound"]


def test_empty_reply_is_safe():
    feats = extract_text_features(OP, "")
    assert all(np.isfinite(v) for v in feats.values())


def test_clean_text_strips_delta_lines():
    text = "Good argument here.\nEDIT: thanks for the ∆!\nMore argument."
    assert clean_text(text) == "Good argument here.\nMore argument."
    assert "delta" not in clean_text("Point one.\nEdit: got a delta, thanks").lower()


def test_context_features_handle_missing_rank():
    ctx = context_features(depth=3, is_root_reply=0, replies_to_op=1, minutes_since_op=60, root_rank=np.nan)
    assert ctx["log_root_rank"] == 0.0
    assert set(ctx) == set(FEATURE_GROUPS["context"])


def test_best_f1_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, s)
    assert 0.3 < t <= 0.8


def test_pairwise_accuracy():
    df = pd.DataFrame({
        "pair_ids": ["p1", "p1", "p2", "p2"],
        "thread_success": [1, 0, 1, 0],
        "score": [0.9, 0.1, 0.2, 0.7],
    })
    acc, n, _ = pairwise_accuracy(df, "score")
    assert n == 2 and acc == pytest.approx(0.5)


# --- Argument transfer (PERSUADE) -------------------------------------------

def test_split_chunks_drops_quotes_and_splits_long_paragraphs():
    text = "> quoted line\n\nFirst paragraph.\n\n" + " ".join(["Sentence number %d." % i for i in range(60)])
    chunks = split_chunks(text)
    assert chunks[0] == "First paragraph."
    assert all(not c.startswith(">") for c in chunks)
    assert len(chunks) >= 3  # the 180-word paragraph is split into pieces
    assert split_chunks("") == []


@pytest.fixture(scope="module")
def tiny_models():
    """Tiny PERSUADE-style models fit on synthetic segments, one or more per role."""
    rows = []
    for role in ROLES:
        for i in range(4):
            rows.append((f"{role.lower()} text about topic {i} and {role.lower()} words", role, float(i % 2)))
    texts, roles, eff = zip(*rows)
    return {
        "role": fit_role_model(list(texts), np.array(roles)),
        "effectiveness": fit_effectiveness_model(list(texts), np.array(roles), np.array(eff)),
    }


def test_transfer_features_shape_and_ranges(tiny_models):
    texts = ["claim text about topic 1.\n\nevidence text about topic 2.", "", "rebuttal words"]
    out = transfer_features(texts, models=tiny_models)
    assert list(out.columns) == TRANSFER_FEATURE_COLS and len(out) == 3
    assert np.isfinite(out.values).all()
    share_cols = [c for c in TRANSFER_FEATURE_COLS if c.startswith("arg_share_")]
    # shares cover claim/evidence/counterclaim/rebuttal/position/other (not lead/conclusion),
    # so they sum to at most 1 for non-empty texts and exactly 0 for empty ones
    assert out.loc[0, share_cols].sum() <= 1.0 + 1e-9
    assert (out.loc[1] == 0).all()
    assert 0.0 <= out.loc[0, "arg_effectiveness_mean"] <= out.loc[0, "arg_effectiveness_max"] <= 1.0
    assert out.loc[0, "arg_n_chunks"] == pytest.approx(np.log1p(2))


def test_external_pairwise_counts_ties_as_half():
    pred = np.array([1, 0, 1, 0])
    truth = np.array([1, 1, 0, 0])
    ties = np.array([False, False, False, True])
    correct = pairwise(pred, truth, ties)
    assert list(correct) == [1.0, 0.0, 0.0, 0.5]
