"""
data_loader.py

Loads the Change My View "winning-args-corpus" via ConvoKit and flattens it
into one row per *challenger comment* (a comment written by someone other
than the OP), with a comment-level persuasion label.

Labeling
--------
The corpus' own `success` field is a *thread-level* flag from the paired
design of Tan et al. (2016): it is set on every utterance in a paired
discussion thread, including the OP's own replies. Using it as a
comment-level label marks OP comments (e.g. "∆ Enjoy your delta") as
"persuasive", which leaks the answer into the training data.

Instead, `delta_awarded` is derived from DeltaBot confirmations:

    challenger comment  <-  OP comment awarding ∆  <-  DeltaBot "Confirmed: 1 delta awarded to /u/<challenger>"

A comment is labeled persuasive only if the OP awarded it a delta that
DeltaBot confirmed, and the confirmed username matches the comment author.

The thread-level `success` flag is still kept (`thread_success`) for root
replies, to support the paired evaluation in train.py.

Cleaning
--------
- Drops comments by the OP, DeltaBot, AutoModerator, and deleted/removed comments.
- HTML-unescapes text (Reddit stores quotes as "&gt;").
- Removes lines that mention a delta (e.g. "EDIT: thanks for the ∆!"), which
  challengers often add *after* winning, so they leak the label.

Usage:
    python src/data_loader.py

Output:
    data/processed/comments.parquet   one row per challenger comment
    data/processed/ops.parquet        one row per original post
"""

import html
import os
import re

import numpy as np
import pandas as pd
from convokit import Corpus, download

RAW_CORPUS_NAME = "winning-args-corpus"
PROCESSED_DIR = os.path.join("data", "processed")
COMMENTS_PATH = os.path.join(PROCESSED_DIR, "comments.parquet")
OPS_PATH = os.path.join(PROCESSED_DIR, "ops.parquet")

EXCLUDED_SPEAKERS = {"DeltaBot", "AutoModerator", "[deleted]"}
EMPTY_TEXTS = {"", "[deleted]", "[removed]"}

DELTABOT_CONFIRM_RE = re.compile(r"Confirmed:\s*1 delta awarded to /u/([\w-]+)", re.I)
DELTA_MENTION_RE = re.compile(r"∆|Δ|&#8710;|!delta\b|\bdeltas?\b", re.I)


def clean_text(text: str) -> str:
    """HTML-unescape and drop any line that mentions a delta."""
    text = html.unescape(text or "")
    lines = [ln for ln in text.split("\n") if not DELTA_MENTION_RE.search(ln)]
    return "\n".join(lines).strip()


def load_corpus() -> Corpus:
    """Download (or load cached) the CMV winning-args-corpus."""
    print(f"Loading corpus: {RAW_CORPUS_NAME} (this may take a few minutes on first run)...")
    corpus = Corpus(filename=download(RAW_CORPUS_NAME))
    corpus.print_summary_stats()
    return corpus


def find_delta_awarded(corpus: Corpus) -> set:
    """Return IDs of comments that the OP awarded a DeltaBot-confirmed delta."""
    awarded = set()
    for utt in corpus.iter_utterances():
        if utt.speaker.id != "DeltaBot":
            continue
        match = DELTABOT_CONFIRM_RE.search(utt.text or "")
        if not match or utt.reply_to is None:
            continue
        try:
            award = corpus.get_utterance(utt.reply_to)
            if award.reply_to is None:
                continue
            target = corpus.get_utterance(award.reply_to)
        except KeyError:
            continue
        op_id = corpus.get_conversation(utt.conversation_id).meta["op-userID"]
        if award.speaker.id != op_id:
            continue  # only deltas awarded by the OP count as "changed OP's view"
        if target.speaker.id.lower() != match.group(1).lower():
            continue
        awarded.add(target.id)
    return awarded


def _depths(utts_by_id: dict, root_id: str) -> dict:
    """Depth of each utterance below the root post (root replies have depth 1)."""
    depths = {root_id: 0}
    for start in utts_by_id:
        chain, uid = [], start
        while uid not in depths:
            chain.append(uid)
            parent = utts_by_id[uid].reply_to
            if parent is None or parent not in utts_by_id:
                parent = root_id  # orphaned comment: treat as top level
            uid = parent
        d = depths[uid]
        for cid in reversed(chain):
            d += 1
            depths[cid] = d
    return depths


def build_dataframes(corpus: Corpus):
    awarded = find_delta_awarded(corpus)
    print(f"Found {len(awarded)} DeltaBot-confirmed, OP-awarded comments.")

    comment_rows, op_rows = [], []
    for convo in corpus.iter_conversations():
        root = convo.get_utterance(convo.id)
        op_id = convo.meta["op-userID"]
        op_text = clean_text(convo.meta.get("op-text-body") or root.text)
        if not op_text:
            continue
        split = "train" if convo.meta.get("train") == 1 else "test"
        op_rows.append({
            "conversation_id": convo.id,
            "op_title": convo.meta.get("op-title", ""),
            "op_text": op_text,
            "split": split,
        })

        utts = {u.id: u for u in convo.iter_utterances()}
        depths = _depths(utts, convo.id)
        # Root posts carry no timestamp in this corpus, so use the first
        # comment's time as a proxy for when the thread opened.
        comment_ts = [u.timestamp for u in utts.values() if u.timestamp]
        root_ts = root.timestamp or (min(comment_ts) if comment_ts else 0)
        root_replies = sorted(
            (u for u in utts.values() if u.reply_to == convo.id),
            key=lambda u: u.timestamp or 0,
        )
        root_rank = {u.id: i + 1 for i, u in enumerate(root_replies)}

        for utt in utts.values():
            if utt.id == convo.id or utt.speaker.id == op_id or utt.speaker.id in EXCLUDED_SPEAKERS:
                continue
            raw = (utt.text or "").strip()
            if raw in EMPTY_TEXTS:
                continue
            text = clean_text(raw)
            if not text:
                continue

            is_root = utt.reply_to == convo.id
            parent = utts.get(utt.reply_to)
            replies_to_op = int(is_root or (parent is not None and parent.speaker.id == op_id))
            success = utt.meta.get("success")
            comment_rows.append({
                "comment_id": utt.id,
                "conversation_id": convo.id,
                "split": split,
                "speaker": utt.speaker.id,
                "reply_text": text,
                "depth": depths.get(utt.id, 1),
                "is_root_reply": int(is_root),
                "replies_to_op": replies_to_op,
                "root_rank": root_rank.get(utt.id, np.nan),
                "minutes_since_op": max((utt.timestamp or root_ts) - root_ts, 0) / 60.0,
                "thread_success": success if (is_root and success in (0, 1)) else np.nan,
                "pair_ids": ";".join(utt.meta.get("pair_ids") or []) if is_root else "",
                "delta_awarded": int(utt.id in awarded),
            })

    comments = pd.DataFrame(comment_rows)
    ops = pd.DataFrame(op_rows)
    comments = comments[comments["conversation_id"].isin(ops["conversation_id"])].reset_index(drop=True)
    return comments, ops


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    corpus = load_corpus()
    comments, ops = build_dataframes(corpus)

    print(f"\n{len(comments)} challenger comments across {len(ops)} threads.")
    print(f"Delta-awarded rate: {comments['delta_awarded'].mean():.4f} "
          f"({comments['delta_awarded'].sum()} comments)")
    print("Comments per split:\n", comments["split"].value_counts())
    paired = comments["thread_success"].notna()
    print(f"Paired root replies (Tan et al. task): {paired.sum()}")

    comments.to_parquet(COMMENTS_PATH, index=False)
    ops.to_parquet(OPS_PATH, index=False)
    print(f"Saved {COMMENTS_PATH} and {OPS_PATH}")


if __name__ == "__main__":
    main()
