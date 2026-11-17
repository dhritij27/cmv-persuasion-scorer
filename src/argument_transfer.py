"""
argument_transfer.py

Transfer learning from the PERSUADE 2.0 corpus (The Learning Agency Lab):
~8,400 student argumentative essays, segmented into discourse elements
(Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement)
that expert annotators rated Effective / Non-Effective.

PERSUADE's label ("an expert judged this well argued") is a different target
from CMV's ("a real person changed their mind"), and the texts are student
essays rather than Reddit comments, so the two datasets are NOT pooled.
Instead, two models are trained on PERSUADE and applied to CMV comments to
produce new features:

    1. Discourse-role model    text chunk -> {Claim, Evidence, Counterclaim, ...}
    2. Effectiveness model     (text chunk, role) -> P(Effective)

Each CMV comment is split into paragraph-sized chunks (PERSUADE segments
average 20-70 words), each chunk is tagged and rated, and the results are
aggregated into the "argument_structure" feature group (role shares, evidence
and rebuttal counts, mean/max effectiveness). train.py then measures whether
expert-rated argument quality adds anything to predicting real persuasion.

Evaluation of the PERSUADE models uses leave-one-prompt-out CV: CMV topics
are unseen by these models, so cross-topic accuracy is the relevant number.

Usage:
    python src/argument_transfer.py [--path /path/to/persuade2_train_srctexts.csv]

    Without --path, the dataset is downloaded with kagglehub
    (julesking/tla-lab-persuade-dataset).

Outputs:
    data/processed/persuade_segments.parquet
    models/argument_models.joblib
    reports/results/persuade_role_metrics.csv
    reports/results/persuade_effectiveness_metrics.csv
"""

import argparse
import glob
import os
import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

KAGGLE_DATASET = "julesking/tla-lab-persuade-dataset"
CSV_NAME = "persuade2_train_srctexts.csv"
SEGMENTS_PATH = os.path.join("data", "processed", "persuade_segments.parquet")
MODEL_PATH = os.path.join("models", "argument_models.joblib")
RESULTS_DIR = os.path.join("reports", "results")
RANDOM_STATE = 42

ROLES = ["Lead", "Position", "Claim", "Counterclaim", "Rebuttal", "Evidence",
         "Concluding Statement", "Other"]
MIN_OTHER_WORDS = 4  # unannotated fragments shorter than this are headings/noise

# Feature columns produced for CMV comments (consumed by features.py).
TRANSFER_FEATURE_COLS = [
    "arg_n_chunks",
    "arg_share_claim", "arg_share_evidence", "arg_share_counterclaim",
    "arg_share_rebuttal", "arg_share_position", "arg_share_other",
    "arg_n_evidence", "arg_has_counter_and_rebuttal",
    "arg_effectiveness_mean", "arg_effectiveness_max",
]

TFIDF_PARAMS = dict(max_features=40000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)


# --- PERSUADE data --------------------------------------------------------------

def find_csv(path=None):
    if path:
        return path
    import kagglehub
    root = kagglehub.dataset_download(KAGGLE_DATASET)
    matches = glob.glob(os.path.join(root, "**", CSV_NAME), recursive=True)
    if not matches:
        raise FileNotFoundError(f"{CSV_NAME} not found under {root}")
    return matches[0]


def load_segments(csv_path) -> pd.DataFrame:
    """One row per discourse segment; unannotated text becomes role 'Other'."""
    df = pd.read_csv(
        csv_path,
        usecols=["essay_id_comp", "prompt_name", "discourse_type",
                 "discourse_text", "discourse_effectiveness"],
    )
    df = df.rename(columns={"essay_id_comp": "essay_id"})
    df["discourse_text"] = df["discourse_text"].fillna("").str.strip()
    df["n_words"] = df["discourse_text"].str.split().str.len()
    df["role"] = df["discourse_type"].replace({"Unannotated": "Other"})
    df = df[(df["role"] != "Other") | (df["n_words"] >= MIN_OTHER_WORDS)]
    df = df[df["n_words"] > 0]
    df["effective"] = (df["discourse_effectiveness"] == "Effective").astype(float)
    df.loc[df["role"] == "Other", "effective"] = np.nan  # unannotated: no rating
    return df.reset_index(drop=True)


# --- Models -----------------------------------------------------------------------

def _role_onehot(roles):
    idx = pd.Categorical(roles, categories=ROLES).codes
    m = np.zeros((len(idx), len(ROLES)))
    m[np.arange(len(idx)), idx] = 1.0
    return m


def _length_feats(texts):
    n = np.array([len(t.split()) for t in texts], dtype=float)
    return np.column_stack([np.log1p(n), np.log1p(n) ** 2 / 10])


def fit_role_model(texts, roles):
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X = sparse.hstack([vec.fit_transform(texts), _length_feats(texts)]).tocsr()
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced",
                             random_state=RANDOM_STATE).fit(X, roles)
    return {"vec": vec, "clf": clf}


def predict_roles(model, texts):
    X = sparse.hstack([model["vec"].transform(texts), _length_feats(texts)]).tocsr()
    proba = model["clf"].predict_proba(X)
    return model["clf"].classes_[proba.argmax(1)], proba


def fit_effectiveness_model(texts, roles, y):
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X = sparse.hstack([vec.fit_transform(texts), _role_onehot(roles),
                       _length_feats(texts)]).tocsr()
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                             random_state=RANDOM_STATE).fit(X, y)
    return {"vec": vec, "clf": clf}


def predict_effectiveness(model, texts, roles):
    X = sparse.hstack([model["vec"].transform(texts), _role_onehot(roles),
                       _length_feats(texts)]).tocsr()
    return model["clf"].predict_proba(X)[:, 1]


def evaluate_leave_prompt_out(seg):
    """Cross-topic evaluation: train on 7 prompts, test on the held-out one."""
    role_rows, eff_rows = [], []
    rated = seg["effective"].notna()
    for prompt in sorted(seg["prompt_name"].unique()):
        tr, te = seg["prompt_name"] != prompt, seg["prompt_name"] == prompt

        rm = fit_role_model(seg.loc[tr, "discourse_text"].tolist(), seg.loc[tr, "role"].values)
        pred, _ = predict_roles(rm, seg.loc[te, "discourse_text"].tolist())
        y_role = seg.loc[te, "role"].values
        role_rows.append({
            "held_out_prompt": prompt, "n": int(te.sum()),
            "accuracy": accuracy_score(y_role, pred),
            "macro_f1": f1_score(y_role, pred, average="macro"),
        })

        tr_e, te_e = tr & rated, te & rated
        em = fit_effectiveness_model(seg.loc[tr_e, "discourse_text"].tolist(),
                                     seg.loc[tr_e, "role"].values, seg.loc[tr_e, "effective"].values)
        # Score with *predicted* roles, as happens on CMV.
        te_texts = seg.loc[te_e, "discourse_text"].tolist()
        te_roles, _ = predict_roles(rm, te_texts)
        p = predict_effectiveness(em, te_texts, te_roles)
        y = seg.loc[te_e, "effective"].values
        eff_rows.append({
            "held_out_prompt": prompt, "n": int(te_e.sum()), "pos_rate": y.mean(),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        })
        print(f"  {prompt[:40]:<40} role acc={role_rows[-1]['accuracy']:.3f}  "
              f"macro-F1={role_rows[-1]['macro_f1']:.3f}  eff ROC-AUC={eff_rows[-1]['roc_auc']:.3f}")
    return pd.DataFrame(role_rows), pd.DataFrame(eff_rows)


# --- Applying to CMV ----------------------------------------------------------------

SENT_RE = re.compile(r"(?<=[.!?])\s+")
MAX_CHUNK_WORDS = 90


def split_chunks(text: str):
    """Split a comment into paragraph-sized chunks, excluding quoted lines."""
    own = "\n".join(ln for ln in (text or "").split("\n") if not ln.lstrip().startswith(">"))
    chunks = []
    for para in re.split(r"\n\s*\n", own):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para.split()) <= MAX_CHUNK_WORDS:
            chunks.append(para)
            continue
        current = []
        for sent in SENT_RE.split(para):
            current.append(sent)
            if sum(len(s.split()) for s in current) >= MAX_CHUNK_WORDS // 2:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
    return chunks


@lru_cache(maxsize=1)
def load_models(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python src/argument_transfer.py` first to train "
            "the PERSUADE argument models.")
    return joblib.load(path)


def transfer_features(texts, models=None) -> pd.DataFrame:
    """Batch-compute argument-structure features for a list of comment texts."""
    models = models or load_models()
    owner, chunks = [], []
    for i, t in enumerate(texts):
        cs = split_chunks(t)
        owner.extend([i] * len(cs))
        chunks.extend(cs)
    owner = np.array(owner, dtype=int)

    out = pd.DataFrame(0.0, index=range(len(texts)), columns=TRANSFER_FEATURE_COLS)
    if not chunks:
        return out

    roles, _ = predict_roles(models["role"], chunks)
    eff = predict_effectiveness(models["effectiveness"], chunks, roles)
    ch = pd.DataFrame({"owner": owner, "role": roles, "eff": eff})

    g = ch.groupby("owner")
    n = g.size()
    share = pd.crosstab(ch["owner"], ch["role"]).reindex(columns=ROLES, fill_value=0).div(n, axis=0)
    # Effectiveness is only meaningful for argumentative chunks.
    argumentative = ch[ch["role"] != "Other"].groupby("owner")["eff"]

    out.loc[n.index, "arg_n_chunks"] = np.log1p(n)
    for role, col in [("Claim", "arg_share_claim"), ("Evidence", "arg_share_evidence"),
                      ("Counterclaim", "arg_share_counterclaim"), ("Rebuttal", "arg_share_rebuttal"),
                      ("Position", "arg_share_position"), ("Other", "arg_share_other")]:
        out.loc[share.index, col] = share[role]
    counts = pd.crosstab(ch["owner"], ch["role"]).reindex(columns=ROLES, fill_value=0)
    out.loc[counts.index, "arg_n_evidence"] = counts["Evidence"]
    out.loc[counts.index, "arg_has_counter_and_rebuttal"] = (
        (counts["Counterclaim"] > 0) & (counts["Rebuttal"] > 0)).astype(float)
    mean_eff, max_eff = argumentative.mean(), argumentative.max()
    out.loc[mean_eff.index, "arg_effectiveness_mean"] = mean_eff
    out.loc[max_eff.index, "arg_effectiveness_max"] = max_eff
    return out


# --- Main ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PERSUADE argument-role/effectiveness models.")
    parser.add_argument("--path", help=f"path to {CSV_NAME} (default: download via kagglehub)")
    parser.add_argument("--skip-eval", action="store_true", help="skip leave-one-prompt-out evaluation")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(SEGMENTS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    seg = load_segments(find_csv(args.path))
    seg.to_parquet(SEGMENTS_PATH, index=False)
    print(f"Loaded {len(seg)} segments from {seg['essay_id'].nunique()} essays, "
          f"{seg['prompt_name'].nunique()} prompts.")
    print(seg["role"].value_counts().to_string())

    if not args.skip_eval:
        print("\nLeave-one-prompt-out evaluation:")
        role_m, eff_m = evaluate_leave_prompt_out(seg)
        role_m.to_csv(os.path.join(RESULTS_DIR, "persuade_role_metrics.csv"), index=False)
        eff_m.to_csv(os.path.join(RESULTS_DIR, "persuade_effectiveness_metrics.csv"), index=False)
        print(f"\nMean across held-out prompts: role accuracy={role_m['accuracy'].mean():.3f}, "
              f"macro-F1={role_m['macro_f1'].mean():.3f}; "
              f"effectiveness ROC-AUC={eff_m['roc_auc'].mean():.3f}")

    print("\nFitting final models on all prompts...")
    rated = seg[seg["effective"].notna()]
    models = {
        "role": fit_role_model(seg["discourse_text"].tolist(), seg["role"].values),
        "effectiveness": fit_effectiveness_model(
            rated["discourse_text"].tolist(), rated["role"].values, rated["effective"].values),
    }
    joblib.dump(models, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
