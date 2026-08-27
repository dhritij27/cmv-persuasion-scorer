# Persuasion Scorer: What Makes an Argument Actually Change Minds?

A machine learning system that predicts argument persuasiveness — not sentiment, not toxicity, but actual mind-changing power — trained on real outcomes from Reddit's r/changemyview.

## The Problem

Most NLP projects classify *tone* (positive/negative, toxic/civil). Almost none tackle *persuasion* directly, because it's genuinely harder: a persuasive argument can be blunt, polite, short, or long — the signal isn't in surface sentiment at all. This project asks a more interesting question: **what linguistic and structural features actually correlate with changing someone's mind?**

## The Approach

Using the Change My View subreddit as a natural experiment — where users explicitly award a "Δ" (delta) when a reply changes their view — this project builds a classifier that predicts whether a given reply will be persuasive, based on hand-engineered linguistic, structural, and argumentative features rather than relying purely on black-box embeddings.

Features include:
- **Linguistic**: word count, sentence length, readability, hedging language ("might", "perhaps"), certainty markers
- **Structural**: paragraph count, rebuttal/quote presence, rhetorical questions
- **Argumentative**: counterargument acknowledgment, evidence markers, citations
- **Style-matching**: vocabulary overlap between reply and original post

A key open question this project is positioned to test — not yet implemented — is whether persuasion is a distinct signal from sentiment: whether positive or negative tone alone is actually a weak predictor of whether an argument works. That comparison is a natural next step (see Results for what's been measured so far).

## Why It Matters

Understanding persuasion computationally has real applications: content moderation, debate/argument analysis tools, writing assistants, and misinformation research all depend on being able to separate *how something sounds* from *how effective it is*.

## Tech Stack

- `scikit-learn` — modeling and evaluation
- `ConvoKit` (Cornell Conversational Analysis Toolkit) — dataset loading
- `pandas` / `numpy` — data handling
- `nltk` / `textstat` — linguistic feature extraction

## Project Structure

```
cmv-persuasion-scorer/
├── data/               # raw/processed data (gitignored)
├── notebooks/          # exploration notebooks
├── src/
│   ├── data_loader.py  # loads and preps ConvoKit corpus
│   ├── features.py     # feature engineering
│   └── train.py        # model training + evaluation
├── models/              # saved model artifacts (gitignored)
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/cmv-persuasion-scorer.git
cd cmv-persuasion-scorer
pip install -r requirements.txt
```

## Usage

```bash
python src/data_loader.py     # downloads and preps the CMV corpus
python src/features.py        # extracts linguistic/structural/argumentative features
python src/train.py           # trains baseline, hand-crafted, and combined models; evaluates
```

## Results

Three models were trained and compared on a held-out 20% test set (stratified, ~290K total replies, ~4.3% labeled persuasive):

| Model | PR-AUC | ROC-AUC | F1 (tuned threshold) |
|---|---|---|---|
| Hand-crafted features only (Logistic Regression) | 0.104 | 0.724 | 0.177 |
| TF-IDF only (Logistic Regression) | 0.251 | 0.826 | 0.313 |
| **Combined (TF-IDF + hand-crafted)** | **0.255** | **0.834** | **0.326** |

*(Full metrics, including precision/recall at both default and F1-optimal thresholds, are in `models/metrics.csv` after running `train.py`.)*

**Class imbalance.** Only ~4.3% of replies in the dataset earned a delta, so plain accuracy is meaningless here (a model that never predicts "persuasive" scores ~96% accuracy and is useless). All models use `class_weight='balanced'`, and metrics are reported as precision/recall/F1 on the minority class plus PR-AUC, alongside performance at the classification threshold that maximizes F1 rather than the default 0.5 cutoff.

**Takeaway 1: word choice carries most of the signal, but structure adds real value on top.** TF-IDF alone already captures most of what's predictive; the hand-crafted linguistic/structural features are too weak to stand alone (PR-AUC 0.104), but combining them with TF-IDF beats TF-IDF alone on every metric — confirming that persuasion has structural signal beyond word choice, even if that signal is modest relative to vocabulary.

**Takeaway 2: mimicking the original poster's vocabulary is the single strongest hand-crafted predictor.** In the standalone feature model, `op_reply_vocab_overlap` (vocabulary overlap between the reply and the original post) had by far the largest positive coefficient — more than double the next strongest feature. Persuasive replies tend to speak the OP's language, literally.

**Takeaway 3: hedging helps, certainty doesn't.** Counterintuitively, hedging language ("might," "perhaps") and counterargument acknowledgment ("I see your point, but...") were both positively associated with persuasion, while certainty markers ("definitely," "clearly") had almost no effect. This pushes back on the intuitive assumption that confident, assertive arguments are what change minds — engaging with the other side and qualifying claims correlates more strongly with success than sounding sure of yourself.

**Takeaway 4: readability beats sophistication.** Simpler, more readable replies (higher Flesch reading ease) were more persuasive than complex ones — being easy to follow matters more than sounding authoritative.

**Takeaway 5: rhetorical questions and quoting the OP were mildly counterproductive.** Both had small negative coefficients, suggesting these tactics may read as combative or lecturing rather than persuasive.

## License

MIT