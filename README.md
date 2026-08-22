# cmv-persuasion-scorer
Predicts argument persuasiveness (not sentiment) using linguistic and structural features, trained on real persuasion outcomes from r/changemyview. Built to show that what changes minds isn't the same as what sounds nice.
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

A key part of the analysis is an explicit comparison against sentiment-based baselines, to empirically demonstrate that **persuasion is a distinct signal from sentiment** — positive or negative tone alone is a weak predictor of whether an argument actually works.

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
python src/train.py           # extracts features, trains model, evaluates
```

## Results

*(to be filled in as the project develops — feature importances, accuracy vs. baselines, ablation results)*

## License

MIT
