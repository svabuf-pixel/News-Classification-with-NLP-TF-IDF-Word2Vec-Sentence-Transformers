# 📰 News Classification with NLP — TF-IDF, Word2Vec & Sentence Transformers

A comparative study of three NLP pipelines for binary news classification (real vs. fake), evaluated across accuracy, F1, Cohen's κ, ROC-AUC, and 5-fold cross-validation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Models](#models)
- [Results](#results)
- [Key Visualisations](#key-visualisations)
- [Setup & Usage](#setup--usage)
- [Dependencies](#dependencies)
- [Design Decisions](#design-decisions)

---

## Overview

This project benchmarks three end-to-end NLP classification pipelines on a binary news dataset. Each pipeline uses a different text representation strategy — sparse bag-of-words, static word embeddings, and contextual sentence embeddings — all paired with a **LinearSVC** classifier for a controlled comparison.

| Representation | Classifier | Accuracy | Weighted F1 | ROC-AUC |
|---|---|---|---|---|
| TF-IDF (bigrams) | LinearSVC | 93.78% | 93.78% | 0.9849 |
| Word2Vec (TF-IDF weighted) | LinearSVC | 87.60% | 87.60% | 0.9415 |
| all-MiniLM-L6-v2 | LinearSVC | **94.06%** | **94.06%** | **0.9850** |

---

## Dataset

- **Training set**: ~34,152 samples (17,572 class 0 · 16,580 class 1)
- **Format**: Single-column CSV; each row is `<label><text>` (label is the first character)
- **Balance**: Near-perfectly balanced — no class weighting required, but `class_weight='balanced'` is applied defensively via the `IMBALANCE_THRESHOLD = 0.15` guard
- **Domain**: Political news (dominant tokens: *trump*, *video*, *obama*, *hillary*, *republican*, *russia*)

---

## Project Structure

```
project/
├── dataset/
│   ├── training_data.csv
│   ├── testing_data.csv
│   ├── testing_data_predicted_tfidf.csv        # Model 1 predictions
│   ├── testing_data_predicted_w2v.csv          # Model 2 predictions
│   ├── testing_data_predicted_transformer.csv  # Model 3 predictions
│   └── model_comparison.csv                    # Metrics summary table
└── nlp_classifier.py                           # Main script (all models)
```

---

## Pipeline Architecture

### Preprocessing

Three dedicated preprocessors handle the different representation needs:

**`TextPreprocessorTFIDF`**
- URL / email / hashtag removal
- Contraction expansion (`don't` → `do not`)
- Lowercasing, punctuation stripping
- NLTK word tokenisation + POS-aware lemmatisation (WordNet)
- Stopword removal, minimum token length filtering
- Output: space-joined token string for `TfidfVectorizer`

**`TextPreprocessorWord2Vec`**
- Same cleaning as above, but **sentence-aware** — uses `sent_tokenize` to preserve sentence boundaries
- Output: list of token lists (one per sentence) for Word2Vec training

**`TextPreprocessorTransformer`**
- Light-touch cleaning only (URLs, emails, extra whitespace)
- No lowercasing, no stopword removal — preserves casing and semantics for the pretrained model
- Output: cleaned string passed directly to `SentenceTransformer.encode()`

---

## Models

### Model 1 — TF-IDF + LinearSVC

```
TfidfVectorizer(max_features=25000, ngram_range=(1,2), sublinear_tf=True)
    ↓
LinearSVC(C=1.0, max_iter=2000)
```

- Unigrams + bigrams capture both individual terms and short phrase patterns
- `sublinear_tf=True` applies log-normalisation to term frequencies, reducing the dominance of very frequent tokens
- Compared against **LogisticRegression** (93.19%) and **MultinomialNB** (92.99%) baselines

### Model 2 — Word2Vec + TF-IDF Weighted Averaging + LinearSVC

```
Word2Vec(vector_size=100, window=5, min_count=2)   ← trained on train split only
    ↓
TF-IDF weighted mean pooling → 100-dim document vector
    ↓
LinearSVC(C=1.0, max_iter=3000)
```

- Word2Vec trained **only on the training split** to prevent data leakage
- Plain mean pooling replaced by **TF-IDF weighted averaging**: each word vector is scaled by its IDF score, giving rarer/more informative words more influence in the final document representation
- 2-D PCA visualisation shows the embedding space

### Model 3 — Sentence Transformer (all-MiniLM-L6-v2) + LinearSVC

```
SentenceTransformer('all-MiniLM-L6-v2')  ← pretrained, frozen
    ↓
384-dim sentence embedding
    ↓
LinearSVC(C=1.0, max_iter=3000)
```

- `all-MiniLM-L6-v2` is a distilled BERT model optimised for semantic similarity tasks — fast and compact (22M parameters, 384-dim output)
- Embeddings are computed once and reused for cross-validation to avoid repeated encoding cost
- 2-D PCA visualisation of the embedding space

---

## Results

### Validation Set (80/20 stratified split, same indices across all models)

| Model | Accuracy | F1 (weighted) | Cohen's κ | ROC-AUC | CV F1 | CV Std |
|---|---|---|---|---|---|---|
| TF-IDF + LinearSVC | 93.78% | 93.78% | 0.8755 | 0.9849 | 93.81% | ±0.38% |
| Word2Vec + LinearSVC | 87.60% | 87.60% | 0.7517 | 0.9415 | — | — |
| all-MiniLM-L6-v2 + LinearSVC | **94.06%** | **94.06%** | **0.8811** | **0.9850** | **93.98%** | **±0.13%** |

### Confusion Matrices (validation set)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| TF-IDF + LinearSVC | 3281 | 234 | 191 | 3125 |
| Word2Vec + LinearSVC | 3127 | 388 | 459 | 2857 |
| all-MiniLM-L6-v2 + LinearSVC | 3266 | 249 | 157 | 3159 |

### Error Analysis

All models struggle with the same types of articles: politically ambiguous headlines, satire, and short titles where context is insufficient to distinguish real from fake.

| Model | Error Rate | Avg words (correct) | Avg words (misclassified) |
|---|---|---|---|
| TF-IDF + LinearSVC | 6.2% | 8.8 | 7.9 |
| Word2Vec + LinearSVC | 12.4% | 8.8 | 8.3 |
| all-MiniLM-L6-v2 + LinearSVC | 5.9% | 11.7 | 11.8 |

Shorter articles tend to be harder — less signal for all models. The Transformer model is the exception: misclassified articles are *not* noticeably shorter, suggesting its errors stem from semantic ambiguity rather than text length.

---

## Key Visualisations

| Plot | Description |
|---|---|
| Class distribution bar | Near-balanced dataset (17,572 vs 16,580) |
| Top-20 TF-IDF tokens | Political domain signal: *trump*, *obama*, *clinton*, *russia* |
| Confusion matrices | Per-model, colour-coded (blue / green / purple) |
| ROC & PR curves | One-vs-Rest for both classes; TF-IDF & Transformer at AUC 0.98 |
| PCA 2-D (Word2Vec) | Embeddings capture some class structure (PC1 39.2%, PC2 29.9%) |
| PCA 2-D (Transformer) | Much lower explained variance (4.1% + 3.5%) — information distributed across all 384 dims |
| Model comparison bar chart | Side-by-side Accuracy / F1 / CV F1 for all three models |

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn nltk gensim sentence-transformers
```

### 2. Download NLTK resources

The script auto-downloads on first run:

```python
# punkt_tab, stopwords, wordnet, averaged_perceptron_tagger_eng
```

Or manually:

```python
import nltk
nltk.download(['punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng'])
```

### 3. Prepare data

Place your CSV files (no header, label as first character of each row) in a `dataset/` folder:

```
dataset/
├── training_data.csv
└── testing_data.csv
```

### 4. Run

```bash
python nlp_classifier.py
```

Output CSVs and a model comparison table are saved to `dataset/`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` | Plotting |
| `scikit-learn` | TF-IDF, LinearSVC, evaluation metrics, cross-validation |
| `nltk` | Tokenisation, lemmatisation, POS tagging, stopwords |
| `gensim` | Word2Vec training |
| `sentence-transformers` | Pretrained `all-MiniLM-L6-v2` encoder |

---

## Design Decisions

**Why LinearSVC for all three models?** Using the same classifier isolates the effect of the text representation — any performance difference is attributable to the embedding method, not the downstream model.

**Why TF-IDF weighted averaging for Word2Vec?** Plain mean pooling treats all words equally; *"the"* gets the same weight as *"impeachment"*. Weighting by IDF pushes rarer, more informative tokens to dominate the document vector.

**Why `all-MiniLM-L6-v2`?** It offers a strong accuracy/speed trade-off for a bootcamp-scale experiment — 22M parameters, fast inference (~90 batches/s), and pre-trained on a broad corpus that already captures political language semantics.

**Why the same train/val split indices for all models?** Ensures a fair head-to-head comparison on identical validation examples.

**Why is the Transformer PCA showing only ~7.6% explained variance across 2 components?** The 384-dimensional embeddings distribute information much more evenly across dimensions than the 100-dim Word2Vec vectors. Low PCA variance does not indicate poor embeddings — it means the class structure lives in higher-dimensional subspaces that PCA cannot capture in 2-D.
