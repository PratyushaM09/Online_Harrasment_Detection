# Deep Learning Model for Online Harassment Detection

## Problem Statement

Online platforms receive large volumes of user-generated text, making it difficult to identify harmful or abusive content consistently and at scale. This project aims to build a deep learning system for detecting online harassment in text.

## Objectives

- Build a multi-label text classification pipeline for online harassment detection.
- Detect multiple toxicity categories from a single input text.
- Use explainability techniques to support interpretation of model predictions.
- Provide a simple user-facing interface after the core model pipeline is implemented.

## Planned Toxicity Labels

- Toxic
- Severe toxic
- Obscene
- Threat
- Insult
- Identity hate

This project is planned as a multi-label classification task, where one text sample may belong to more than one toxicity label.

## Planned Stack

- Python
- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa
- Streamlit
- LIME

## Current Status

Milestone 2B: Slang and obfuscation detection foundation.

The project currently contains the minimal repository structure, centralized configuration loading, a standard-library seed utility, dataset validation and analysis helpers, deterministic multi-label splitting, conservative text normalization, and metadata-only slang/obfuscation detection. Model training, inference, explainability, and UI functionality are not implemented yet.

## Dataset Setup

This project is planned to use the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle:

```text
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
```

The project does not redistribute the dataset. Download the dataset manually from Kaggle and place the training file here:

```text
data/raw/train.csv
```

The expected file is `train.csv`. The `data/raw/` directory is ignored by Git so the real dataset is not committed.

## Dataset Analysis

The dataset analysis layer reports early evidence about the training data before preprocessing or modeling. It examines class imbalance, positive-label frequency, multi-label co-occurrence, empty or missing text, and the number of toxicity labels assigned per comment.

These statistics are intended to inform later decisions such as threshold tuning, class weighting, sampling, or loss-function changes. No balancing strategy is implemented yet.

## Dataset Splitting

The configured dataset split is:

- 80% training
- 10% validation
- 10% test

The splitter uses iterative multi-label stratification because the Jigsaw dataset can assign multiple toxicity labels to the same comment and contains severe class imbalance, including rare categories such as `threat`.

The splitter attempts to preserve each toxicity label's frequency across training, validation, and test partitions. It does not claim perfect equality between partitions, and it does not balance, preprocess, or modify comment text.

## Text Preprocessing

Preprocessing is intentionally conservative. The original text is preserved separately from normalized text, and each change records a deterministic normalization event.

Current normalization preserves Unicode, punctuation, casing, emoji, slang, obfuscated words, hashtags, repeated letters, profanity, and non-Latin scripts. URLs are replaced with `<URL>`, and simple social-media mentions are replaced with `<USER>`.

Slang normalization is intentionally deferred to a later milestone.

## Slang and Obfuscation Detection

The project includes a small detection layer for curated slang and explicit obfuscated terms. Detected expressions are returned as metadata with the original surface form, canonical meaning, kind, and character offsets.

Detection does not automatically rewrite text and does not make toxicity decisions. Keeping detection separate from classification helps reduce false positives later, because terms such as `unalive` may appear in benign or quoted contexts.

## Local Environment Setup

Target environment: Python 3.12 on Windows with VS Code.

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
python -m pytest
python scripts/check_dataset.py
python scripts/analyze_dataset.py
python scripts/split_dataset.py
python scripts/split_dataset.py --save
python scripts/check_preprocessing.py
python scripts/check_slang_detection.py
```
