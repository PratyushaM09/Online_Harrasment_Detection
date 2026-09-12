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

Milestone 0B: Configuration and reproducibility foundation.

The project currently contains the minimal repository structure, centralized configuration loading, and a standard-library seed utility. Data processing, model training, inference, explainability, and UI functionality are not implemented yet.

## Local Environment Setup

Target environment: Python 3.12 on Windows with VS Code.

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
pytest
```
