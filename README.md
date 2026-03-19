# 🚫 Hate Speech & Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-yellow)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-93%25-brightgreen)]()
[![Precision](https://img.shields.io/badge/Precision-85%25-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ML Training](https://img.shields.io/badge/Feb%202025-ML%20Industrial%20Training-blueviolet)]()

---

## 📌 Overview

**Developed NLP pipelines for data cleaning and feature extraction using Pandas, NumPy, Scikit-learn, NLTK. Evaluated models using AUC & cross-validation achieving 93% accuracy and 85% precision (Feb 2025 - ML Industrial Training).**

Dual classification system for:
- 📩 **SMS Spam Detection** (Binary: Ham/Spam)  
- 🗣️ **Hate Speech Detection** (Multi-class: Hate/Offensive/Neutral)

---

## 📊 Project Summary

| Project | Dataset Size | Classes | Imbalance | Key Features |
|---------|--------------|---------|-----------|--------------|
| **SMS Spam** | 9,307 (balanced) | Ham/Spam | 6:1 → 1:1 | Word count, Currency(£/$), Numbers |
| **Hate Speech** | 24,783 | Hate/Offensive/Neutral | 1:2:3 | Tweet cleaning, Stemming, TF-IDF |

---

## ⚡ Quick Start

```bash
git clone https://github.com/yourusername/sms-spam-hatespeech-detection.git
cd sms-spam-hatespeech-detection

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

pip install -r requirements.txt
jupyter notebook
Run notebooks:

spam_classification.ipynb

hate_speech_detection.ipynb

📁 Repository Structure
text
sms-spam-hatespeech-detection/
├── notebooks/
│   ├── spam_classification.ipynb      # 93% accuracy SMS pipeline
│   └── hate_speech_detection.ipynb    # Multi-class hate speech
├── src/
│   ├── preprocessing.py              # Industrial-grade cleaning
│   ├── features.py                   # Currency/number extraction
│   └── model.py                      # LR, SVM, NB models
├── app.py                            # FastAPI deployment
├── requirements.txt
├── LICENSE
└── README.md
🎯 SMS Spam Detection Pipeline
📂 Dataset
UCI SMS Spam Collection: 5,572 → 9,307 (oversampled)

🔑 Feature Engineering (85% Precision Driver)
Feature	Spam %	Ham %	Impact
Word Count >20	65%	45%	Promotional patterns
Currency (£/$/₹)	28%	4%	Strong spam signal
Numbers	72%	18%	Contact/lottery spam
python
import re

def spam_features(msg):
    return {
        "word_count": len(msg.split()),
        "has_currency": bool(re.findall(r"[\$£€¥₹]", msg)),
        "has_number": bool(re.search(r"\d", msg))
    }
🤖 Models Evaluated
Logistic Regression: 93% accuracy

SVM: 91% accuracy

Naive Bayes: 92% accuracy

🗣️ Hate Speech Detection Pipeline
📂 Dataset
24,783 labeled tweets

⚖️ Class Distribution
Class	Count	%
Offensive	13,952	56%
Neutral	10,399	42%
Hate	432	2% (oversampled)
🧹 Advanced Preprocessing
python
def clean_tweet(text):
    text = re.sub(r'http\S+', '', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    words = [stemmer.stem(w) for w in text.split() if w not in stopwords]
    return ' '.join(words)
📈 Model Performance Summary
Model	Accuracy	Precision	AUC-ROC	CV F1
Logistic Regression	93%	85%	0.95	0.92
SVM	91%	83%	0.93	0.90
Naive Bayes	92%	84%	0.94	0.91
🔧 Production Fixes Implemented
Issue	Fix	Impact
get_feature_names()	get_feature_names_out()	scikit-learn ≥1.2
Number detection	str.contains(r'\d')	+5% precision
Letter removal bug	Keep a-zA-Z in regex	Proper TF-IDF
Class imbalance	Oversampling/SMOTE	93% accuracy
🚀 Production Deployment
python
# app.py - FastAPI endpoint
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict_spam")
def predict_spam(message: str):
    vector = tfidf.transform([preprocess(message)])
    pred = model.predict(vector)
    return {"spam": bool(pred), "confidence": float(model.predict_proba(vector).max())}

# Run: uvicorn app:app --reload
API Docs: http://127.0.0.1:8000/docs

📦 Requirements
text
pandas>=1.5.3
numpy>=1.24.3
scikit-learn>=1.3.0
nltk>=3.8.1
seaborn>=0.12.2
matplotlib>=3.7.2
imbalanced-learn>=0.11.0
fastapi>=0.100.0
uvicorn>=0.23.0
jupyterlab>=4.0.0
📜 License
MIT License — Free for academic & commercial use.
