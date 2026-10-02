"""Machine Learning model pipeline, training, evaluation, and inference for SpamGuard AI.

Implements TF-IDF (word + character n-grams) with a balanced Logistic Regression
classifier, blended with rule-based explainability signals.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.preprocessing import clean_text, extract_signals, validate_input

DEFAULT_DATA_PATH = Path("data/cleaned_dataset.csv")
DEFAULT_MODEL_PATH = Path("models/model.pkl")
DEFAULT_METRICS_PATH = Path("models/metrics.json")

# Blend weights: 80% ML model probability + 20% deterministic rule signals
ML_WEIGHT = 0.80
RULE_WEIGHT = 0.20


def build_pipeline(c_param: float = 3.0, random_state: int = 42) -> Pipeline:
    """Build a scikit-learn text classification pipeline.

    Combines word n-grams (1-2) for vocabulary semantics and character n-grams (3-5)
    for catching character substitutions, typos, and leetspeak obfuscations.
    """
    feature_union = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                lowercase=True,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=3,
                sublinear_tf=True,
                lowercase=True,
            ),
        ),
    ])

    classifier = LogisticRegression(
        C=c_param,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="liblinear",
    )

    return Pipeline([
        ("union", feature_union),
        ("clf", classifier),
    ])


def train_and_evaluate(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_save_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_save_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Train the model pipeline on the dataset, evaluate holdout metrics, and save artifacts."""
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}. Please check the path.")

    start_time = time.time()
    df = pd.read_csv(data_file)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    # Preprocess text
    X_cleaned = df["text"].apply(clean_text)
    y = (df["label"].str.lower().str.strip() == "spam").astype(int)

    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(c_param=3.0, random_state=42)
    pipeline.fit(X_train, y_train)

    fit_duration = time.time() - start_time

    # Predictions on test holdout
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model_type": "TF-IDF (Word 1-2 + Char 3-5) + Balanced Logistic Regression",
        "dataset_total": int(len(df)),
        "dataset_ham": int((y == 0).sum()),
        "dataset_spam": int((y == 1).sum()),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        },
        "confusion_matrix": cm,
        "training_time_seconds": round(fit_duration, 2),
    }

    # Save model pipeline
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)

    # Save metrics JSON
    metrics_path = Path(metrics_save_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load the trained model pipeline. If absent, train and save it first."""
    target_path = Path(model_path)
    if not target_path.exists():
        train_and_evaluate(model_save_path=target_path)
    return joblib.load(target_path)


def predict_message(
    text: str,
    model: Pipeline | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Analyze a single message end-to-end with validation, ML inference, and explanation.

    Returns:
        Structured prediction dict with verdict, risk score, confidence, and reasons.
    """
    start_time = time.time()

    # Step 1: Input Validation
    is_valid, error_msg = validate_input(text)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
            "raw_text": text,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
        }

    # Step 2: Preprocessing
    cleaned = clean_text(text)

    # Step 3: Load Model & Predict Probability
    if model is None:
        model = load_model(model_path)

    ml_prob = float(model.predict_proba([cleaned])[0, 1])

    # Step 4: Rule-based Signal Extraction
    signals = extract_signals(text)
    signal_score = signals["signal_score"]

    # Step 5: Blended Risk Calculation
    blended_risk = (ML_WEIGHT * ml_prob) + (RULE_WEIGHT * signal_score)
    blended_risk = round(min(max(blended_risk, 0.0), 1.0), 4)

    # Step 6: Verdict Categorisation
    if blended_risk >= 0.50 or ml_prob >= 0.65:
        verdict = "SPAM"
        verdict_badge = "🚨 SPAM / PHISHING"
        verdict_color = "#e74c3c"
    elif blended_risk >= 0.35:
        verdict = "SUSPICIOUS"
        verdict_badge = "⚠️ SUSPICIOUS (REVIEW)"
        verdict_color = "#f39c12"
    else:
        verdict = "HAM"
        verdict_badge = "✅ LEGITIMATE (HAM)"
        verdict_color = "#27ae60"

    # Step 7: Confidence Level
    margin = abs(ml_prob - 0.50)
    if margin >= 0.35:
        confidence = "High"
    elif margin >= 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Step 8: Actionable Guidance
    if verdict == "SPAM":
        advice = (
            "Do not click any embedded links, do not reply with passwords, OTPs, or "
            "financial details, and report or block this sender immediately."
        )
    elif verdict == "SUSPICIOUS":
        advice = (
            "Exercise caution. This message has indicators of an unsolicited campaign. "
            "Verify the sender's identity through an independent, official channel before acting."
        )
    else:
        advice = (
            "The message appears standard and exhibits typical conversational language "
            "with no detected phishing patterns."
        )

    # Explanations list
    explanations: list[str] = list(signals["signals_found"])
    if ml_prob >= 0.70:
        explanations.insert(0, f"Statistical language model detected heavy spam vocabulary patterns (P(Spam) = {ml_prob * 100:.1f}%)")
    elif ml_prob <= 0.20:
        explanations.insert(0, f"Statistical language model detected normal legitimate communication patterns (P(Spam) = {ml_prob * 100:.1f}%)")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "success": True,
        "raw_text": text,
        "cleaned_text": cleaned,
        "verdict": verdict,
        "verdict_badge": verdict_badge,
        "verdict_color": verdict_color,
        "risk_score": blended_risk,
        "risk_percentage": round(blended_risk * 100, 1),
        "spam_probability": round(ml_prob, 4),
        "signal_score": signal_score,
        "confidence": confidence,
        "signals_found": signals["signals_found"],
        "explanations": explanations,
        "advice": advice,
        "latency_ms": latency_ms,
    }
