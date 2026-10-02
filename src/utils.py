"""Utility helpers, sample datasets, and batch scoring utilities for SpamGuard AI.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from src.model import predict_message


def get_sample_messages() -> list[dict[str, str]]:
    """Return a curated set of realistic test messages representing diverse scenarios."""
    return [
        {
            "category": "🚨 Phishing Scam (Bank / Security)",
            "message": "URGENT: Suspicious activity detected on your bank account. Your debit card is temporarily suspended. Verify your identity immediately at http://192.168.1.10/login to restore access.",
            "expected": "SPAM / High Risk",
        },
        {
            "category": "🚨 Lottery & Prize Scam",
            "message": "WINNER!! You have been selected to receive a £1,000 cash prize reward or luxury holiday! Call 09061701461 to claim immediately. Valid 12 hours only. Reply STOP to opt out.",
            "expected": "SPAM / High Risk",
        },
        {
            "category": "⚠️ Delivery Notification Phishing",
            "message": "Your parcel could not be delivered due to missing house number. Update your delivery preferences within 24 hours at https://bit.ly/parcel-redelivery-notice to avoid return to sender.",
            "expected": "SUSPICIOUS / High Risk",
        },
        {
            "category": "✅ Legitimate Personal Chat",
            "message": "Hey, are you free for lunch today around 1:30 PM? Let me know, we can head over to the cafe near the park.",
            "expected": "HAM / Safe",
        },
        {
            "category": "✅ Legitimate Work / Academic Communication",
            "message": "Hi team, please find the meeting notes attached. Let us review section 3 before the sprint demo tomorrow morning. Thanks!",
            "expected": "HAM / Safe",
        },
        {
            "category": "✅ Legitimate Service / 2FA Notification",
            "message": "Your security verification code is 482910. It will expire in 10 minutes. Do not share this code with anyone for your safety.",
            "expected": "HAM / Safe",
        },
    ]


def process_batch_dataframe(
    df: pd.DataFrame,
    text_column: str,
    model=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process an uploaded pandas DataFrame in batch and attach prediction verdicts.

    Returns:
        tuple (processed_df, summary_metrics)
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the uploaded file.")

    results: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        raw_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        res = predict_message(raw_text, model=model)
        if res.get("success"):
            results.append({
                "Verdict": res["verdict"],
                "Risk Score": res["risk_score"],
                "Risk %": res["risk_percentage"],
                "P(Spam)": res["spam_probability"],
                "Confidence": res["confidence"],
                "Signals Count": len(res["signals_found"]),
                "Signals Details": " | ".join(res["signals_found"]) if res["signals_found"] else "None",
            })
        else:
            results.append({
                "Verdict": "INVALID",
                "Risk Score": 0.0,
                "Risk %": 0.0,
                "P(Spam)": 0.0,
                "Confidence": "N/A",
                "Signals Count": 0,
                "Signals Details": res.get("error", "Invalid input"),
            })

    results_df = pd.DataFrame(results)
    combined_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    total_count = len(combined_df)
    spam_count = int((combined_df["Verdict"] == "SPAM").sum())
    suspicious_count = int((combined_df["Verdict"] == "SUSPICIOUS").sum())
    ham_count = int((combined_df["Verdict"] == "HAM").sum())
    invalid_count = int((combined_df["Verdict"] == "INVALID").sum())

    summary = {
        "total": total_count,
        "spam": spam_count,
        "suspicious": suspicious_count,
        "ham": ham_count,
        "invalid": invalid_count,
        "spam_rate": round((spam_count / total_count) * 100, 1) if total_count > 0 else 0.0,
    }

    return combined_df, summary


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 encoded CSV bytes for Streamlit download button."""
    return df.to_csv(index=False).encode("utf-8")
