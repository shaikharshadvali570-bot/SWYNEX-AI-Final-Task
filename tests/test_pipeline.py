"""Automated test suite for SpamGuard AI pipeline and input validation."""

import pytest
import pandas as pd
from src.preprocessing import clean_text, validate_input, extract_signals
from src.model import load_model, predict_message
from src.utils import process_batch_dataframe, get_sample_messages


@pytest.fixture(scope="session")
def model():
    """Load model once for test session."""
    return load_model()


def test_validation_empty():
    valid, err = validate_input("")
    assert not valid
    assert "empty" in err.lower()


def test_validation_whitespace():
    valid, err = validate_input("    \t\n  ")
    assert not valid
    assert "empty" in err.lower()


def test_validation_symbols_only():
    valid, err = validate_input("???!!!!....---")
    assert not valid
    assert "symbols" in err.lower() or "punctuation" in err.lower()


def test_validation_valid_input():
    valid, err = validate_input("Hey, are you free for coffee today?")
    assert valid
    assert err == ""


def test_clean_text_normalizes_entities():
    sample = "Call +1234567890 or visit https://example.com/login to claim $500!"
    cleaned = clean_text(sample)
    assert "<PHONE>" in cleaned
    assert "<URL>" in cleaned
    assert "<CURRENCY>" in cleaned


def test_extract_signals_detects_ip_and_urgency():
    sample = "URGENT: Verify your account at http://192.168.1.1/login immediately."
    signals = extract_signals(sample)
    assert signals["signal_score"] > 0.40
    assert any("raw numerical IP" in s for s in signals["signals_found"])
    assert any("urgency" in s.lower() for s in signals["signals_found"])


def test_prediction_phishing(model):
    sample = "URGENT: Suspicious activity on bank account. Verify identity at http://192.168.1.10/login immediately."
    res = predict_message(sample, model=model)
    assert res["success"] is True
    assert res["verdict"] == "SPAM"
    assert res["risk_score"] > 0.60
    assert res["spam_probability"] > 0.50
    assert len(res["signals_found"]) > 0


def test_prediction_ham(model):
    sample = "Hey team, the project sync is scheduled for 10:30 AM tomorrow. Please review the slides."
    res = predict_message(sample, model=model)
    assert res["success"] is True
    assert res["verdict"] == "HAM"
    assert res["risk_score"] < 0.35
    assert res["confidence"] in ["High", "Medium"]


def test_sample_messages_structure():
    samples = get_sample_messages()
    assert len(samples) >= 5
    for s in samples:
        assert "category" in s
        assert "message" in s
        assert "expected" in s


def test_batch_processing(model):
    df = pd.DataFrame({
        "msg_id": [1, 2],
        "content": [
            "WINNER!! You won 1,000 cash! Call 09061701461 to claim.",
            "Can we have lunch together at 1pm?"
        ]
    })
    result_df, summary = process_batch_dataframe(df, "content", model=model)
    assert "Verdict" in result_df.columns
    assert "Risk Score" in result_df.columns
    assert summary["total"] == 2
    assert summary["spam"] >= 1
    assert summary["ham"] >= 1
