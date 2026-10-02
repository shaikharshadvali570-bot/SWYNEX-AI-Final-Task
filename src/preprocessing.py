"""Text preprocessing, normalization, and explainable signal extraction for SpamGuard AI.

Provides deterministic cleaning and feature inspection ensuring identical
transformations across training, validation, and production inference.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse, urlsplit
from typing import Any

# Regular expression patterns for tokenization and normalization
URL_PATTERN = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\"']+|(?:\b[\w-]+\.(?:com|net|org|biz|info|co|xyz|"
    r"ly|tk|ml|ga|cf|gq|sh|top|click|link|life|live|online|shop|store|uk|us|in|de|ru|"
    r"cn|vip|win|loan|work|account|support|secure|update|verify|pdf)\b(?:/\S*)?)",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CURRENCY_PATTERN = re.compile(r"(?:[\$£€¥₹]|(?:(?:usd|gbp|eur|inr|rs\.?)\s*))\d+(?:[.,]\d+)?", re.IGNORECASE)
REPEATED_PUNCT = re.compile(r"([!?.:,;])\1{2,}")
WHITESPACE = re.compile(r"\s+")
IP_URL_PATTERN = re.compile(r"(?:https?://)?\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?:/\S*)?", re.IGNORECASE)

# High-risk URL shorteners frequently abused by phishers to mask malicious targets
URL_SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorturl.at", "cutt.ly", "rb.gy", "rebrand.ly", "bit.do",
    "t.ly", "s.id", "shorte.st", "da.gd", "v.gd", "tiny.cc",
})

# Suspicious top-level domains commonly associated with spam campaigns
RISKY_TLDS = frozenset({
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click", ".link",
    ".zip", ".work", ".loan", ".vip", ".cam", ".rest"
})

# Keyword signal dictionaries for rule-based explainability
URGENCY_KEYWORDS = [
    "urgent", "immediately", "immediate", "action required", "account suspended",
    "suspended", "final notice", "within 24 hours", "24 hours", "expires today",
    "last chance", "act now", "critical security", "restricted"
]

FINANCIAL_LURE_KEYWORDS = [
    "won", "winner", "prize", "cash", "lottery", "claim", "jackpot",
    "free entry", "reward", "million", "voucher", "congratulations",
    "guaranteed", "selected to receive", "inheritance", "bonus"
]

CREDENTIAL_HARVEST_KEYWORDS = [
    "verify your account", "verify identity", "confirm password", "login now",
    "log in immediately", "update payment", "security alert", "suspicious activity",
    "unauthorized login", "otp", "one-time password", "enter pin", "bank account",
    "credit card", "billing update"
]

CALL_OR_SMS_CTA = [
    "call now", "call free", "reply stop", "text stop", "txt", "txt to",
    "call customer service", "claim by calling", "free ringtone"
]


def clean_text(text: str) -> str:
    """Normalize raw message text into a standardized representation for ML vectorization.

    Normalizations applied:
      1. Unicode NFKC normalization
      2. Mask URLs with <URL> token
      3. Mask Emails with <EMAIL> token
      4. Mask Phone numbers with <PHONE> token
      5. Mask Currency with <CURRENCY> token
      6. Collapse repeated punctuation (e.g. '!!!!' -> '!')
      7. Collapse multiple whitespace characters
    """
    if not isinstance(text, str) or not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = EMAIL_PATTERN.sub(" <EMAIL> ", cleaned)
    cleaned = URL_PATTERN.sub(" <URL> ", cleaned)
    cleaned = PHONE_PATTERN.sub(" <PHONE> ", cleaned)
    cleaned = CURRENCY_PATTERN.sub(" <CURRENCY> ", cleaned)
    cleaned = REPEATED_PUNCT.sub(r" \1 ", cleaned)
    cleaned = WHITESPACE.sub(" ", cleaned).strip()
    return cleaned


def validate_input(text: str | Any) -> tuple[bool, str]:
    """Validate user input from the UI or API before processing.

    Ensures the application does not crash and provides clear, beginner-friendly
    feedback on input errors.

    Returns:
        tuple (is_valid: bool, error_message: str)
    """
    if text is None:
        return False, "Input cannot be empty. Please enter a message to analyze."

    if not isinstance(text, str):
        return False, "Invalid input type. Expected text."

    stripped = text.strip()
    if len(stripped) == 0:
        return False, "The message is empty. Please type or paste a message into the box."

    if len(stripped) < 3:
        return False, "Message is too short to evaluate. Please enter at least 3 characters."

    if len(stripped) > 5000:
        return False, f"Message is too long ({len(stripped)} characters). Please limit messages to 5,000 characters."

    # Check if input contains at least one alphanumeric character
    if not any(c.isalnum() for c in stripped):
        return False, "The input contains only symbols or punctuation. Please enter meaningful text."

    return True, ""


def extract_urls(text: str) -> list[str]:
    """Extract all distinct URL tokens from text."""
    found: list[str] = []
    for match in URL_PATTERN.finditer(text or ""):
        token = match.group(0).strip().rstrip(").,;!?'\"]")
        if token and token.lower() not in found:
            found.append(token.lower())
    return found


def url_host(url: str) -> str:
    """Safely extract the domain or IP host from a URL."""
    candidate = url if "://" in url else f"http://{url}"
    try:
        return (urlparse(candidate).netloc or urlsplit(candidate).netloc).lower()
    except Exception:
        return ""


def extract_signals(text: str) -> dict[str, Any]:
    """Extract human-readable heuristic risk signals from the message.

    Used by the explanation layer to provide clear reasoning to non-technical users.

    Returns:
        Dictionary with:
          - 'signal_score': float in range [0.0, 1.0]
          - 'signals_found': list of human-readable signal explanations
          - 'details': dict containing detailed counts and matched items
    """
    signals: list[str] = []
    score = 0.0
    text_lower = text.lower()

    # 1. URL Analysis
    urls = extract_urls(text)
    has_ip_url = bool(IP_URL_PATTERN.search(text))
    has_shortener = False
    has_risky_tld = False

    for url in urls:
        host = url_host(url)
        if any(host == s or host.endswith(f".{s}") for s in URL_SHORTENERS):
            has_shortener = True
        if any(host.endswith(tld) for tld in RISKY_TLDS):
            has_risky_tld = True

    if has_ip_url:
        signals.append("Link points directly to a raw numerical IP address (classic phishing technique)")
        score += 0.40

    if has_shortener:
        signals.append("Uses a URL shortener service (often used to obscure malicious links)")
        score += 0.25

    if has_risky_tld:
        signals.append("Link points to an abused or high-risk top-level domain")
        score += 0.20

    if urls and not (has_ip_url or has_shortener or has_risky_tld):
        signals.append(f"Contains {len(urls)} external web link(s)")
        score += 0.15

    # 2. Urgency and Coercion
    matched_urgency = [kw for kw in URGENCY_KEYWORDS if kw in text_lower]
    if matched_urgency:
        signals.append(f"High-urgency language detected: '{matched_urgency[0]}'")
        score += 0.25

    # 3. Financial and Prize Lures
    matched_prizes = [kw for kw in FINANCIAL_LURE_KEYWORDS if kw in text_lower]
    if matched_prizes:
        signals.append(f"Prize or financial reward lure detected: '{matched_prizes[0]}'")
        score += 0.30

    # 4. Credential and Phishing Keywords
    matched_creds = [kw for kw in CREDENTIAL_HARVEST_KEYWORDS if kw in text_lower]
    if matched_creds:
        signals.append(f"Account credential or identity verification request: '{matched_creds[0]}'")
        score += 0.35

    # 5. Call or Premium SMS CTA
    matched_cta = [kw for kw in CALL_OR_SMS_CTA if kw in text_lower]
    if matched_cta:
        signals.append(f"Direct call or SMS response call-to-action: '{matched_cta[0]}'")
        score += 0.20

    # 6. Phone Numbers and Currency Mentions
    phones = PHONE_PATTERN.findall(text)
    if len(phones) > 0 and len(text) < 200:
        signals.append(f"Contains direct callback or phone number ({phones[0].strip()})")
        score += 0.15

    currencies = CURRENCY_PATTERN.findall(text)
    if currencies:
        signals.append(f"Contains explicit monetary claim ({currencies[0].strip()})")
        score += 0.15

    # 7. Uppercase character ratio
    letters = [c for c in text if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.40 and len(letters) >= 10:
            signals.append(f"Excessive capital letters ({int(upper_ratio * 100)}% uppercase shouting)")
            score += 0.15

    # Clamp score between 0.0 and 1.0
    normalized_score = min(max(score, 0.0), 1.0)

    return {
        "signal_score": round(normalized_score, 4),
        "signals_found": signals,
        "details": {
            "urls": urls,
            "has_ip_url": has_ip_url,
            "has_shortener": has_shortener,
            "matched_urgency": matched_urgency,
            "matched_prizes": matched_prizes,
            "matched_creds": matched_creds,
        }
    }
