# ⚖️ Ethics, Transparency, and Responsible AI in SpamGuard AI

## Overview

SpamGuard AI is an automated text classification and risk scoring application designed to protect end users and communication networks from unsolicited spam, SMS phishing ("smishing"), and identity theft scams.

Because spam filters operate at the boundary between public communication channels and private correspondence, deploying AI in this space raises critical ethical, legal, and operational considerations. This document formalizes our responsible AI framework.

---

## 1. Data Privacy and Confidentiality

### The Risk
Personal text messages, emails, and SMS alerts frequently contain sensitive personally identifiable information (PII), such as:
- Banking transaction alerts and account balance notifications
- Two-factor authentication (2FA) codes and one-time passwords (OTPs)
- Private family dialogues, medical reminders, and travel confirmations

### Our Implementation
- **Zero Cloud Data Exfiltration:** All message tokenization, vectorization, and inference routines run strictly on the local runtime (`app.py` / `src/model.py`). No raw text or telemetry is sent to third-party APIs or external LLM providers.
- **Stateless Analysis:** User inputs evaluated in the web interface are processed in memory and immediately discarded. The system does not maintain an input log database.
- **Client-Side Entity Masking:** During preprocessing (`clean_text`), detected telephone numbers, email addresses, and monetary amounts are normalized into generic replacement tokens (`<PHONE>`, `<EMAIL>`, `<CURRENCY>`) before reaching the vectorization layer.

---

## 2. Managing False Positives and False Negatives

In binary classification for threat detection, the costs of the two error types are fundamentally asymmetric:

| Error Type | What Happens | Real-World Consequence | Mitigation Strategy |
|---|---|---|---|
| **False Positive (Type I)** | Legitimate message (Ham) misclassified as Spam | Critical communications missed (e.g. urgent hospital update, flight change alert, password reset token). | High-precision tuning (97.95% holdout precision), three-tier output ("Safe", "Suspicious", "Spam"), and confidence scoring. |
| **False Negative (Type II)** | Malicious phishing SMS classified as Legitimate | User may click a fraudulent link, install malware, or compromise credentials. | Multi-layered defense: TF-IDF statistical model is reinforced with deterministic heuristic rules that flag suspicious IP-based URLs and shorteners even if phrasing is novel. |

### The "Three-Tier" Verdict Architecture
Rather than forcing a rigid binary "Spam / Not Spam" decision, SpamGuard AI uses three distinct risk states:
1. **✅ Safe (HAM) (Risk < 35%):** Clean text with zero threat indicators.
2. **⚠️ Suspicious / Review (35% ≤ Risk < 50%):** Borderline indicators; user is advised to manually verify sender identity before acting.
3. **🚨 Spam / Phishing (Risk ≥ 50%):** Strong statistical and heuristic evidence; user is advised to delete and block.

---

## 3. Human Oversight & Automated Action Guardrails

- **Advisory Role:** SpamGuard AI is explicitly designed as a **decision-support tool**, not an autonomous censoring agent.
- **No Automatic Deletion:** The application advises users on potential risks; it does not automatically delete messages or disconnect user accounts without explicit human authorization.
- **Explainability as a Core Safety Mechanism:** Every verdict is accompanied by a transparent breakdown of detected triggers (such as matched urgency phrases, external shorteners, or numeric IP hosts). Users are never left with an opaque "black-box" decision.

---

## 4. Bias and Dataset Limitations

### Dataset Demographics
The core ML pipeline is trained on the curated UCI SMS Spam Collection dataset (5,574 messages).
- **Language Bias:** The training data is overwhelmingly in modern English. The model cannot reliably evaluate messages written in other languages, regional dialects (e.g., Hinglish, Singlish), or non-Latin scripts.
- **Era & Channel Bias:** The dataset includes early-to-mid 2000s SMS campaign patterns (e.g., ringtone offers, premium rate numbers). While modern smishing (such as crypto lures and package delivery scams) is addressed via our dynamic heuristic rules, continuous dataset retraining is necessary to track evolving cybercrime tactics.

---

## 5. Guidelines for Responsible Deployment

### Permissible Use Cases
- Assisting individuals in screening suspicious SMS, email, or chat messages.
- Triage assistance for enterprise help desks reviewing reported phishing attempts.
- Educational demonstrations of NLP classification, feature extraction, and explainable AI.

### Prohibited / Discouraged Use Cases
- Autonomous, unmonitored mass-filtering where false positives could lead to safety, legal, or financial harm.
- Deploying as a sole security control without email authentication (SPF, DKIM, DMARC) and endpoint protection.
