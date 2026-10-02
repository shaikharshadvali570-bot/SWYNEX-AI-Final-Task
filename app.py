"""SpamGuard AI — Explainable SMS & Message Spam/Phishing Detection System.

Built with Streamlit, Scikit-learn, and Pandas for the Swynex Technologies AI Final Task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, load_model, predict_message
from src.utils import convert_df_to_csv, get_sample_messages, process_batch_dataframe

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="SpamGuard AI — Phishing & Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, professional UI styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .verdict-box {
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    .signal-item {
        background-color: #fff;
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        font-size: 0.95rem;
    }
    .safe-item {
        background-color: #fff;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading machine learning pipeline...")
def get_cached_model():
    """Load model once and cache across user interactions."""
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_data
def get_cached_metrics():
    """Load model training metrics from JSON."""
    if Path(DEFAULT_METRICS_PATH).exists():
        with open(DEFAULT_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    # Load Model Pipeline
    model = get_cached_model()
    metrics_data = get_cached_metrics()

    # Sidebar Information
    with st.sidebar:
        st.markdown("### 🛡️ **SpamGuard AI**")
        st.markdown(
            "An explainable machine learning application designed to detect "
            "**SMS spam**, **financial scams**, and **credential phishing** in text messages."
        )
        st.divider()

        st.markdown("#### ⚙️ **System Specifications**")
        st.markdown("- **Architecture:** TF-IDF + Logistic Regression")
        st.markdown("- **Vocabulary:** Word n-grams (1-2) + Char (3-5)")
        st.markdown("- **Dataset:** 5,574 UCI SMS messages")
        st.markdown("- **Rule Blend:** 80% ML + 20% Heuristic")
        st.markdown("- **Latency:** < 20 ms / message")
        st.divider()

        st.markdown("#### 👤 **Author & Submission**")
        st.markdown("- **Author:** Harshadvali Shaik")
        st.markdown("- **Role:** AI Intern")
        st.markdown("- **Project:** Swynex Technologies AI Final Task")
        st.markdown("- **Version:** 1.0.0 (Production Ready)")

    # Main Header
    st.markdown('<div class="main-title">🛡️ SpamGuard AI — Intelligent Message Security</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-time spam and phishing detection combining statistical NLP '
        'with transparent, human-readable explainability.</div>',
        unsafe_allow_html=True,
    )

    # Navigation Tabs
    tab_single, tab_batch, tab_metrics, tab_ethics = st.tabs([
        "🔍 Single Message Analysis",
        "📂 Batch File Scoring",
        "📊 Model Evaluation & Metrics",
        "⚖️ Ethics & Limitations",
    ])

    # TAB 1: Single Message Analysis
    with tab_single:
        st.markdown("### Analyze an Incoming Message")
        st.write("Test any SMS, WhatsApp notification, or email text to verify its safety and legitimacy.")

        samples = get_sample_messages()
        sample_options = ["-- Select a Pre-configured Example --"] + [f"{s['category']}" for s in samples]

        col_sample, col_clear = st.columns([3, 1])
        with col_sample:
            selected_sample_idx = st.selectbox(
                "Quick Example Scenarios (Try these pre-configured test cases):",
                range(len(sample_options)),
                format_func=lambda i: sample_options[i],
                key="sample_selector",
            )
        with col_clear:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🧹 Clear Input", use_container_width=True):
                st.session_state["user_message_input"] = ""
                st.rerun()

        # Handle preset selection
        initial_text = ""
        if selected_sample_idx > 0:
            initial_text = samples[selected_sample_idx - 1]["message"]

        user_input = st.text_area(
            "Message Content:",
            value=initial_text,
            height=130,
            placeholder="Type or paste the message here (e.g., 'URGENT: Suspicious activity on your bank card...')",
            key="user_message_input",
            help="Enter text between 3 and 5,000 characters. Input containing only symbols will be rejected.",
        )

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
        with col_info:
            char_count = len(user_input.strip())
            st.caption(f"Character count: {char_count} / 5,000 | Words: {len(user_input.split()) if user_input else 0}")

        # When analysis is requested
        if analyze_clicked or (selected_sample_idx > 0 and user_input):
            result = predict_message(user_input, model=model)

            if not result["success"]:
                st.error(f"⚠️ **Input Error:** {result['error']}")
            else:
                st.divider()

                # Verdict Banner
                verdict = result["verdict"]
                bg_color = "#fee2e2" if verdict == "SPAM" else ("#fef3c7" if verdict == "SUSPICIOUS" else "#dcfce7")
                border_color = "#ef4444" if verdict == "SPAM" else ("#f59e0b" if verdict == "SUSPICIOUS" else "#10b981")
                text_color = "#991b1b" if verdict == "SPAM" else ("#92400e" if verdict == "SUSPICIOUS" else "#166534")

                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 8px; padding: 18px; text-align: center; margin-bottom: 20px;">
                        <h2 style="color: {text_color}; margin: 0; font-size: 1.8rem;">{result['verdict_badge']}</h2>
                        <p style="color: {text_color}; margin: 5px 0 0 0; font-size: 1.05rem;">
                            Blended Threat Risk Score: <strong>{result['risk_percentage']}%</strong> | Confidence: <strong>{result['confidence']}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Threat Gauge / Progress Bar
                st.markdown("**Risk Score Meter:**")
                st.progress(result["risk_score"])

                # Metrics Summary Cards
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Classification", result["verdict"])
                with c2:
                    st.metric("Threat Risk Score", f"{result['risk_percentage']}%")
                with c3:
                    st.metric("ML Spam Probability", f"{round(result['spam_probability'] * 100, 1)}%")
                with c4:
                    st.metric("Analysis Latency", f"{result['latency_ms']} ms")

                st.markdown("---")

                # Explainability & Reasoning Section
                col_exp, col_adv = st.columns([1.2, 1])

                with col_exp:
                    st.markdown("#### 🔬 **Why Did the AI Reach This Conclusion?**")
                    st.markdown("Detailed breakdown of signals identified in the message:")

                    if result["explanations"]:
                        for exp in result["explanations"]:
                            if verdict in ["SPAM", "SUSPICIOUS"]:
                                st.markdown(f"<div class='signal-item'>🚩 {exp}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='safe-item'>✓ {exp}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='safe-item'>✓ No suspicious patterns, phishing keywords, or malicious links detected.</div>", unsafe_allow_html=True)

                with col_adv:
                    st.markdown("#### 💡 **Recommended Action**")
                    if verdict == "SPAM":
                        st.error(f"**Security Advisory:**\n\n{result['advice']}")
                    elif verdict == "SUSPICIOUS":
                        st.warning(f"**Caution Advisory:**\n\n{result['advice']}")
                    else:
                        st.success(f"**Advisory:**\n\n{result['advice']}")

                    with st.expander("🛠️ View Normalized Message & Debug Details"):
                        st.write("**Original Text:**", result["raw_text"])
                        st.write("**Preprocessed Text for ML:**", result["cleaned_text"])
                        st.write("**Heuristic Rule Score:**", result["signal_score"])

    # TAB 2: Batch CSV Scoring
    with tab_batch:
        st.markdown("### 📂 Batch File Processing")
        st.write("Upload a CSV file containing multiple messages to scan and classify them in bulk.")

        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"], help="The CSV must contain a text column with messages.")

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Loaded {len(batch_df)} rows and {len(batch_df.columns)} columns.")

                # Select message column
                candidate_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
                default_col = "text" if "text" in batch_df.columns else (candidate_cols[0] if candidate_cols else batch_df.columns[0])

                text_col = st.selectbox("Select the column containing the message text:", batch_df.columns, index=batch_df.columns.get_loc(default_col))

                if st.button("🚀 Process Batch Messages", type="primary"):
                    with st.spinner(f"Scoring {len(batch_df)} messages..."):
                        scored_df, summary = process_batch_dataframe(batch_df, text_col, model=model)

                    st.markdown("#### 📊 Batch Scan Results")
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Total Messages", summary["total"])
                    b2.metric("Spam / Phishing Flagged", summary["spam"])
                    b3.metric("Legitimate (Ham)", summary["ham"])
                    b4.metric("Flagged Rate", f"{summary['spam_rate']}%")

                    st.dataframe(
                        scored_df[[text_col, "Verdict", "Risk Score", "Confidence", "Signals Details"]].head(50),
                        use_container_width=True,
                    )

                    csv_data = convert_df_to_csv(scored_df)
                    st.download_button(
                        label="📥 Download Annotated CSV Results",
                        data=csv_data,
                        file_name="spamguard_scored_results.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
        else:
            st.info("Tip: You can upload your own customer service logs, SMS inbox exports, or notification tables in CSV format.")

    # TAB 3: Model Evaluation & Transparency
    with tab_metrics:
        st.markdown("### 📊 Model Performance & Validation Metrics")
        st.write("All performance numbers below reflect actual evaluation against holdout test data (80/20 stratified split, 1,115 test samples).")

        if metrics_data:
            test_m = metrics_data["test_metrics"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{test_m['accuracy'] * 100:.2f}%")
            m2.metric("Precision", f"{test_m['precision'] * 100:.2f}%")
            m3.metric("Recall", f"{test_m['recall'] * 100:.2f}%")
            m4.metric("F1-Score", f"{test_m['f1_score'] * 100:.2f}%")
            m5.metric("ROC-AUC", f"{test_m['roc_auc'] * 100:.2f}%")

            st.markdown("---")
            col_cm, col_ds = st.columns([1, 1])

            with col_cm:
                st.markdown("#### 🎯 Confusion Matrix (Holdout Test Set)")
                cm = metrics_data["confusion_matrix"]
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Ham (Safe)", "Actual Spam"],
                    columns=["Predicted Ham (Safe)", "Predicted Spam"],
                )
                st.dataframe(cm_df, use_container_width=True)
                st.caption(
                    f"**Analysis:** Out of 966 legitimate messages, {cm[0][0]} were correctly classified with only {cm[0][1]} false alarms (99.7% specificity). "
                    f"Out of 149 spam messages, {cm[1][1]} were successfully caught ({test_m['recall'] * 100:.1f}% recall)."
                )

            with col_ds:
                st.markdown("#### 📁 Dataset & Architecture Details")
                st.markdown(f"- **Total Samples:** {metrics_data['dataset_total']:,}")
                st.markdown(f"- **Legitimate Messages (Ham):** {metrics_data['dataset_ham']:,} ({(metrics_data['dataset_ham'] / metrics_data['dataset_total']) * 100:.1f}%)")
                st.markdown(f"- **Spam Messages:** {metrics_data['dataset_spam']:,} ({(metrics_data['dataset_spam'] / metrics_data['dataset_total']) * 100:.1f}%)")
                st.markdown(f"- **Training Split:** {metrics_data['train_samples']:,} samples (80%)")
                st.markdown(f"- **Testing Split:** {metrics_data['test_samples']:,} samples (20%)")
                st.markdown(f"- **Training Latency:** {metrics_data['training_time_seconds']} seconds")
                st.markdown(f"- **Pipeline:** {metrics_data['model_type']}")

    # TAB 4: Ethics & Limitations
    with tab_ethics:
        st.markdown("### ⚖️ Ethics, Privacy & Responsible AI Considerations")
        st.markdown(
            """
            #### 1. 🔒 Data Privacy & Confidentiality
            - **Local Processing:** All message validation and model inferences occur entirely on the local runtime.
            - **Zero Data Logging:** No user messages, contact numbers, or uploaded CSV files are stored or transmitted to external third-party servers.

            #### 2. ⚠️ False Positives & False Negatives
            - **False Positives (0.3% rate):** Important legitimate messages (such as bank OTPs, urgent family notifications, or delivery updates) might occasionally trigger spam signals if they contain words like "urgent" or callback numbers.
            - **False Negatives (4.0% rate):** Novel, highly sophisticated spear-phishing messages or zero-day scam templates without common keywords may evade statistical filters.
            - **Human in the Loop:** High-impact actions (such as account deactivation or message deletion) must never rely solely on automated classification; human review is strongly recommended for borderline cases.

            #### 3. 🎯 Model & Dataset Limitations
            - **Language Constraint:** The model is trained predominantly on English SMS text. Messages in non-English languages or non-Latin alphabets will not be reliably classified.
            - **Context Blindness:** The model evaluates each message independently without knowing historical sender reputation or conversation history.
            """
        )


if __name__ == "__main__":
    main()
