"""
Streamlit dashboard for Deep-NIDS-TF.

Usage:
    streamlit run dashboard.py
"""

import json
import os

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.txt")
LOG_PATH = os.path.join(BASE_DIR, "logs", "detections.jsonl")
CM_IMAGE = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
CLF_CURVE = os.path.join(BASE_DIR, "models", "classifier_training_curve.png")
AE_CURVE = os.path.join(BASE_DIR, "models", "autoencoder_training_curve.png")

st.set_page_config(page_title="Deep-NIDS-TF Dashboard", layout="wide")
st.title("🛰️ Deep-NIDS-TF — Network Intrusion Detection Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🚨 Live Detection Log", "📈 Training Curves"])

with tab1:
    st.subheader("Evaluation Report")
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            st.code(f.read(), language="text")
    else:
        st.info("Run `python src/evaluate.py` first to generate an evaluation report.")

    if os.path.exists(CM_IMAGE):
        st.subheader("Confusion Matrix")
        st.image(CM_IMAGE)

with tab2:
    st.subheader("Live Detection Feed")
    if os.path.exists(LOG_PATH):
        rows = []
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            df = pd.DataFrame(rows)

            # Verdicts are either the exact benign class name (e.g. "BENIGN",
            # "Normal Traffic") or "ATTACK:<type>" / "ANOMALY:unclassified".
            # Checking prefixes instead of a hardcoded string keeps this
            # working no matter what the dataset calls its benign class.
            is_alert = df["verdict"].str.startswith(("ATTACK:", "ANOMALY:"))

            col1, col2, col3 = st.columns(3)
            col1.metric("Total flows scored", len(df))
            col2.metric("Alerts", int(is_alert.sum()))
            col3.metric("Benign", int((~is_alert).sum()))

            st.subheader("Verdict breakdown")
            st.bar_chart(df["verdict"].value_counts())

            st.subheader("Reconstruction error over time")
            st.line_chart(df["reconstruction_error"])

            st.subheader("Most recent alerts")
            alerts = df[is_alert].tail(50).sort_index(ascending=False)
            st.dataframe(alerts, use_container_width=True)
        else:
            st.info("Log file is empty. Run `python src/detect.py` to generate detections.")
    else:
        st.info("No detection log yet. Run `python src/detect.py` first.")

with tab3:
    st.subheader("Training Curves")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Classifier**")
        if os.path.exists(CLF_CURVE):
            st.image(CLF_CURVE)
        else:
            st.info("Run `python src/train.py` first.")
    with c2:
        st.markdown("**Autoencoder**")
        if os.path.exists(AE_CURVE):
            st.image(AE_CURVE)
        else:
            st.info("Run `python src/train.py` first.")

st.sidebar.header("Pipeline")
st.sidebar.markdown(
    """
    1. `python scripts/preprocess_traffic.py --csv data/cicids_sample.csv --label-col "Attack Type"`
    2. `python src/train.py`
    3. `python src/evaluate.py`
    4. `python src/detect.py`
    5. `streamlit run dashboard.py` (this app)
    """
)