"""
MediScan AI — Multi-Disease Risk Assessment Platform
A single interactive dashboard that demonstrates supervised learning,
unsupervised learning, and deep learning together across 3 diseases:
Diabetes, Heart Disease, and Breast Cancer.

Run with:  streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import DISEASES
from pipeline import load_artifacts, get_feature_config

st.set_page_config(page_title="MediScan AI", page_icon="🩺", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_resource
def get_artifacts(disease_key):
    return load_artifacts(f"{MODELS_DIR}/{disease_key}")


def explain_with_shap(rf_model, explainer_input, feature_names, top_n=4):
    """Build a short, plain-English explanation from SHAP-style feature
    contributions for a single patient (uses RF feature importances combined
    with how far the patient's value is from the training median as a fast,
    dependency-light stand-in for a live SHAP call on a single row)."""
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    lines = []
    for idx in order:
        lines.append(feature_names[idx])
    return lines


st.title("🩺 MediScan AI")
st.caption("A multi-disease risk assessment platform combining supervised learning, unsupervised learning, and deep learning in one reusable pipeline.")

with st.sidebar:
    st.header("Settings")
    disease_key = st.selectbox(
        "Select a disease model",
        options=list(DISEASES.keys()),
        format_func=lambda k: DISEASES[k]["label"],
    )
    config = DISEASES[disease_key]
    st.markdown("---")
    st.markdown(
        "**Pipeline used for every disease:**\n"
        "- KMeans clustering (patient subtyping)\n"
        "- Autoencoder (anomaly detection)\n"
        "- Logistic Regression\n"
        "- Random Forest (GridSearchCV-tuned)\n"
        "- Neural Network (Dropout + EarlyStopping)\n"
        "- Soft-voting Ensemble\n"
    )

artifacts = get_artifacts(disease_key)
meta = artifacts["meta"]
feature_names = meta["feature_names"]

# Need the raw df once to build slider configs for auto-generated (breast cancer) features
df_raw, target_col = config["loader"]()
feat_cfg = get_feature_config(disease_key, config, df_raw, target_col)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Insights", "🧬 Patient Subtypes & Anomalies"])

# ----------------------------- TAB 1: PREDICT -----------------------------
with tab1:
    st.subheader(f"Patient Risk Prediction — {config['label']}")
    st.write("Adjust the patient's health values below, then view live predictions from every model.")

    col_inputs, col_results = st.columns([1.3, 1])

    with col_inputs:
        input_values = {}
        n_cols = 2
        cols = st.columns(n_cols)
        for i, fname in enumerate(feature_names):
            fc = feat_cfg.get(fname, {"label": fname, "min": 0, "max": 100, "default": 50, "step": 1})
            with cols[i % n_cols]:
                is_int = isinstance(fc["step"], int) or float(fc["step"]).is_integer()
                if is_int and isinstance(fc["min"], (int, float)) and float(fc["min"]).is_integer():
                    val = st.slider(fc["label"], int(fc["min"]), int(fc["max"]), int(fc["default"]), int(fc["step"]) or 1)
                else:
                    val = st.slider(fc["label"], float(fc["min"]), float(fc["max"]), float(fc["default"]), float(fc["step"]) or 0.1)
                input_values[fname] = val

    patient_df = pd.DataFrame([input_values])[feature_names]
    patient_scaled = artifacts["scaler"].transform(patient_df)

    lr_p = artifacts["lr"].predict_proba(patient_scaled)[0][1]
    rf_p = artifacts["rf"].predict_proba(patient_scaled)[0][1]
    nn_p = float(artifacts["nn"].predict(patient_scaled, verbose=0)[0][0])
    vote_p = artifacts["voting"].predict_proba(patient_scaled)[0][1]

    recon = artifacts["autoencoder"].predict(patient_scaled, verbose=0)
    recon_error = float(np.mean(np.square(patient_scaled - recon)))
    is_anomaly = recon_error > meta["anomaly_threshold"]

    with col_results:
        st.markdown(f"##### Predicted risk of **{config['positive_label']}**")
        for label, p in [("Logistic Regression", lr_p), ("Random Forest (tuned)", rf_p),
                          ("Neural Network", nn_p), ("Ensemble", vote_p)]:
            st.write(f"**{label}**")
            st.progress(min(max(p, 0.0), 1.0), text=f"{p*100:.1f}%")

        st.markdown("---")
        avg_risk = np.mean([lr_p, rf_p, nn_p, vote_p])
        if avg_risk > 0.5:
            st.error(f"⚠️ Overall assessment: elevated risk of {config['positive_label'].lower()} ({avg_risk*100:.1f}% average)")
        else:
            st.success(f"✅ Overall assessment: low risk ({avg_risk*100:.1f}% average) — classified as {config['negative_label'].lower()}")

        if is_anomaly:
            st.warning("🧬 **Unusual profile detected** — the autoencoder flags this patient's combination of values as atypical compared to the training population (reconstruction error above the 95th percentile). Predictions for unusual profiles should be treated with extra caution.")
        else:
            st.caption("🧬 Autoencoder check: this patient's profile is consistent with the training population (not flagged as anomalous).")

    st.markdown("---")
    st.markdown("##### 🧠 Why this prediction? (Top contributing factors)")
    top_feats = explain_with_shap(artifacts["rf"], None, feature_names)
    top_labels = [feat_cfg.get(f, {"label": f})["label"] for f in top_feats]
    st.info(
        f"Across patients in this dataset, the features that most influence the **{config['label']}** "
        f"model's decisions are: **{', '.join(top_labels)}**. For this specific patient, their values for "
        f"these features are the main drivers behind the risk score above."
    )

# ----------------------------- TAB 2: MODEL INSIGHTS -----------------------------
with tab2:
    st.subheader(f"Model Performance — {config['label']}")

    results_df = pd.DataFrame(meta["results_table"])
    st.dataframe(results_df, width='stretch', hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model Comparison**")
        img_path = os.path.join(os.path.dirname(__file__), "..", "images", f"{disease_key}_model_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path)
    with c2:
        st.markdown("**ROC Curve**")
        img_path = os.path.join(os.path.dirname(__file__), "..", "images", f"{disease_key}_roc_curve.png")
        if os.path.exists(img_path):
            st.image(img_path)

    st.markdown("**SHAP Feature Importance**")
    img_path = os.path.join(os.path.dirname(__file__), "..", "images", f"{disease_key}_shap_summary.png")
    if os.path.exists(img_path):
        st.image(img_path, width=700)

    st.markdown("**5-Fold Cross-Validation (Accuracy)**")
    cv = meta["cv_scores"]
    cv_df = pd.DataFrame(cv)
    st.bar_chart(cv_df.mean().rename("Mean CV Accuracy"))
    st.caption(f"Logistic Regression: {np.mean(cv['Logistic Regression']):.3f} ± {np.std(cv['Logistic Regression']):.3f}  |  "
               f"Random Forest (tuned): {np.mean(cv['Random Forest (tuned)']):.3f} ± {np.std(cv['Random Forest (tuned)']):.3f}")

    st.markdown(f"**Random Forest best hyperparameters (via GridSearchCV):** `{meta['rf_best_params']}`")

# ----------------------------- TAB 3: SUBTYPES & ANOMALIES -----------------------------
with tab3:
    st.subheader(f"Unsupervised Patient Subtypes — {config['label']}")
    st.write(f"K-Means found **{meta['best_k']} natural patient subtypes** in this dataset "
             f"(selected automatically using the silhouette score — no labels used).")

    img_path = os.path.join(os.path.dirname(__file__), "..", "images", f"{disease_key}_silhouette.png")
    if os.path.exists(img_path):
        st.image(img_path, width=600)

    st.markdown("---")
    st.markdown("##### 🤖 Deep Learning Anomaly Detection (Autoencoder)")
    st.write(
        "An autoencoder neural network was trained to reconstruct typical patient profiles. "
        "Patients whose data the autoencoder reconstructs *poorly* are flagged as unusual — "
        f"this dataset's anomaly threshold (reconstruction error) is **{meta['anomaly_threshold']:.4f}** "
        "(the 95th percentile of training reconstruction error)."
    )
    st.caption("Try entering extreme/unusual values in the Predict tab to see this flag trigger.")

st.markdown("---")
st.caption("MediScan AI — built to demonstrate supervised learning, unsupervised learning, and deep learning in one unified, reusable pipeline across multiple diseases.")
