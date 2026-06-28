"""
pipeline.py
A single, disease-agnostic ML pipeline used for every dataset in this project.
This is the core idea behind "MediScan AI": one well-tested pipeline,
reused across diabetes / heart disease / breast cancer, rather than
copy-pasted notebook code per dataset.

Stages:
  1. Preprocessing      -> scaling
  2. Unsupervised        -> KMeans clustering (patient subtyping)
  3. Deep unsupervised   -> Autoencoder (reconstruction-error anomaly detection)
  4. Supervised          -> Logistic Regression + GridSearchCV-tuned Random Forest
  5. Deep supervised     -> Keras Neural Network (Dropout + EarlyStopping)
  6. Ensemble            -> Soft-voting classifier
  7. Evaluation          -> Accuracy/Precision/Recall/F1, 5-fold CV, ROC-AUC
  8. Explainability      -> SHAP (TreeExplainer on the tuned Random Forest)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_curve, auc, silhouette_score)

import tensorflow as tf
from tensorflow import keras
import shap

tf.get_logger().setLevel('ERROR')
np.random.seed(42)


def get_feature_config(disease_key, config, df, target_col):
    """Return (and auto-fill if needed) the feature slider config for a disease."""
    feats = config["features"]
    if feats is not None:
        return feats
    auto = {}
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        auto[col] = {
            "label": col.replace('_', ' ').title(),
            "min": float(series.min()),
            "max": float(series.max()),
            "default": float(series.median()),
            "step": float((series.max() - series.min()) / 100) or 0.1,
        }
    return auto


def build_autoencoder(input_dim, encoding_dim=4):
    """A small autoencoder: learns to compress+reconstruct 'normal' patient data.
    Patients it reconstructs poorly are flagged as anomalous/unusual profiles."""
    inputs = keras.layers.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(max(input_dim // 2, encoding_dim * 2), activation='relu')(inputs)
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
    decoded = keras.layers.Dense(max(input_dim // 2, encoding_dim * 2), activation='relu')(encoded)
    decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)
    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def train_pipeline(disease_key, config, verbose=True):
    """Run the full pipeline for one disease and return a results dict
    containing every fitted artifact + computed metric, ready to be saved
    or consumed directly by the Streamlit app."""

    df, target_col = config["loader"]()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Unsupervised: KMeans (best k via silhouette) ---
    sil_scores = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(sil_scores, key=sil_scores.get)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_scaled)

    # --- 3. Deep unsupervised: Autoencoder anomaly detection ---
    tf.random.set_seed(42)
    autoencoder = build_autoencoder(X_train.shape[1])
    autoencoder.fit(X_train, X_train, epochs=100, batch_size=16, verbose=0,
                     validation_split=0.2,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                               restore_best_weights=True)])
    train_recon = autoencoder.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - train_recon), axis=1)
    anomaly_threshold = float(np.percentile(train_errors, 95))  # top 5% = "unusual"

    # --- 4. Supervised: Logistic Regression + tuned Random Forest ---
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    rf_pred = rf.predict(X_test)

    cv_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring='accuracy')
    cv_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')

    # --- 5. Deep supervised: Neural Network ---
    tf.random.set_seed(42)
    nn = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = nn.fit(X_train, y_train, epochs=120, batch_size=16, validation_split=0.2, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                                                restore_best_weights=True)])
    nn_pred_prob = nn.predict(X_test, verbose=0).flatten()
    nn_pred = (nn_pred_prob > 0.5).astype(int)

    # --- 6. Ensemble ---
    voting = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
    voting.fit(X_train, y_train)
    voting_pred = voting.predict(X_test)

    # --- 7. Evaluation ---
    def m(name, y_true, y_pred):
        return {'Model': name,
                'Accuracy': round(accuracy_score(y_true, y_pred), 3),
                'Precision': round(precision_score(y_true, y_pred, zero_division=0), 3),
                'Recall': round(recall_score(y_true, y_pred, zero_division=0), 3),
                'F1-score': round(f1_score(y_true, y_pred, zero_division=0), 3)}

    results = pd.DataFrame([
        m('Logistic Regression', y_test, lr_pred),
        m('Random Forest (tuned)', y_test, rf_pred),
        m('Neural Network', y_test, nn_pred),
        m('Ensemble', y_test, voting_pred),
    ])

    roc_data = {}
    for probs, name in [(lr.predict_proba(X_test)[:, 1], 'Logistic Regression'),
                         (rf.predict_proba(X_test)[:, 1], 'Random Forest (tuned)'),
                         (nn_pred_prob, 'Neural Network'),
                         (voting.predict_proba(X_test)[:, 1], 'Ensemble')]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(float(auc(fpr, tpr)), 3)}

    # --- 8. Explainability ---
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    if verbose:
        print(f"[{disease_key}] best_k={best_k}  |  RF best params={grid.best_params_}")
        print(results)

    return {
        "disease_key": disease_key,
        "feature_names": feature_names,
        "scaler": scaler,
        "kmeans": kmeans,
        "best_k": best_k,
        "silhouette_scores": sil_scores,
        "autoencoder": autoencoder,
        "anomaly_threshold": anomaly_threshold,
        "lr": lr,
        "rf": rf,
        "rf_best_params": grid.best_params_,
        "nn": nn,
        "nn_history": history.history,
        "voting": voting,
        "results_table": results,
        "cv_scores": {"Logistic Regression": cv_lr.tolist(), "Random Forest (tuned)": cv_rf.tolist()},
        "roc_data": roc_data,
        "shap_values": sv,
        "X_test": X_test,
        "y_test": y_test.values,
        "explainer": explainer,
    }


def save_artifacts(results, out_dir):
    """Persist everything the Streamlit app needs to make instant predictions
    without retraining at startup."""
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(results["scaler"], f"{out_dir}/scaler.joblib")
    joblib.dump(results["kmeans"], f"{out_dir}/kmeans.joblib")
    joblib.dump(results["lr"], f"{out_dir}/lr.joblib")
    joblib.dump(results["rf"], f"{out_dir}/rf.joblib")
    joblib.dump(results["voting"], f"{out_dir}/voting.joblib")
    results["nn"].save(f"{out_dir}/nn.keras")
    results["autoencoder"].save(f"{out_dir}/autoencoder.keras")

    meta = {
        "disease_key": results["disease_key"],
        "feature_names": results["feature_names"],
        "best_k": results["best_k"],
        "silhouette_scores": results["silhouette_scores"],
        "anomaly_threshold": results["anomaly_threshold"],
        "rf_best_params": results["rf_best_params"],
        "results_table": results["results_table"].to_dict(orient="records"),
        "cv_scores": results["cv_scores"],
        "roc_data": results["roc_data"],
    }
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved artifacts to {out_dir}/")


def load_artifacts(out_dir):
    """Load everything the app needs for a given disease."""
    scaler = joblib.load(f"{out_dir}/scaler.joblib")
    kmeans = joblib.load(f"{out_dir}/kmeans.joblib")
    lr = joblib.load(f"{out_dir}/lr.joblib")
    rf = joblib.load(f"{out_dir}/rf.joblib")
    voting = joblib.load(f"{out_dir}/voting.joblib")
    nn = keras.models.load_model(f"{out_dir}/nn.keras")
    autoencoder = keras.models.load_model(f"{out_dir}/autoencoder.keras")
    with open(f"{out_dir}/metadata.json") as f:
        meta = json.load(f)
    return {
        "scaler": scaler, "kmeans": kmeans, "lr": lr, "rf": rf, "voting": voting,
        "nn": nn, "autoencoder": autoencoder, "meta": meta,
    }
