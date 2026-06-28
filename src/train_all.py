"""
train_all.py
Trains the MediScan AI pipeline for every supported disease, saves all
model artifacts to models/<disease>/, and exports comparison figures
to images/ for the README and the Streamlit app.

Run from the project root:
    python src/train_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from config import DISEASES
from pipeline import train_pipeline, save_artifacts, get_feature_config

sns.set_style('whitegrid')
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

all_results_summary = []

for disease_key, config in DISEASES.items():
    print(f"\n{'='*60}\nTraining pipeline for: {config['label']}\n{'='*60}")
    results = train_pipeline(disease_key, config)
    save_artifacts(results, f"models/{disease_key}")

    # Save results table for this disease
    rt = results["results_table"].copy()
    rt.insert(0, "Disease", config["label"])
    all_results_summary.append(rt)

    # --- Per-disease figures ---
    # 1. Model comparison bar chart
    plt.figure(figsize=(8, 5))
    results["results_table"].set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", ax=plt.gca())
    plt.title(f"{config['label']} — Model Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{disease_key}_model_comparison.png", dpi=120)
    plt.close()

    # 2. ROC curves
    plt.figure(figsize=(6.5, 6))
    for name, rd in results["roc_data"].items():
        plt.plot(rd["fpr"], rd["tpr"], label=f"{name} (AUC={rd['auc']})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{config['label']} — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{disease_key}_roc_curve.png", dpi=120)
    plt.close()

    # 3. SHAP summary
    X_test_df = pd.DataFrame(results["X_test"], columns=results["feature_names"])
    shap.summary_plot(results["shap_values"], X_test_df, show=False, max_display=10)
    plt.title(f"{config['label']} — SHAP Feature Impact")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{disease_key}_shap_summary.png", dpi=120)
    plt.close()

    # 4. Silhouette scores by k
    sil = results["silhouette_scores"]
    plt.figure(figsize=(6, 4))
    plt.plot(list(sil.keys()), list(sil.values()), marker="o")
    plt.title(f"{config['label']} — Silhouette Score by k (best k={results['best_k']})")
    plt.xlabel("k"); plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{disease_key}_silhouette.png", dpi=120)
    plt.close()

    print(f"Saved figures for {disease_key}")

# --- Combined cross-disease comparison ---
combined = pd.concat(all_results_summary, ignore_index=True)
combined.to_csv("results_comparison.csv", index=False)
print("\nCombined results:\n", combined)

# Best F1 per disease, grouped bar chart across diseases
pivot = combined.pivot(index="Disease", columns="Model", values="F1-score")
plt.figure(figsize=(9, 5))
pivot.plot(kind="bar", ax=plt.gca())
plt.title("F1-score Comparison Across Diseases & Models")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/cross_disease_comparison.png", dpi=120)
plt.close()

print("\nAll diseases trained and all figures exported.")
