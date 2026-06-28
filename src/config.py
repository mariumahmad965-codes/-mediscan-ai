"""
config.py
Central configuration for all supported diseases: how to load data,
which column is the target, and UI metadata (ranges/defaults) for each
feature — used by both the training pipeline and the Streamlit app.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = "data"


def _load_diabetes():
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(f"{DATA_DIR}/diabetes.csv", names=cols)
    for c in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[c] = df[c].replace(0, df[c].median())
    return df, 'Outcome'


def _load_heart():
    df = pd.read_csv(f"{DATA_DIR}/heart.csv")
    return df, 'target'


def _load_breast_cancer():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Flip so 1 = malignant (higher risk), matching the "1 = disease present" convention
    df['target'] = 1 - df['target']
    return df, 'target'


DISEASES = {
    "diabetes": {
        "label": "Diabetes",
        "loader": _load_diabetes,
        "positive_label": "Diabetic",
        "negative_label": "Non-Diabetic",
        "features": {
            "Pregnancies": {"label": "Pregnancies", "min": 0, "max": 17, "default": 2, "step": 1},
            "Glucose": {"label": "Glucose (mg/dL)", "min": 40, "max": 250, "default": 120, "step": 1},
            "BloodPressure": {"label": "Blood Pressure (mm Hg)", "min": 30, "max": 130, "default": 70, "step": 1},
            "SkinThickness": {"label": "Skin Thickness (mm)", "min": 5, "max": 100, "default": 25, "step": 1},
            "Insulin": {"label": "Insulin (mu U/mL)", "min": 0, "max": 600, "default": 80, "step": 1},
            "BMI": {"label": "BMI", "min": 12.0, "max": 65.0, "default": 27.0, "step": 0.1},
            "DiabetesPedigreeFunction": {"label": "Diabetes Pedigree Function", "min": 0.05, "max": 2.5, "default": 0.4, "step": 0.01},
            "Age": {"label": "Age", "min": 15, "max": 90, "default": 33, "step": 1},
        },
    },
    "heart": {
        "label": "Heart Disease",
        "loader": _load_heart,
        "positive_label": "At Risk",
        "negative_label": "Low Risk",
        "features": {
            "age": {"label": "Age", "min": 20, "max": 90, "default": 50, "step": 1},
            "sex": {"label": "Sex (1=Male, 0=Female)", "min": 0, "max": 1, "default": 1, "step": 1},
            "cp": {"label": "Chest Pain Type (0-3)", "min": 0, "max": 3, "default": 1, "step": 1},
            "trestbps": {"label": "Resting Blood Pressure", "min": 80, "max": 220, "default": 130, "step": 1},
            "chol": {"label": "Cholesterol (mg/dL)", "min": 100, "max": 600, "default": 240, "step": 1},
            "fbs": {"label": "Fasting Blood Sugar > 120 (1=Yes,0=No)", "min": 0, "max": 1, "default": 0, "step": 1},
            "restecg": {"label": "Resting ECG (0-2)", "min": 0, "max": 2, "default": 1, "step": 1},
            "thalach": {"label": "Max Heart Rate Achieved", "min": 60, "max": 220, "default": 150, "step": 1},
            "exang": {"label": "Exercise-Induced Angina (1=Yes,0=No)", "min": 0, "max": 1, "default": 0, "step": 1},
            "oldpeak": {"label": "ST Depression (oldpeak)", "min": 0.0, "max": 6.5, "default": 1.0, "step": 0.1},
            "slope": {"label": "Slope of ST Segment (0-2)", "min": 0, "max": 2, "default": 1, "step": 1},
            "ca": {"label": "Major Vessels Colored (0-4)", "min": 0, "max": 4, "default": 0, "step": 1},
            "thal": {"label": "Thalassemia (0-3)", "min": 0, "max": 3, "default": 2, "step": 1},
        },
    },
    "breast_cancer": {
        "label": "Breast Cancer",
        "loader": _load_breast_cancer,
        "positive_label": "Malignant",
        "negative_label": "Benign",
        "features": None,  # auto-generated from data (30 numeric features) — see pipeline.get_feature_config
    },
}
