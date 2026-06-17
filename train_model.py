"""
Disease Risk Predictor — Model Training Script
Author: Mohammad Maaz
GitHub: https://github.com/MohammadMaaz6229
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")
np.random.seed(42)


def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """Generate a realistic synthetic diabetes dataset."""
    age              = np.random.randint(20, 80, n_samples)
    bmi              = np.round(np.random.normal(27, 6, n_samples).clip(15, 55), 1)
    glucose          = np.random.randint(70, 200, n_samples)
    blood_pressure   = np.random.randint(60, 130, n_samples)
    hba1c            = np.round(np.random.normal(5.5, 1.2, n_samples).clip(4, 10), 1)
    family_history   = np.random.randint(0, 2, n_samples)
    physical_activity = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2])
    smoking          = np.random.randint(0, 2, n_samples)

    risk_score = (
        (age > 45).astype(int)        * 0.30 +
        (bmi > 30).astype(int)        * 0.25 +
        (glucose > 140).astype(int)   * 0.40 +
        (blood_pressure > 90).astype(int) * 0.15 +
        (hba1c > 6.5).astype(int)     * 0.45 +
        family_history                * 0.20 +
        (physical_activity == 0).astype(int) * 0.15 +
        smoking                       * 0.10 +
        np.random.normal(0, 0.1, n_samples)
    )
    diabetes = (risk_score > 0.6).astype(int)

    return pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "glucose_level": glucose,
        "blood_pressure": blood_pressure,
        "hba1c": hba1c,
        "family_history": family_history,
        "physical_activity_level": physical_activity,
        "smoking": smoking,
        "diabetes_risk": diabetes,
    })


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Train a Gradient Boosting pipeline and print evaluation metrics."""
    X = df.drop("diabetes_risk", axis=1)
    y = df["diabetes_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.1, random_state=42
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, y_prob):.4f}")
    cv = cross_val_score(pipeline, X, y, cv=5)
    print(f"  CV Score  : {cv.mean():.4f} ± {cv.std():.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["No Diabetes", "Diabetes"]))
    return pipeline


def save_artifacts(pipeline: Pipeline, feature_names: list) -> None:
    """Save model and feature importances."""
    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("✅ model.pkl saved")

    importances = pipeline.named_steps["clf"].feature_importances_
    feat_imp = {k: round(float(v), 4)
                for k, v in zip(feature_names, importances)}
    with open("feature_importance.json", "w") as f:
        json.dump(feat_imp, f, indent=2)
    print("✅ feature_importance.json saved")


if __name__ == "__main__":
    print("🔄 Generating dataset...")
    df = generate_dataset(2000)
    df.to_csv("diabetes_data.csv", index=False)
    print(f"✅ Dataset saved — {len(df)} rows, "
          f"{df['diabetes_risk'].mean()*100:.1f}% positive class\n")

    pipeline = train_and_evaluate(df)
    feature_names = [c for c in df.columns if c != "diabetes_risk"]
    save_artifacts(pipeline, feature_names)
    print("\n🚀 All done! Run: streamlit run app.py")
