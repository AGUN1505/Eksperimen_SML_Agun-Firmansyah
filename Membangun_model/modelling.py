"""
modelling.py
=============
Melatih model machine learning menggunakan MLflow Tracking UI
dengan autolog (Basic level).

Cara penggunaan:
    python modelling.py
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_PATH  = "heart-disease_preprocessing.csv"
EXPERIMENT_NAME = "Heart-Disease-Classification"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2


def load_data(path: str):
    """Load dan split dataset preprocessing."""
    df = pd.read_csv(path)
    X  = df.drop('target', axis=1)
    y  = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[DATA] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    """Latih model dengan MLflow autolog."""

    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest-Autolog"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("\n=== HASIL EVALUASI ===")
        print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision : {precision_score(y_test, y_pred):.4f}")
        print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred):.4f}")

    print("\n[MLflow] Run selesai. Jalankan: mlflow ui")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(DATASET_PATH)
    train(X_train, X_test, y_train, y_test)
