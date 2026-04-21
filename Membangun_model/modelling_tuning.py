"""
modelling_tuning.py
====================
Melatih model dengan hyperparameter tuning menggunakan MLflow manual logging
dan menyimpan artefak ke DagsHub (Advance level).

Fitur:
- Manual logging (menggantikan autolog)
- GridSearchCV hyperparameter tuning
- Logging metrik lengkap (autolog + 2 artefak tambahan)
- Penyimpanan artefak ke DagsHub

Cara penggunaan:
    python modelling_tuning.py
"""

import os
import json
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# KONFIGURASI DAGSHUB & MLFLOW
# ─────────────────────────────────────────────
DAGSHUB_USERNAME = "AGUN1505"
DAGSHUB_REPO     = "Eksperimen_SML_Agun-Firmansyah"
EXPERIMENT_NAME  = "Heart-Disease-Tuning"
DATASET_PATH     = "heart-disease_preprocessing.csv"
RANDOM_STATE     = 42
TEST_SIZE        = 0.2


def init_dagshub():
    """Inisialisasi koneksi DagsHub dan MLflow."""
    dagshub.init(
        repo_owner=DAGSHUB_USERNAME,
        repo_name=DAGSHUB_REPO,
        mlflow=True
    )
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[DagsHub] Terhubung ke {DAGSHUB_USERNAME}/{DAGSHUB_REPO}")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str):
    """Load dan split dataset preprocessing."""
    df = pd.read_csv(path)
    X  = df.drop('target', axis=1)
    y  = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[DATA] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# ARTEFAK TAMBAHAN 1 — CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, path="confusion_matrix.png"):
    """Simpan plot confusion matrix sebagai artefak."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix — Best Model', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ─────────────────────────────────────────────
# ARTEFAK TAMBAHAN 2 — ROC CURVE PLOT
# ─────────────────────────────────────────────
def save_roc_curve(model, X_test, y_test, path="roc_curve.png"):
    """Simpan plot ROC Curve sebagai artefak."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — Best Model', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ─────────────────────────────────────────────
# ARTEFAK TAMBAHAN 3 — FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
def save_feature_importance(model, feature_names, path="feature_importance.png"):
    """Simpan plot feature importance sebagai artefak."""
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax.barh(
        range(top_n),
        importances[indices[:top_n]][::-1],
        color=colors, edgecolor='black', alpha=0.85
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n]][::-1], fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title('Top Feature Importances — Best Model', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ─────────────────────────────────────────────
# TRAINING + TUNING SATU MODEL
# ─────────────────────────────────────────────
def train_with_tuning(
    model_name, estimator, param_grid,
    X_train, X_test, y_train, y_test,
    feature_names
):
    """
    Latih satu model dengan GridSearchCV dan log manual ke MLflow/DagsHub.
    """
    print(f"\n{'='*50}")
    print(f"  Training: {model_name}")
    print(f"{'='*50}")

    # ── GridSearchCV ───────────────────────────────
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"  Best params: {best_params}")

    # ── Prediksi & metrik ──────────────────────────
    y_pred  = best_model.predict(X_test)
    y_prob  = best_model.predict_proba(X_test)[:, 1] \
              if hasattr(best_model, 'predict_proba') else None

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    # Cross-val score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])
    print(f"\n{report}")

    # ── MLflow manual logging ──────────────────────
    with mlflow.start_run(run_name=f"{model_name}-Tuning"):

        # Log hyperparameter terbaik
        mlflow.log_params(best_params)
        mlflow.log_param("model_name",   model_name)
        mlflow.log_param("cv_folds",     5)
        mlflow.log_param("test_size",    TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log metrik standar (sama dengan autolog)
        mlflow.log_metric("accuracy",           acc)
        mlflow.log_metric("precision",          prec)
        mlflow.log_metric("recall",             rec)
        mlflow.log_metric("f1_score",           f1)
        mlflow.log_metric("roc_auc",            roc_auc)

        # Log metrik tambahan (melebihi autolog)
        mlflow.log_metric("cv_f1_mean",         cv_mean)
        mlflow.log_metric("cv_f1_std",          cv_std)
        mlflow.log_metric("best_cv_score",      grid_search.best_score_)

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # ── Artefak tambahan 1: Confusion Matrix ──
        cm_path = save_confusion_matrix(y_test, y_pred, f"confusion_matrix_{model_name}.png")
        mlflow.log_artifact(cm_path, artifact_path="plots")
        os.remove(cm_path)

        # ── Artefak tambahan 2: ROC Curve ─────────
        if y_prob is not None:
            roc_path = save_roc_curve(best_model, X_test, y_test, f"roc_curve_{model_name}.png")
            mlflow.log_artifact(roc_path, artifact_path="plots")
            os.remove(roc_path)

        # ── Artefak tambahan 3: Feature Importance ─
        fi_path = save_feature_importance(best_model, feature_names, f"feature_importance_{model_name}.png")
        if fi_path:
            mlflow.log_artifact(fi_path, artifact_path="plots")
            os.remove(fi_path)

        # ── Artefak tambahan 4: Classification Report (txt) ─
        report_path = f"classification_report_{model_name}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Best Params: {best_params}\n\n")
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")
        os.remove(report_path)

        run_id = mlflow.active_run().info.run_id

    print(f"  [MLflow] Run ID: {run_id}")
    print(f"  Accuracy={acc:.4f} | F1={f1:.4f} | ROC-AUC={roc_auc:.4f}")

    return {
        "model_name": model_name,
        "run_id":     run_id,
        "best_params": best_params,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1_score":   f1,
        "roc_auc":    roc_auc,
        "cv_f1_mean": cv_mean,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # 1. Init DagsHub
    init_dagshub()

    # 2. Load data
    X_train, X_test, y_train, y_test = load_data(DATASET_PATH)
    feature_names = list(X_train.columns)

    # 3. Definisi model + param grid
    models = [
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth":    [4, 6, 8],
                "min_samples_split": [2, 5],
            }
        },
        {
            "name": "GradientBoosting",
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5],
            }
        },
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            "param_grid": {
                "C":      [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"],
            }
        },
    ]

    # 4. Latih semua model
    results = []
    for m in models:
        result = train_with_tuning(
            model_name    = m["name"],
            estimator     = m["estimator"],
            param_grid    = m["param_grid"],
            X_train       = X_train,
            X_test        = X_test,
            y_train       = y_train,
            y_test        = y_test,
            feature_names = feature_names,
        )
        results.append(result)

    # 5. Ringkasan hasil
    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    print("\n" + "="*60)
    print("         RINGKASAN HASIL TUNING")
    print("="*60)
    print(results_df[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))
    print("="*60)

    best = results_df.iloc[0]
    print(f"\n🏆 Model terbaik: {best['model_name']}")
    print(f"   F1-Score : {best['f1_score']:.4f}")
    print(f"   ROC-AUC  : {best['roc_auc']:.4f}")
    print(f"   Run ID   : {best['run_id']}")
    print(f"\n[DagsHub] Cek artefak di:")
    print(f"  https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")


if __name__ == "__main__":
    main()
