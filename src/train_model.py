"""
Обучает модели риска отчисления/провала для нескольких "снимков" курса
(неделя 4, 8, 12) и сравнивает, как растёт качество прогноза со временем.

Запуск:
    python src/train_model.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

from data_processing import load_raw, make_snapshot
from features import build_feature_matrix, TARGET

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

SNAPSHOT_WEEKS = [4, 8, 12]
RANDOM_STATE = 42


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "recall_at_risk": round(recall_score(y_test, preds), 4),
        "precision_at_risk": round(precision_score(y_test, preds, zero_division=0), 4),
    }


def train_for_week(student_info, weekly, week: int):
    snapshot = make_snapshot(student_info, weekly, week)
    X, y, ids = build_feature_matrix(snapshot)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    fitted_models = {}

    # Baseline (со масштабированием признаков для устойчивой сходимости)
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    logreg.fit(X_train, y_train)
    results["logistic_regression"] = evaluate(logreg, X_test, y_test)
    fitted_models["logistic_regression"] = logreg

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate(rf, X_test, y_test)
    fitted_models["random_forest"] = rf

    # LightGBM
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1,
    )
    lgbm.fit(X_train, y_train)
    results["lightgbm"] = evaluate(lgbm, X_test, y_test)
    fitted_models["lightgbm"] = lgbm

    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_model = fitted_models[best_name]

    with open(MODELS_DIR / f"model_week{week}.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_name,
            "feature_columns": list(X_train.columns),
            "week": week,
        }, f)

    return results, best_name


def main():
    student_info, weekly = load_raw()

    all_results = {}
    for week in SNAPSHOT_WEEKS:
        print(f"\n=== Обучение на снимке недели {week} ===")
        results, best_name = train_for_week(student_info, weekly, week)
        for model_name, metrics in results.items():
            marker = " <- лучшая" if model_name == best_name else ""
            print(f"  {model_name:20s} {metrics}{marker}")
        all_results[f"week_{week}"] = {"metrics": results, "best_model": best_name}

    with open(REPORTS_DIR / "metrics_by_week.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nМетрики сохранены в {REPORTS_DIR / 'metrics_by_week.json'}")
    print(f"Модели сохранены в {MODELS_DIR}/model_week{{4,8,12}}.pkl")


if __name__ == "__main__":
    main()
