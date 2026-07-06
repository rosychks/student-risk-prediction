"""
Интерпретируемость моделей через SHAP: глобальная важность признаков
и объяснение конкретного прогноза для одного студента.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_model(week: int):
    with open(MODELS_DIR / f"model_week{week}.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle


def get_shap_explainer(bundle, background: pd.DataFrame = None):
    """
    Для древесных моделей (random_forest, lightgbm) используется TreeExplainer
    (быстрый, точный, фон не нужен).
    Для остальных моделей (logistic_regression и т.п.) нужен фоновый датасет
    (masker) — небольшая выборка данных, относительно которой считается вклад
    каждого признака.
    """
    model = bundle["model"]
    model_name = bundle["model_name"]
    if model_name in ("random_forest", "lightgbm"):
        return shap.TreeExplainer(model)
    if background is None:
        raise ValueError("Для не-древесных моделей нужен background-датасет (параметр background)")
    background_sample = shap.sample(background, min(100, len(background)), random_state=0)
    return shap.Explainer(model.predict_proba, background_sample)


def _extract_risk_class_values(shap_values):
    """Приводит выход shap к 1D-массиву вкладов признаков для класса 'риск' (1)."""
    if isinstance(shap_values, list):
        return np.array(shap_values[1])
    arr = np.array(shap_values)
    # Для бинарной классификации Explainer часто возвращает форму (n, n_features, 2)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def explain_student(bundle, X_row: pd.DataFrame, background: pd.DataFrame = None, top_n: int = 6):
    """Возвращает top_n признаков, сильнее всего повлиявших на прогноз
    для одного студента (по абсолютному SHAP-значению)."""
    explainer = get_shap_explainer(bundle, background)
    raw_values = explainer.shap_values(X_row) if hasattr(explainer, "shap_values") else explainer(X_row).values
    sv = _extract_risk_class_values(raw_values).flatten()

    n_features = len(bundle["feature_columns"])
    if sv.shape[0] != n_features:
        sv = sv[-n_features:]

    contrib = pd.DataFrame({
        "feature": bundle["feature_columns"],
        "value": X_row.iloc[0].values,
        "shap_value": sv,
    })
    contrib["abs_shap"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).head(top_n)
    contrib["direction"] = np.where(contrib["shap_value"] > 0, "повышает риск", "снижает риск")
    return contrib[["feature", "value", "shap_value", "direction"]]


def global_importance(bundle, X_sample: pd.DataFrame, background: pd.DataFrame = None, top_n: int = 15):
    explainer = get_shap_explainer(bundle, background if background is not None else X_sample)
    raw_values = explainer.shap_values(X_sample) if hasattr(explainer, "shap_values") else explainer(X_sample).values
    sv = _extract_risk_class_values(raw_values)

    importance = pd.DataFrame({
        "feature": bundle["feature_columns"],
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)
    return importance
