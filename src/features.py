"""
Feature engineering: превращает "сырой" snapshot в матрицу признаков,
готовую для обучения модели (кодирование категорий, отбор колонок).
"""

import pandas as pd

CATEGORICAL = [
    "course", "gender", "region", "highest_education", "age_band", "disability",
]
NUMERIC = [
    "num_prev_attempts", "studied_credits",
    "cum_clicks", "avg_weekly_clicks", "last_week_clicks",
    "n_weeks_active", "avg_assessment_so_far", "had_assessment",
]
TARGET = "at_risk"
ID_COL = "student_id"


def build_feature_matrix(snapshot: pd.DataFrame):
    df = snapshot.copy()
    X = pd.get_dummies(df[CATEGORICAL + NUMERIC], columns=CATEGORICAL, drop_first=True)
    X = X.astype(float)  # get_dummies даёт bool-колонки — приводим всё к float для SHAP/sklearn
    y = df[TARGET]
    ids = df[ID_COL]
    return X, y, ids


def align_columns(X: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """Приводит матрицу признаков к тому же набору колонок, что при обучении
    (нужно при инференсе на новых данных, где могут отсутствовать категории)."""
    X = X.reindex(columns=reference_columns, fill_value=0)
    return X
