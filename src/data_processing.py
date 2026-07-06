"""
Загрузка сырых таблиц и построение "снимков" (snapshot) состояния студента
на конкретную неделю курса — без утечки данных из будущего (data leakage).

Ключевая идея проекта: модель должна уметь предсказывать риск как можно
раньше, используя только ту информацию, которая реально доступна
преподавателю на данной неделе.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

RISK_RESULTS = {"Withdrawn", "Fail"}


def load_raw():
    student_info = pd.read_csv(RAW_DIR / "student_info.csv")
    weekly = pd.read_csv(RAW_DIR / "student_weekly_activity.csv")
    return student_info, weekly


def make_snapshot(student_info: pd.DataFrame, weekly: pd.DataFrame, week: int) -> pd.DataFrame:
    """
    Возвращает одну строку на студента с информацией, доступной ТОЛЬКО
    до указанной недели включительно (weekly содержит только week <= snapshot_week).
    """
    w = weekly[weekly["week"] <= week].copy()

    agg = (
        w.sort_values("week")
        .groupby("student_id")
        .agg(
            cum_clicks=("cum_clicks", "last"),
            avg_weekly_clicks=("week_clicks", "mean"),
            last_week_clicks=("week_clicks", "last"),
            avg_assessment_so_far=("avg_assessment_so_far", "last"),
            n_weeks_active=("week_clicks", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    df = student_info.merge(agg, on="student_id", how="left")

    # Студенты без активности к этой неделе — заполняем нулями/NaN осмысленно
    df["cum_clicks"] = df["cum_clicks"].fillna(0)
    df["avg_weekly_clicks"] = df["avg_weekly_clicks"].fillna(0)
    df["last_week_clicks"] = df["last_week_clicks"].fillna(0)
    df["n_weeks_active"] = df["n_weeks_active"].fillna(0)
    # Если ещё не было оценивания — используем нейтральное значение медианы
    median_score = df["avg_assessment_so_far"].median()
    df["had_assessment"] = df["avg_assessment_so_far"].notna().astype(int)
    df["avg_assessment_so_far"] = df["avg_assessment_so_far"].fillna(median_score)

    df["snapshot_week"] = week
    df["at_risk"] = df["final_result"].isin(RISK_RESULTS).astype(int)
    return df


if __name__ == "__main__":
    info, weekly = load_raw()
    snap = make_snapshot(info, weekly, week=4)
    print(snap.head())
    print("Доля риска:", snap["at_risk"].mean().round(3))
