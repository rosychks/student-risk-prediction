"""
Генератор синтетических данных, повторяющих структуру реального датасета
OULAD (Open University Learning Analytics Dataset, analyse.kmi.open.ac.uk).

Зачем синтетика вместо реального файла:
- OULAD распространяется отдельным архивом с сайта Open University и не лежит
  в открытом виде на GitHub/PyPI, поэтому его нельзя скачать автоматически
  из этого окружения.
- Схема таблиц и статистические свойства (доли исходов, распределения кликов,
  дисбаланс классов) в генераторе воспроизведены по документации датасета,
  так что весь пайплайн (features -> модель -> SHAP -> dashboard) работает
  один в один, как на реальных данных.

Как перейти на настоящий OULAD:
1. Скачать архив с https://analyse.kmi.open.ac.uk/open_dataset
2. Взять studentInfo.csv, studentRegistration.csv, studentVle.csv, studentAssessment.csv
3. Заменить вызов generate_all() в этом файле на реальную загрузку и агрегацию
   этих таблиц в тот же формат, что возвращает build_student_week_table()
   (см. функцию ниже — колонки должны совпадать).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STUDENTS = 6000
COURSES = ["AAA", "BBB", "CCC", "DDD", "EEE"]
REGIONS = [
    "London Region", "South Region", "North Region", "Scotland",
    "Wales", "East Anglian Region", "South West Region", "Ireland",
]
EDUCATION = [
    "Lower Than A Level", "A Level or Equivalent",
    "HE Qualification", "No Formal quals", "Post Graduate Qualification",
]
AGE_BANDS = ["0-35", "35-55", "55<="]
MAX_WEEK = 12


def _base_risk_score(df):
    """Скрытая 'истинная' склонность к отчислению/провалу, на основе которой
    затем зашумлённо генерируются клики, оценки и итоговый результат."""
    score = np.zeros(len(df))
    score += (df["age_band"] == "0-35").astype(int) * 0.15
    score += (df["highest_education"] == "No Formal quals").astype(int) * 0.35
    score += (df["highest_education"] == "Lower Than A Level").astype(int) * 0.15
    score += (df["num_prev_attempts"] >= 1).astype(int) * 0.25
    score += (df["disability"] == "Y").astype(int) * 0.10
    score += RNG.normal(0, 0.25, len(df))
    return score


def generate_student_info():
    student_id = np.arange(100000, 100000 + N_STUDENTS)
    df = pd.DataFrame({
        "student_id": student_id,
        "course": RNG.choice(COURSES, N_STUDENTS),
        "gender": RNG.choice(["M", "F"], N_STUDENTS),
        "region": RNG.choice(REGIONS, N_STUDENTS),
        "highest_education": RNG.choice(
            EDUCATION, N_STUDENTS, p=[0.35, 0.32, 0.20, 0.08, 0.05]
        ),
        "age_band": RNG.choice(AGE_BANDS, N_STUDENTS, p=[0.62, 0.33, 0.05]),
        "num_prev_attempts": RNG.choice(
            [0, 1, 2, 3], N_STUDENTS, p=[0.78, 0.15, 0.05, 0.02]
        ),
        "studied_credits": RNG.choice(
            [60, 90, 120, 150], N_STUDENTS, p=[0.5, 0.25, 0.2, 0.05]
        ),
        "disability": RNG.choice(["N", "Y"], N_STUDENTS, p=[0.9, 0.1]),
    })
    risk = _base_risk_score(df)
    df["_risk"] = risk

    # Итоговый результат курса: чем выше скрытый риск, тем выше шанс
    # Withdrawn/Fail и ниже шанс Pass/Distinction
    probs = np.stack([
        1 / (1 + np.exp(-4 * (risk - 0.55))),       # Withdrawn
        1 / (1 + np.exp(-4 * (risk - 0.35))) * 0.5,  # Fail
    ], axis=1)
    withdrawn_p = probs[:, 0]
    fail_p = np.clip(probs[:, 1] * (1 - withdrawn_p), 0, 1)
    remaining = 1 - withdrawn_p - fail_p
    distinction_p = np.clip(remaining * (0.5 - risk).clip(0, 1), 0, remaining)
    pass_p = remaining - distinction_p

    final_result = []
    for w, f, p, d in zip(withdrawn_p, fail_p, pass_p, distinction_p):
        probs_row = np.array([w, f, p, d])
        probs_row = np.clip(probs_row, 0, None)
        probs_row = probs_row / probs_row.sum()
        final_result.append(
            RNG.choice(["Withdrawn", "Fail", "Pass", "Distinction"], p=probs_row)
        )
    df["final_result"] = final_result
    return df


def generate_weekly_activity(student_info: pd.DataFrame):
    """Генерирует клики в VLE и баллы за задания по неделям (1..MAX_WEEK)."""
    rows = []
    for _, s in student_info.iterrows():
        risk = s["_risk"]
        base_clicks = max(2, RNG.normal(18 - 12 * risk, 4))
        cum_clicks = 0
        cum_score_sum = 0
        n_assessments = 0
        for week in range(1, MAX_WEEK + 1):
            # Активность падает у студентов в зоне риска ближе к отчислению
            decay = 1.0
            if risk > 0.5 and week > 4:
                decay = max(0.15, 1 - 0.12 * (week - 4))
            week_clicks = max(0, RNG.poisson(base_clicks * decay))
            cum_clicks += week_clicks

            has_assessment = week % 4 == 0
            week_score = np.nan
            if has_assessment:
                mean_score = 85 - 55 * risk
                week_score = float(np.clip(RNG.normal(mean_score, 12), 0, 100))
                cum_score_sum += week_score
                n_assessments += 1

            rows.append({
                "student_id": s["student_id"],
                "week": week,
                "week_clicks": int(week_clicks),
                "cum_clicks": int(cum_clicks),
                "assessment_score": week_score,
                "avg_assessment_so_far": (
                    cum_score_sum / n_assessments if n_assessments else np.nan
                ),
            })
    return pd.DataFrame(rows)


def build_student_week_table():
    student_info = generate_student_info()
    weekly = generate_weekly_activity(student_info)
    student_info_public = student_info.drop(columns=["_risk"])
    return student_info_public, weekly


def generate_all():
    student_info, weekly = build_student_week_table()
    student_info.to_csv(OUT_DIR / "student_info.csv", index=False)
    weekly.to_csv(OUT_DIR / "student_weekly_activity.csv", index=False)
    print(f"Сохранено: {OUT_DIR / 'student_info.csv'} ({len(student_info)} студентов)")
    print(f"Сохранено: {OUT_DIR / 'student_weekly_activity.csv'} ({len(weekly)} строк)")
    print(student_info["final_result"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    generate_all()
