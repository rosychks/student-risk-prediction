"""
Разведочный анализ данных (EDA): строит ключевые графики, которые
объясняют логику проекта, и сохраняет их в reports/figures/.

Запуск:
    python src/eda.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from data_processing import load_raw, make_snapshot

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110


def plot_outcome_distribution(student_info):
    counts = student_info["final_result"].value_counts()
    order = ["Withdrawn", "Fail", "Pass", "Distinction"]
    counts = counts.reindex(order)
    colors = ["#d64545", "#e08a3c", "#4a7fd6", "#3caf5c"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Распределение итоговых результатов курса")
    ax.set_ylabel("Число студентов")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "outcome_distribution.png")
    plt.close(fig)


def plot_activity_by_risk(student_info, weekly):
    df = weekly.merge(student_info[["student_id", "final_result"]], on="student_id")
    df["at_risk"] = df["final_result"].isin(["Withdrawn", "Fail"])
    agg = df.groupby(["week", "at_risk"])["week_clicks"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    for is_risk, label, color in [(False, "Успешные (Pass/Distinction)", "#4a7fd6"),
                                    (True, "В зоне риска (Fail/Withdrawn)", "#d64545")]:
        sub = agg[agg["at_risk"] == is_risk]
        ax.plot(sub["week"], sub["week_clicks"], marker="o", label=label, color=color)
    ax.axvline(4, color="gray", linestyle="--", linewidth=1, label="Снимок: неделя 4")
    ax.set_title("Средняя активность в VLE по неделям")
    ax.set_xlabel("Неделя курса")
    ax.set_ylabel("Среднее число кликов в неделю")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "activity_by_risk_group.png")
    plt.close(fig)


def plot_metrics_by_week():
    import json
    with open(ROOT / "reports" / "metrics_by_week.json") as f:
        data = json.load(f)

    weeks = []
    roc_auc = []
    pr_auc = []
    for key, val in data.items():
        week = int(key.split("_")[1])
        best = val["best_model"]
        weeks.append(week)
        roc_auc.append(val["metrics"][best]["roc_auc"])
        pr_auc.append(val["metrics"][best]["pr_auc"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(weeks, roc_auc, marker="o", label="ROC-AUC", color="#4a7fd6")
    ax.plot(weeks, pr_auc, marker="o", label="PR-AUC", color="#d64545")
    ax.set_title("Качество прогноза растёт по мере накопления данных")
    ax.set_xlabel("Неделя курса (снимок)")
    ax.set_ylabel("Метрика (лучшая модель)")
    ax.set_ylim(0.5, 0.85)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_by_week.png")
    plt.close(fig)


def main():
    student_info, weekly = load_raw()
    plot_outcome_distribution(student_info)
    plot_activity_by_risk(student_info, weekly)
    try:
        plot_metrics_by_week()
    except FileNotFoundError:
        print("Пропускаю metrics_by_week.png — сначала запусти train_model.py")
    print(f"Графики сохранены в {FIG_DIR}")


if __name__ == "__main__":
    main()
