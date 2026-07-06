"""
Простой дашборд: показывает, у каких студентов высокий риск бросить курс
или не сдать его, и позволяет добавить нового студента вручную, чтобы
сразу увидеть его оценку риска.

Запуск:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_processing import load_raw, make_snapshot, RISK_RESULTS
from features import build_feature_matrix, align_columns, CATEGORICAL, NUMERIC
from explain import load_model, explain_student
from labels import humanize_feature

st.set_page_config(page_title="Риск отчисления студентов", layout="wide")

st.title("🎓 Кто из студентов рискует бросить курс")
st.caption(
    "Программа смотрит на активность и оценки студента и показывает, "
    "у кого выше шанс не закончить курс. Данные учебные (придуманные), "
    "чтобы можно было безопасно потренироваться."
)

WEEK_OPTIONS = [4, 8, 12]


@st.cache_data
def get_raw_data():
    return load_raw()


@st.cache_resource
def get_model(week):
    return load_model(week)


def render_explanation(bundle, X_row, background):
    with st.spinner("Считаю, какие причины повлияли на оценку риска..."):
        contrib = explain_student(bundle, X_row, background=background, top_n=6)
    contrib = contrib.copy()
    contrib["Показатель"] = contrib["feature"].apply(humanize_feature)
    contrib = contrib.rename(columns={"direction": "Как влияет", "value": "Значение"})

    colA, colB = st.columns([1, 1])
    with colA:
        st.dataframe(
            contrib[["Показатель", "Значение", "Как влияет"]],
            use_container_width=True, hide_index=True,
        )
    with colB:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#d64545" if v > 0 else "#4a7fd6" for v in contrib["shap_value"]]
        ax.barh(contrib["Показатель"], contrib["shap_value"], color=colors)
        ax.set_xlabel("Красное — повышает риск, синее — снижает")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)


student_info, weekly = get_raw_data()

tab1, tab2 = st.tabs(["📋 Список студентов", "➕ Добавить своего студента"])

# ──────────────────────────────────────────────────────────────
# ВКЛАДКА 1 — список студентов из учебных данных
# ──────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        week = st.selectbox(
            "На какой неделе курса смотрим", WEEK_OPTIONS, index=1,
            help="Чем раньше неделя — тем раньше можно заметить риск, но оценка менее точная",
        )
    with col2:
        course = st.selectbox("Курс", ["Все"] + sorted(student_info["course"].unique().tolist()))

    snapshot = make_snapshot(student_info, weekly, week)
    if course != "Все":
        snapshot = snapshot[snapshot["course"] == course]

    X, y, ids = build_feature_matrix(snapshot)
    bundle = get_model(week)
    X_aligned = align_columns(X, bundle["feature_columns"])

    proba = bundle["model"].predict_proba(X_aligned)[:, 1]
    snapshot = snapshot.reset_index(drop=True)
    snapshot["Оценка риска"] = proba

    st.markdown(f"**Неделя:** {week} · **Всего студентов:** {len(snapshot)}")

    threshold = st.slider(
        "Показывать студентов с оценкой риска выше, чем", 0, 100, 50, 5,
        format="%d%%",
    ) / 100

    at_risk_df = snapshot[snapshot["Оценка риска"] >= threshold].sort_values(
        "Оценка риска", ascending=False
    )

    st.subheader(f"🚨 В зоне риска: {len(at_risk_df)} из {len(snapshot)} студентов")
    display_df = at_risk_df.rename(columns={
        "student_id": "ID студента",
        "course": "Курс",
        "cum_clicks": "Заходов в систему",
        "avg_assessment_so_far": "Средний балл",
        "num_prev_attempts": "Неудачных попыток раньше",
    })
    display_cols = ["ID студента", "Курс", "Оценка риска", "Заходов в систему",
                     "Средний балл", "Неудачных попыток раньше"]
    st.dataframe(
        display_df[display_cols].style.format({
            "Оценка риска": "{:.0%}", "Средний балл": "{:.1f}"
        }),
        use_container_width=True, height=350, hide_index=True,
    )

    st.divider()
    st.subheader("🔍 Почему у конкретного студента такая оценка")

    if len(at_risk_df) > 0:
        selected_id = st.selectbox("Выбери студента из списка выше", at_risk_df["student_id"].tolist())
        row_idx = snapshot.index[snapshot["student_id"] == selected_id][0]
        X_row = X_aligned.iloc[[row_idx]]
        background = X_aligned.sample(min(100, len(X_aligned)), random_state=0)

        st.metric("Оценка риска", f"{snapshot.loc[row_idx, 'Оценка риска']:.0%}")
        render_explanation(bundle, X_row, background)
    else:
        st.info("Нет студентов с такой высокой оценкой риска — попробуй снизить порог выше.")

# ──────────────────────────────────────────────────────────────
# ВКЛАДКА 2 — добавить своего студента вручную
# ──────────────────────────────────────────────────────────────
with tab2:
    st.write(
        "Впиши данные одного студента, и программа сразу покажет, "
        "какой у него риск не закончить курс."
    )

    new_week = st.selectbox(
        "На какой неделе курса он сейчас находится", WEEK_OPTIONS, index=1, key="new_week"
    )
    new_bundle = get_model(new_week)

    with st.form("new_student_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_course = st.selectbox("Курс", sorted(student_info["course"].unique().tolist()))
            in_gender = st.selectbox("Пол", ["M", "F"], format_func=lambda x: "Мужской" if x == "M" else "Женский")
            in_age = st.selectbox("Возраст", ["0-35", "35-55", "55<="])
        with c2:
            in_education = st.selectbox(
                "Образование", sorted(student_info["highest_education"].unique().tolist())
            )
            in_region = st.selectbox("Регион", sorted(student_info["region"].unique().tolist()))
            in_disability = st.selectbox(
                "Инвалидность", ["N", "Y"], format_func=lambda x: "Нет" if x == "N" else "Да"
            )
        with c3:
            in_prev_attempts = st.number_input("Сколько раз уже безуспешно проходил курс", 0, 5, 0)
            in_credits = st.number_input("Учебная нагрузка (кредиты)", 30, 180, 60, step=30)
            in_clicks = st.number_input("Сколько раз заходил в систему за всё время", 0, 500, 100)
        in_score = st.slider("Средний балл за уже сданные задания", 0, 100, 60)
        in_had_assessment = st.checkbox("Уже сдавал хотя бы одно задание", value=True)

        submitted = st.form_submit_button("Посчитать риск")

    if submitted:
        n_weeks_active_guess = max(1, round(in_clicks / 15))
        row = pd.DataFrame([{
            "course": in_course,
            "gender": in_gender,
            "region": in_region,
            "highest_education": in_education,
            "age_band": in_age,
            "num_prev_attempts": in_prev_attempts,
            "studied_credits": in_credits,
            "disability": in_disability,
            "cum_clicks": in_clicks,
            "avg_weekly_clicks": in_clicks / max(new_week, 1),
            "last_week_clicks": in_clicks / max(new_week, 1),
            "n_weeks_active": min(n_weeks_active_guess, new_week),
            "avg_assessment_so_far": in_score,
            "had_assessment": int(in_had_assessment),
        }])

        X_new = pd.get_dummies(row[CATEGORICAL + NUMERIC], columns=CATEGORICAL, drop_first=True).astype(float)
        X_new_aligned = align_columns(X_new, new_bundle["feature_columns"])

        risk = new_bundle["model"].predict_proba(X_new_aligned)[:, 1][0]

        st.divider()
        st.metric("Оценка риска для этого студента", f"{risk:.0%}")
        if risk >= 0.5:
            st.warning("Высокий риск — стоит обратить внимание на этого студента.")
        else:
            st.success("Риск невысокий.")

        background_source, _ = get_raw_data()
        demo_snapshot = make_snapshot(*get_raw_data(), new_week)
        X_demo, _, _ = build_feature_matrix(demo_snapshot)
        background = align_columns(X_demo, new_bundle["feature_columns"]).sample(
            min(100, len(X_demo)), random_state=0
        )

        st.subheader("Почему программа так решила")
        render_explanation(new_bundle, X_new_aligned, background)

st.divider()
st.caption(
    "Красный цвет в графике — этот показатель повышает риск. "
    "Синий — снижает риск."
)
