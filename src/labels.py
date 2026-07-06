"""
Человеко-понятные подписи для технических названий признаков.
Нужен только для того, чтобы в интерфейсе показывать "Средний балл за задания"
вместо "avg_assessment_so_far".
"""

BASE_LABELS = {
    "num_prev_attempts": "Сколько раз уже проходил курс безуспешно",
    "studied_credits": "Учебная нагрузка (кредиты)",
    "cum_clicks": "Всего заходов в систему с начала курса",
    "avg_weekly_clicks": "Сколько раз в среднем заходит в неделю",
    "last_week_clicks": "Заходы за последнюю неделю",
    "n_weeks_active": "Сколько недель хоть раз заходил",
    "avg_assessment_so_far": "Средний балл за уже сданные задания",
    "had_assessment": "Уже сдавал хотя бы одно задание",
    "gender_M": "Пол — мужской",
    "disability_Y": "Есть инвалидность",
}

PREFIX_LABELS = {
    "course_": "Курс",
    "region_": "Регион",
    "highest_education_": "Образование",
    "age_band_": "Возраст",
}


def humanize_feature(name: str) -> str:
    if name in BASE_LABELS:
        return BASE_LABELS[name]
    for prefix, label in PREFIX_LABELS.items():
        if name.startswith(prefix):
            value = name[len(prefix):]
            return f"{label}: {value}"
    return name
