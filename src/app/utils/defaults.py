"""Загрузка схемы формы CatBoost-предсказателя из ``catboost_form_schema.json``.

Схема предсчитана офлайн на X_train (23 530 строк, 34 колонки) — медианы
для числовых, моды для категориальных, top-30 значений для каждой
категориальной колонки. Всё в одном JSON-файле размером 8.6 КБ.

Зачем не грузить ``catboost_features.pkl`` напрямую:
- Pickle на 5 МБ (X_train DataFrame целиком) + joblib + pandas-impl
- Streamlit-контейнер не должен видеть pickle с не доверенным контентом
- Чтение JSON в 100 раз быстрее (mmap-парсинг vs joblib reconstruct)

Структура schema:
    {
      "feature_columns": [...] (34, порядок как в catboost.fit),
      "cat_features": [...] (7),
      "bool_features": [...] (9),
      "numeric_features": [...] (18),
      "defaults": {col: value},
      "cat_choices": {col: [top-30 values]},
      "numeric_ranges": {col: {min, max, p05, p95}},
      "class_priors": {label: share},
      "n_train": int
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

# Путь к pre-computed schema. Относительно корня проекта (rfs CWD от
# ``streamlit run``), `Path` resolve к абсолютному.
_SCHEMA_PATH = Path("data/processed/catboost_form_schema.json")


@st.cache_resource(show_spinner=False)
def load_form_schema() -> dict[str, Any]:
    """Полная схема формы CatBoost. Один JSON-load на процесс."""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {_SCHEMA_PATH}. Запусти "
            "`python scripts/build_catboost_form_schema.py` "
            "для регенерации."
        )
    with _SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_defaults() -> dict[str, Any]:
    """Дефолты (медиана/мода) для всех 34 признаков. Готовый payload-baseline."""
    return dict(load_form_schema()["defaults"])


# Человекочитаемые подписи для группировки в форме.
# Основа — семантическая близость признаков, не их позиция в feature_columns.
FORM_GROUPS: dict[str, list[str]] = {
    "Время": ["hour", "dow", "month", "year", "is_weekend", "is_holiday"],
    "Место": ["lat", "lon", "np_top", "is_highway", "is_in_region"],
    "Условия": [
        "light_type",
        "traffic_area_state",
        "clouds_top",
        "mt_rate",
        "has_defect",
    ],
    "ТС и участники": [
        "veh_amount",
        "veh_count_actual",
        "rhd_share",
        "classified_veh",
        "avg_vehicle_year",
        "mark_top",
        "has_moto",
        "has_truck_or_bus",
    ],
    "Поведение и пешеходы": [
        "part_count",
        "drunk_share",
        "med_known_count",
        "unbelted_share",
        "avg_age_from_tg",
        "ped_count",
        "avg_ped_age_from_tg",
        "has_known_age",
        "has_known_ped_age",
    ],
    "Тип ДТП": ["em_type"],
}


# Человекочитаемые подписи для каждого признака (для UI).
FEATURE_LABELS: dict[str, str] = {
    "hour": "Час суток",
    "dow": "День недели (0=Вс, 6=Сб)",
    "month": "Месяц",
    "year": "Год",
    "lat": "Широта",
    "lon": "Долгота",
    "veh_amount": "Кол-во ТС (сообщённое)",
    "veh_count_actual": "Кол-во ТС в БД",
    "rhd_share": "Доля правого руля (RHD)",
    "classified_veh": "Классифицировано ТС",
    "avg_vehicle_year": "Средний год выпуска ТС",
    "part_count": "Кол-во участников",
    "drunk_share": "Доля пьяных среди участников",
    "med_known_count": "Кол-во прошедших медэкспертизу",
    "unbelted_share": "Доля не пристёгнутых",
    "avg_age_from_tg": "Средний возраст (Telegram NER)",
    "ped_count": "Кол-во пешеходов",
    "avg_ped_age_from_tg": "Средний возраст пешеходов",
    "light_type": "Освещение",
    "traffic_area_state": "Состояние покрытия",
    "mt_rate": "Категория местности",
    "clouds_top": "Погода",
    "em_type": "Тип ДТП",
    "np_top": "Населённый пункт",
    "mark_top": "Доминирующая марка ТС",
    "is_weekend": "Выходной день",
    "is_holiday": "Праздничный день",
    "is_highway": "На трассе",
    "is_in_region": "В Приморском крае",
    "has_defect": "Есть дефекты дороги",
    "has_moto": "Есть мото",
    "has_truck_or_bus": "Есть грузовик/автобус",
    "has_known_age": "Известен возраст водителя",
    "has_known_ped_age": "Известен возраст пешехода",
}


CLASS_LABELS_RU: dict[str, str] = {
    "light": "Лёгкая",
    "severe": "Тяжёлая",
    "severe_multiple": "Множественные ранения",
    "dead": "Смертельная",
}


CLASS_COLORS: dict[str, str] = {
    "light": "#3498db",
    "severe": "#f39c12",
    "severe_multiple": "#e67e22",
    "dead": "#c0392b",
}
