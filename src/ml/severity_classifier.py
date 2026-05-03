"""
CatBoost-классификатор тяжести ДТП (4 класса).

Что делает: строит accident-level feature matrix через единый SQL-запрос
с агрегатами по vehicles + participants + accident_pedestrians, разделяет
её на train/test со стратификацией по severity и обучает CatBoost с
auto_class_weights='Balanced' (компенсация дисбаланса 72/17/8.7/2.1%).
Возвращает обученную модель, метрики (accuracy, F1-macro, ROC-AUC,
PR-AUC по классам, confusion matrix) и feature importance.

Зачем нужно: tabular-классификация с большим количеством категориальных
признаков — "родная" задача для CatBoost. Tree-ensemble не требует
one-hot encoding (CatBoost сам обрабатывает категориальные через
ordered target statistics), не чувствителен к шкалам и пропускам.

Архитектурные решения:
- Источник истины — единый SQL CTE: accident-level + агрегаты по ТС/
  участникам/пешеходам в одной транзакции. Никаких догрузок в pandas
  через множественные запросы.
- Геопространство WGS84 lat/lon: на baseline — числа (CatBoost умеет),
  + категориальный np (топ-30 + 'other'). UTM-проекция оставлена на
 1 (тюнинг).
- Возрасты из Telegram NLP очень sparse (193 / 47 000+).
  Агрегируем на accident-level через mean(age) FILTER WHERE age IS NOT NULL,
  плюс отдельный binary-флаг has_known_age — чтобы NaN не размывал mean.
- Категориальные с большой кардинальностью (mark, np) — топ-N + 'other'.
- НЕ ИСПОЛЬЗУЕМ как признаки: lost_amount, suffer_amount, pers_amount —
  все три однозначно или почти однозначно определяют severity (data leakage).
  veh_amount — OK (число ТС, не пострадавших).

ВАЖНО — что НЕ leakage и почему:
- veh_amount: число ТС-участников; НЕ пропорционально severity (light=1.47
  ср., dead=1.47 ср., severe=1.83). Не leakage.
- features из vehicles/participants — да, это per-accident агрегаты, но
  они не содержат severity-определяющих признаков (раненых/погибших).
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# =====================================================================
# Константы
# =====================================================================

SEVERITY_CLASSES = ["light", "severe", "severe_multiple", "dead"]

# Признаки-leakage — никогда не использовать
LEAKAGE_COLUMNS = ("lost_amount", "suffer_amount", "pers_amount")

# Топ-N для категориальных с большой кардинальностью
NP_TOP_N = 30
MARK_TOP_N = 15  # топ-15 марок ТС, остальные → 'other'

# Сплит train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =====================================================================
# 1. Построение feature matrix
# =====================================================================

# Базовый SQL — accident-level с агрегатами через CTE
# Считаем агрегаты в SQL (не в pandas), потому что:
# (1) PostgreSQL быстрее в группировке;
# (2) индексы accident_id уже стоят на vehicles/participants/pedestrians.
FEATURE_QUERY = text(
    r"""
    WITH veh_agg AS (
        SELECT
            v.accident_id,
            COUNT(*) AS veh_count,
            SUM(CASE WHEN v.is_right_hand_drive IS TRUE THEN 1 ELSE 0 END)::float
                / NULLIF(SUM(CASE WHEN v.is_right_hand_drive IS NOT NULL THEN 1 ELSE 0 END), 0)
                AS rhd_share,
            SUM(CASE WHEN v.is_right_hand_drive IS NOT NULL THEN 1 ELSE 0 END)
                AS classified_veh,
            AVG(NULLIF(v.vehicle_year, 0))::float AS avg_vehicle_year,
            BOOL_OR(
                v.prod_type ILIKE '%Мото%' OR v.prod_type ILIKE '%Мопед%'
            ) AS has_moto,
            BOOL_OR(
                v.prod_type ILIKE '%Грузов%' OR v.prod_type ILIKE '%Седельн%'
                OR v.prod_type ILIKE '%Самосвал%' OR v.prod_type ILIKE '%Автобус%'
            ) AS has_truck_or_bus,
            -- Доминирующая марка (для top-N группировки в pandas)
            (
                SELECT v2.mark FROM vehicles v2
                WHERE v2.accident_id = v.accident_id AND v2.mark IS NOT NULL
                GROUP BY v2.mark
                ORDER BY COUNT(*) DESC, v2.mark
                LIMIT 1
            ) AS dominant_mark
        FROM vehicles v
        GROUP BY v.accident_id
    ),
    -- Объединяем участников (водители + пассажиры в participants) и пешеходов
    -- в единое представление per accident
    part_agg AS (
        SELECT
            v.accident_id,
            COUNT(p.id) AS part_count,
            -- Доля пьяных среди тех, у кого med_result_permille заполнен
            SUM(CASE WHEN p.med_result_permille > 0 THEN 1 ELSE 0 END)::float
                / NULLIF(SUM(CASE WHEN p.med_result_permille IS NOT NULL THEN 1 ELSE 0 END), 0)
                AS drunk_share,
            -- Сколько участников с known med_result (для контроля)
            SUM(CASE WHEN p.med_result_permille IS NOT NULL THEN 1 ELSE 0 END)
                AS med_known_count,
            -- Доля непристёгнутых среди водителей (где safety_belt='Нет')
            SUM(CASE WHEN p.safety_belt = 'Нет' THEN 1 ELSE 0 END)::float
                / NULLIF(SUM(CASE WHEN p.safety_belt IN ('Да','Нет') THEN 1 ELSE 0 END), 0)
                AS unbelted_share,
            -- Возраст из Telegram NLP — sparse признак (~193 на проект)
            AVG(p.age_from_telegram::float) FILTER (WHERE p.age_from_telegram IS NOT NULL)
                AS avg_age_from_tg
        FROM vehicles v
        LEFT JOIN participants p ON p.vehicle_id = v.id
        GROUP BY v.accident_id
    ),
    ped_agg AS (
        SELECT
            ap.accident_id,
            COUNT(*) AS ped_count,
            AVG(ap.age_from_telegram::float) FILTER (WHERE ap.age_from_telegram IS NOT NULL)
                AS avg_ped_age_from_tg
        FROM accident_pedestrians ap
        GROUP BY ap.accident_id
    )
    SELECT
        a.id,
        a.severity,
        -- ===== Временные =====
        EXTRACT(HOUR FROM a.datetime)::int AS hour,
        EXTRACT(DOW FROM a.datetime)::int  AS dow,
        EXTRACT(MONTH FROM a.datetime)::int AS month,
        EXTRACT(YEAR FROM a.datetime)::int  AS year,
        -- ===== Геопространственные =====
        ST_Y(a.point) AS lat,
        ST_X(a.point) AS lon,
        a.np,
        (a.roads IS NOT NULL AND LENGTH(TRIM(a.roads)) > 0) AS is_highway,
        COALESCE(a.is_in_region, FALSE) AS is_in_region,
        -- ===== Условия =====
        a.light_type,
        a.traffic_area_state,
        a.mt_rate,
        -- Первое значение clouds JSONB-массива
        (a.clouds->>0) AS clouds_top,
        -- has_defect = есть ли реальный дефект, кроме «Не установлены»
        (
            a.defects IS NOT NULL
            AND a.defects::text != '["Не установлены"]'
            AND jsonb_array_length(a.defects) > 0
        ) AS has_defect,
        -- ===== Тип ДТП =====
        a.em_type,
        -- ===== Аккуратное число ТС (не leakage, см. модуль docstring) =====
        a.veh_amount,
        -- ===== Агрегаты по ТС =====
        COALESCE(va.veh_count, 0) AS veh_count_actual,
        va.rhd_share,
        va.classified_veh,
        va.avg_vehicle_year,
        COALESCE(va.has_moto, FALSE) AS has_moto,
        COALESCE(va.has_truck_or_bus, FALSE) AS has_truck_or_bus,
        va.dominant_mark,
        -- ===== Агрегаты по участникам =====
        COALESCE(pa.part_count, 0) AS part_count,
        pa.drunk_share,
        pa.med_known_count,
        pa.unbelted_share,
        pa.avg_age_from_tg,
        -- ===== Пешеходы =====
        COALESCE(pe.ped_count, 0) AS ped_count,
        pe.avg_ped_age_from_tg
    FROM accidents a
    LEFT JOIN veh_agg va ON va.accident_id = a.id
    LEFT JOIN part_agg pa ON pa.accident_id = a.id
    LEFT JOIN ped_agg pe ON pe.accident_id = a.id
    WHERE a.severity IS NOT NULL
    """
)


def _topn_or_other(s: pd.Series, top_n: int, other: str = "other") -> pd.Series:
    """Оставляет top_n самых частых значений, остальное → other.

    NaN/None заменяет на other тоже (чтобы не было путаницы с
    'отсутствует' vs 'редкая категория').
    """
    counts = s.value_counts(dropna=True)
    top_values = set(counts.head(top_n).index.tolist())
    return s.where(s.isin(top_values), other=other).fillna(other)


def build_feature_matrix(session: Session) -> tuple[pd.DataFrame, pd.Series]:
    """Строит feature matrix через SQL + post-processing.

    Возвращает (X, y), где X — DataFrame признаков, y — Series severity.
    accident_id остаётся как индекс DataFrame для трассировки.

    Важно — leakage-колонки lost_amount / suffer_amount / pers_amount в
    SQL-запросе **не выбираются вообще**, чтобы не было соблазна их
    использовать.
    """
    logger.info("Загружаю accident-level feature matrix из БД...")
    df = pd.read_sql(FEATURE_QUERY, session.connection())
    logger.info("Загружено %d ДТП × %d сырых колонок", len(df), len(df.columns))

    # Защита от leakage — assert что не попали запрещённые колонки
    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            raise RuntimeError(
                f"Leakage-колонка {col!r} попала в feature matrix — это "
                f"запрещено. Проверь FEATURE_QUERY."
            )

    # accident_id → индекс
    df = df.set_index("id")
    df.index.name = "accident_id"

    # Целевая переменная
    y = df["severity"].astype("category")
    y = y.cat.set_categories(SEVERITY_CLASSES, ordered=False)

    # Удаляем ДТП без severity (если вдруг просочились)
    valid = ~y.isna()
    df = df.loc[valid].copy()
    y = y.loc[valid].copy()

    # ===== Производные временные признаки =====
    df["is_weekend"] = df["dow"].isin([5, 6])
    # Праздники РФ — через библиотеку holidays (стандарт для прогнозов)
    import holidays as _holidays

    years = sorted(df["year"].unique().tolist())
    ru_holidays = _holidays.Russia(years=years)
    # Преобразуем datetime в date для проверки попадания в праздники
    # year/month нужны только в пределах строки — реконструируем дату
    # из исходной accidents.datetime через отдельный SQL-вызов? Нет —
    # уже есть year+month+dow+hour, но не день месяца. Дешевле повторно
    # достать дату из БД.
    # Быстрее: получить день месяца отдельным SQL по accident_id,
    # но проще — пересоздать в SQL. Решение — добавить отдельную колонку
    # date_only в основном запросе. Здесь просто делаем второй проход:
    # из year/month/dow невозможно восстановить дату. Решаем переписав
    # query: см. _attach_holiday_flag.
    df["is_holiday"] = _attach_holiday_flag(session, df.index, ru_holidays)

    # ===== Категориальные top-N + 'other' =====
    df["np_top"] = _topn_or_other(df["np"], NP_TOP_N)
    df["mark_top"] = _topn_or_other(df["dominant_mark"], MARK_TOP_N)

    # ===== Бинарный флаг для возрастов (sparse) =====
    df["has_known_age"] = df["avg_age_from_tg"].notna()
    df["has_known_ped_age"] = df["avg_ped_age_from_tg"].notna()

    # ===== Категориальные NaN → 'unknown' (CatBoost требует строки/инты,
    # NaN в категориальных — частая ошибка) =====
    cat_columns_text = [
        "light_type",
        "traffic_area_state",
        "mt_rate",
        "clouds_top",
        "em_type",
        "np_top",
        "mark_top",
    ]
    for col in cat_columns_text:
        df[col] = df[col].fillna("unknown").astype(str)

    # Числовые признаки — пропуски оставляем как NaN (CatBoost умеет)
    numeric_cols = [
        "hour",
        "dow",
        "month",
        "year",
        "lat",
        "lon",
        "veh_amount",
        "veh_count_actual",
        "rhd_share",
        "classified_veh",
        "avg_vehicle_year",
        "part_count",
        "drunk_share",
        "med_known_count",
        "unbelted_share",
        "avg_age_from_tg",
        "ped_count",
        "avg_ped_age_from_tg",
    ]

    bool_cols = [
        "is_weekend",
        "is_holiday",
        "is_highway",
        "is_in_region",
        "has_defect",
        "has_moto",
        "has_truck_or_bus",
        "has_known_age",
        "has_known_ped_age",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    feature_columns = numeric_cols + cat_columns_text + bool_cols
    X = df[feature_columns].copy()

    # Sanity-check на константы и полностью-NaN колонки
    for col in X.columns:
        nun = X[col].nunique(dropna=True)
        if nun <= 1:
            logger.warning(
                "Колонка %r имеет %d уникальных значений (constant?) — "
                "это снизит её ценность для модели",
                col,
                nun,
            )

    # Sanity-check leakage ещё раз — на финальном X
    for bad in LEAKAGE_COLUMNS:
        if bad in X.columns:
            raise RuntimeError(f"Leakage-колонка {bad!r} в финальном X — баг")

    return X, y


def _attach_holiday_flag(
    session: Session,
    accident_ids: pd.Index,
    ru_holidays,
) -> pd.Series:
    """Достаёт даты для accident_ids и проверяет на праздники РФ.

    Возвращает bool-Series в том же порядке что accident_ids.
    """
    # Загружаем даты одним запросом
    ids_list = accident_ids.tolist()
    if not ids_list:
        return pd.Series([], dtype=bool)
    df_dates = pd.read_sql(
        text("SELECT id, datetime::date AS d FROM accidents WHERE id = ANY(:ids)"),
        session.connection(),
        params={"ids": ids_list},
    ).set_index("id")
    aligned = df_dates.reindex(accident_ids)["d"]
    return aligned.apply(lambda d: d in ru_holidays).astype(bool).values


def get_categorical_features(X: pd.DataFrame) -> list[str]:
    """Возвращает список колонок, которые CatBoost должен трактовать
    как категориальные. Bool-колонки в CatBoost можно как числовые
    (быстрее) или как категориальные — выбираем числовые: 0/1 — это
    интерпретируемо."""
    return [c for c in X.columns if X[c].dtype == "object"]


# =====================================================================
# 2. Сплит train/test
# =====================================================================


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Стратифицированный сплит по severity.

    StratifiedShuffleSplit гарантирует, что в test попадает минимум по
    несколько примеров каждого класса — критично для severe_multiple
    (всего 620 в датасете, 124 в test).
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_dist = Counter(y_train)
    test_dist = Counter(y_test)
    logger.info(
        "Train: %d (%s) | Test: %d (%s)",
        len(X_train),
        dict(train_dist),
        len(X_test),
        dict(test_dist),
    )
    return X_train, X_test, y_train, y_test


# =====================================================================
# 3. Обучение CatBoost
# =====================================================================


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    iterations: int = 1000,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_state: int = RANDOM_STATE,
    eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    verbose: int = 100,
) -> CatBoostClassifier:
    """Обучает CatBoost с auto_class_weights='Balanced'.

    `Balanced` означает inverse-frequency веса:
    weight_class_i = total / (n_classes * count_class_i)

    Это лучше чем oversampling/SMOTE для tree-моделей и стандарт для
    дисбаланса 5-50:1 (у нас 35:1 для severe_multiple vs light).

    early_stopping через `eval_set` + `od_type='Iter'` — сохранит время
    на тюнинге.
    """
    logger.info(
        "CatBoost training: iters=%d lr=%.3f depth=%d cat_features=%d",
        iterations,
        learning_rate,
        depth,
        len(cat_features),
    )

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="MultiClass",
        eval_metric="TotalF1",  # F1-macro / weighted альтернативы
        cat_features=cat_features,
        auto_class_weights="Balanced",
        random_seed=random_state,
        # ранняя остановка только если есть eval_set
        od_type="Iter" if eval_set is not None else None,
        od_wait=50 if eval_set is not None else None,
        verbose=verbose,
        allow_writing_files=False,  # не плодим catboost_info/
    )

    train_pool = Pool(X_train, label=y_train.astype(str), cat_features=cat_features)
    eval_pool = None
    if eval_set is not None:
        X_eval, y_eval = eval_set
        eval_pool = Pool(X_eval, label=y_eval.astype(str), cat_features=cat_features)

    model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
    logger.info("Training complete. Best iter: %s", getattr(model, "best_iteration_", "N/A"))
    return model


# =====================================================================
# 4. Оценка
# =====================================================================


def evaluate(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Считает все ключевые метрики для multiclass-задачи.

    Метрики:
    - accuracy (общая, дисбалансированная — низкая ценность)
    - F1-macro (главная — даёт равный вес всем классам)
    - F1-weighted (учитывает баланс)
    - Per-class precision / recall / F1
    - One-vs-rest ROC-AUC (особо важно для класса 'dead')
    - One-vs-rest PR-AUC (Average Precision)
    - Confusion matrix
    """
    y_pred = pd.Series(np.array(model.predict(X_test)).ravel(), index=y_test.index, name="pred")
    # Probability matrix (n_samples, n_classes)
    y_proba = model.predict_proba(X_test)
    classes_in_model = list(model.classes_)
    # Order classes_in_model in our canonical order
    class_to_col = {c: i for i, c in enumerate(classes_in_model)}

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(
            f1_score(y_test, y_pred, average="macro", labels=SEVERITY_CLASSES, zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", labels=SEVERITY_CLASSES, zero_division=0)
        ),
        "classes_order": SEVERITY_CLASSES,
    }

    # Classification report (per-class)
    report = classification_report(
        y_test,
        y_pred,
        labels=SEVERITY_CLASSES,
        target_names=SEVERITY_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {
        c: {
            "precision": float(report[c]["precision"]),
            "recall": float(report[c]["recall"]),
            "f1": float(report[c]["f1-score"]),
            "support": int(report[c]["support"]),
        }
        for c in SEVERITY_CLASSES
    }

    # ROC-AUC и PR-AUC one-vs-rest
    roc_aucs: dict[str, float] = {}
    pr_aucs: dict[str, float] = {}
    for c in SEVERITY_CLASSES:
        if c not in class_to_col:
            roc_aucs[c] = float("nan")
            pr_aucs[c] = float("nan")
            continue
        col = class_to_col[c]
        y_bin = (y_test == c).astype(int).values
        proba_c = y_proba[:, col]
        # ROC-AUC требует обоих классов в y_bin
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            roc_aucs[c] = float("nan")
            pr_aucs[c] = float("nan")
        else:
            roc_aucs[c] = float(roc_auc_score(y_bin, proba_c))
            pr_aucs[c] = float(average_precision_score(y_bin, proba_c))
    metrics["roc_auc_per_class"] = roc_aucs
    metrics["pr_auc_per_class"] = pr_aucs

    # Confusion matrix (rows = true, cols = predicted)
    cm = confusion_matrix(y_test, y_pred, labels=SEVERITY_CLASSES)
    metrics["confusion_matrix"] = cm.tolist()

    # Feature importance (Prediction values change)
    fi = model.get_feature_importance(prettified=True)
    metrics["feature_importance"] = [
        {"feature": str(row["Feature Id"]), "importance": float(row["Importances"])}
        for _, row in fi.iterrows()
    ]

    return metrics


def predict_proba_df(model: CatBoostClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Удобная обёртка: predict_proba → DataFrame с понятными колонками."""
    proba = model.predict_proba(X)
    return pd.DataFrame(
        proba,
        index=X.index,
        columns=[f"proba_{c}" for c in model.classes_],
    )


# =====================================================================
# 5. Save / load
# =====================================================================


def save_model(model: CatBoostClassifier, path: Path) -> None:
    """Сохраняет модель CatBoost через нативный .cbm-формат (быстрее чем pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    model.save_model(str(tmp))
    tmp.replace(path)
    logger.info("Модель сохранена в %s", path)


def load_model(path: Path) -> CatBoostClassifier:
    """Загрузка модели."""
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """JSON-дамп метрик через atomic temp+rename (UTF-8 принудительно)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("Метрики сохранены в %s", path)


def save_dataframe_pickle(df: pd.DataFrame, path: Path) -> None:
    """Сохранение feature matrix для повторного использования."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(df, f)
    tmp.replace(path)
    logger.info("DataFrame сохранён в %s", path)


# =====================================================================
# 6.1 — SHAP-интерпретация
# =====================================================================


def compute_shap(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    cat_features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Считает SHAP values через встроенный CatBoost-механизм.

    CatBoost умеет нативно отдавать SHAP через
    `get_feature_importance(type='ShapValues', data=Pool(...))`. Этот
    путь надёжнее `shap.TreeExplainer` (который ловит совместимость
    версий), и на multiclass он возвращает массив формы
    (n_samples, n_classes, n_features+1) — последняя колонка expected_value.

    Возвращает:
    - shap_values: (n_samples, n_classes, n_features) — без expected_value
    - expected_value: (n_classes,) — базовые значения per class
    - classes_order: имена классов в порядке shap_values[:, k, :]
    """
    pool = Pool(X, cat_features=cat_features)
    raw = model.get_feature_importance(type="ShapValues", data=pool)
    raw = np.asarray(raw)
    classes = list(model.classes_)
    # CatBoost shape: (n_samples, n_classes, n_features+1) для multiclass.
    # Для бинарной может быть (n_samples, n_features+1) — но у нас 4 класса.
    if raw.ndim == 3:
        shap_values = raw[:, :, :-1]
        expected_value = raw[:, :, -1].mean(axis=0)  # (n_classes,)
    elif raw.ndim == 2:
        # Бинарный случай — оборачиваем в 3D
        shap_values = raw[:, :-1][:, np.newaxis, :]
        expected_value = np.array([raw[:, -1].mean()])
    else:
        raise RuntimeError(f"Unexpected SHAP shape: {raw.shape}")
    logger.info(
        "SHAP: shape=%s, expected_value=%s, classes=%s",
        shap_values.shape,
        expected_value.tolist(),
        classes,
    )
    return shap_values, expected_value, classes


def shap_global_importance(
    shap_values: np.ndarray, feature_names: list[str], class_idx: int | None = None
) -> pd.DataFrame:
    """Глобальная важность по mean(|SHAP|).

    Если class_idx=None — усредняем по всем классам, получаем общий ranking.
    Если class_idx задан — ranking для конкретного класса.
    Возвращает DataFrame с колонками feature, mean_abs_shap, sorted desc.
    """
    if class_idx is None:
        # mean across both samples and classes
        importance = np.mean(np.abs(shap_values), axis=(0, 1))
    else:
        importance = np.mean(np.abs(shap_values[:, class_idx, :]), axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# =====================================================================
# 7.1 — Калибровка вероятностей (isotonic, OvR per class)
# =====================================================================


def compute_ece(
    y_true_bin: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error для бинарной (one-vs-rest) задачи.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|
    где acc_b — реальная доля положительных в bin b, conf_b — средняя
    предсказанная вероятность в bin b. Меньше — лучше; <0.05 хорошо
    откалибровано.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_proba, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    n_total = len(y_proba)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b = float(y_true_bin[mask].mean())
        conf_b = float(y_proba[mask].mean())
        ece += (n_b / n_total) * abs(acc_b - conf_b)
    return float(ece)


def reliability_curve(
    y_true_bin: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability curve: по бинам — средняя предсказанная вероятность,
    реальная частота положительных, размер бина.
    Возвращает (mean_pred_per_bin, frac_pos_per_bin, count_per_bin).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_proba, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            mean_pred[b] = float(y_proba[mask].mean())
            frac_pos[b] = float(y_true_bin[mask].mean())
            count[b] = int(mask.sum())
    return mean_pred, frac_pos, count


def make_calibration_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calib_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделяет TRAIN на refit_train + calib_holdout (стратифицированно).

    Это нужно чтобы калибровать вероятности на данных, которые модель
    НЕ видела (cv='prefit'-логика). Test остаётся нетронутым для финальной
    оценки.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=calib_size, random_state=random_state)
    refit_idx, calib_idx = next(sss.split(X_train, y_train))
    return (
        X_train.iloc[refit_idx],
        X_train.iloc[calib_idx],
        y_train.iloc[refit_idx],
        y_train.iloc[calib_idx],
    )


class IsotonicCalibratedClassifier:
    """Изотонная калибровка вероятностей (one-vs-rest, per class).

    Обертка вокруг базовой обученной CatBoost-модели: для каждого класса
    обучает sklearn.isotonic.IsotonicRegression на (proba_class_k,
    is_class_k) на calib-holdout, затем при predict_proba применяет их
    и нормализует строки в сумму 1 (защита от расхождения суммы).

    Применение: для production-вероятностей (диспетчер 112). НЕ заменяет
    исходную модель — это дополнение к ней.
    """

    def __init__(self, base_model: CatBoostClassifier):
        from sklearn.isotonic import IsotonicRegression

        self.base_model = base_model
        self.classes_ = list(base_model.classes_)
        self._calibrators: dict[str, IsotonicRegression] = {}

    def fit(self, X_calib: pd.DataFrame, y_calib: pd.Series) -> "IsotonicCalibratedClassifier":
        from sklearn.isotonic import IsotonicRegression

        proba = self.base_model.predict_proba(X_calib)
        y_arr = np.asarray(y_calib)
        for k, cls in enumerate(self.classes_):
            y_bin = (y_arr == cls).astype(int)
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(proba[:, k], y_bin)
            self._calibrators[cls] = ir
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.base_model.predict_proba(X)
        out = np.zeros_like(proba)
        for k, cls in enumerate(self.classes_):
            out[:, k] = self._calibrators[cls].predict(proba[:, k])
        # Нормализуем построчно (изотонная регрессия нарушает sum=1)
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        out = out / row_sum
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array([self.classes_[i] for i in np.argmax(proba, axis=1)])


# =====================================================================
# 8.1 — Error analysis
# =====================================================================


def find_top_errors(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_class: str,
    mode: str = "fp",
    k: int = 10,
) -> pd.DataFrame:
    """Возвращает топ-k самых уверенных ошибок модели для класса target_class.

    mode='fp': модель сказала target_class, реально другое → берём те, у
        которых proba[target_class] максимальна (модель уверена).
    mode='fn': реально target_class, модель сказала другое → берём те, у
        которых proba[target_class] минимальна (модель уверена что НЕ).
    Returns: DataFrame с колонками: accident_id (index), true, pred,
        proba_<target>, + ключевые признаки.
    """
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    if target_class not in classes:
        raise ValueError(f"{target_class} not in {classes}")
    k_idx = classes.index(target_class)
    pred = np.array([classes[i] for i in np.argmax(proba, axis=1)])
    y_arr = np.asarray(y_test)

    if mode == "fp":
        mask = (pred == target_class) & (y_arr != target_class)
        sort_proba = proba[:, k_idx]
        order = np.argsort(-sort_proba)  # desc
    elif mode == "fn":
        mask = (pred != target_class) & (y_arr == target_class)
        sort_proba = proba[:, k_idx]
        order = np.argsort(sort_proba)  # asc — модель уверена что НЕ
    else:
        raise ValueError(f"mode must be 'fp' or 'fn', got {mode!r}")

    # Применяем mask + сортировку: берём элементы по убыванию ord, оставляя только mask
    selected: list[int] = []
    for i in order:
        if mask[i]:
            selected.append(i)
            if len(selected) >= k:
                break

    if not selected:
        logger.warning("Нет ошибок типа %s для класса %s", mode, target_class)
        return pd.DataFrame()

    out = X_test.iloc[selected].copy()
    out.insert(0, "true", y_arr[selected])
    out.insert(1, "pred", pred[selected])
    out.insert(2, f"proba_{target_class}", proba[selected, k_idx])
    # Пробросим вероятности всех классов для контекста
    for j, c in enumerate(classes):
        if c != target_class:
            out[f"proba_{c}"] = proba[selected, j]
    return out


# =====================================================================
# 9.1 — Optuna-тюнинг
# =====================================================================


def tune_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    n_trials: int = 50,
    n_folds: int = 3,
    target_class: str = "dead",
    cv_iterations: int = 500,
    random_state: int = RANDOM_STATE,
    log_callback=None,
):
    """Optuna-тюнинг гиперпараметров CatBoost.

    Метрика: ROC-AUC класса target_class (one-vs-rest), усреднённый по
    CV-folds. Это бизнес-метрика для нашей задачи (диспетчер 112: «как
    хорошо модель отличает потенциально-смертельное ДТП от любого другого»).

    Пространство:
      learning_rate ∈ [0.01, 0.15] (log-uniform)
      depth ∈ [4, 10]
      l2_leaf_reg ∈ [1, 10] (log-uniform)
      bagging_temperature ∈ [0, 1]
      random_strength ∈ [1, 10]

    Внутри trial: StratifiedKFold(n_folds), на каждом fold — обучаем
    CatBoost(iterations=cv_iterations) с early stopping по eval-pool
    (fold-out часть), считаем proba и ROC-AUC dead.

    Возвращает (study, best_params).
    """
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.model_selection import StratifiedKFold

    classes_in_y = sorted(y_train.unique())
    if target_class not in classes_in_y:
        raise ValueError(f"target_class {target_class} not in y_train: {classes_in_y}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = list(skf.split(X_train, y_train))

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1.0, 10.0),
        }
        fold_aucs: list[float] = []
        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            train_pool = Pool(X_tr, label=y_tr.astype(str), cat_features=cat_features)
            eval_pool = Pool(X_va, label=y_va.astype(str), cat_features=cat_features)
            model = CatBoostClassifier(
                iterations=cv_iterations,
                learning_rate=params["learning_rate"],
                depth=params["depth"],
                l2_leaf_reg=params["l2_leaf_reg"],
                bagging_temperature=params["bagging_temperature"],
                random_strength=params["random_strength"],
                loss_function="MultiClass",
                eval_metric="TotalF1",
                cat_features=cat_features,
                auto_class_weights="Balanced",
                random_seed=random_state,
                od_type="Iter",
                od_wait=30,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
            proba = model.predict_proba(X_va)
            classes_model = list(model.classes_)
            k_idx = classes_model.index(target_class)
            y_bin = (y_va.values == target_class).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                fold_aucs.append(float("nan"))
            else:
                fold_aucs.append(float(roc_auc_score(y_bin, proba[:, k_idx])))
        mean_auc = float(np.nanmean(fold_aucs))
        if log_callback:
            log_callback(trial.number, params, fold_aucs, mean_auc)
        return mean_auc

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study, dict(study.best_params)
