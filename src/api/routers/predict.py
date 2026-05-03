"""
Router /predict — CatBoost severity-классификатор + isotonic-калибровка.

Эндпоинты:
- POST /predict/severity — на вход 34 признака, на выход 4 калиброванные
  вероятности + предсказанный класс

Используем `IsotonicCalibratedClassifier` из1 — он оборачивает
CatBoost-модель и предоставляет `predict_proba` с per-class изотонной
калибровкой (ECE<0.05 для всех классов на test). Результат — production-grade
вероятности, годятся для бинарного thresholding (диспетчер 112: «вероятность
смертельного ≥ 0.4 → приоритет»).

Порядок и типы признаков валидируем против `ModelRegistry.catboost_feature_columns`
и `catboost_cat_features` — если CatBoost обучали с другим порядком, поднимется
ошибка ещё на этапе DataFrame-сборки, а не внутри C++.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import ModelRegistry, get_models
from src.api.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    CounterfactualResult,
    SeverityPredictRequest,
    SeverityPredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


# Bool-колонки CatBoost обучен трактовать как numeric (0/1), а не категориальные —
# см. severity_classifier.get_categorical_features.
_BOOL_COLS = {
    "is_weekend",
    "is_holiday",
    "is_highway",
    "is_in_region",
    "has_defect",
    "has_moto",
    "has_truck_or_bus",
    "has_known_age",
    "has_known_ped_age",
}


def _request_to_dataframe(
    req: SeverityPredictRequest,
    feature_columns: list[str],
    cat_features: list[str],
) -> pd.DataFrame:
    """Pydantic SeverityPredictRequest → 1-row DataFrame с правильными dtype.

    feature_columns — порядок и набор колонок, которые CatBoost ожидает
    (берём из catboost_features.pkl). Это защищает от тихого ошибочного
    скоринга если фронт передаст неполный набор.
    """
    payload = req.model_dump()
    missing = [c for c in feature_columns if c not in payload]
    if missing:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Не хватает признаков: {missing}",
        )

    row = {}
    for col in feature_columns:
        v = payload[col]
        if col in cat_features:
            # Категориальные строкой — fallback 'unknown' если None
            row[col] = "unknown" if v is None else str(v)
        elif col in _BOOL_COLS:
            # Bool как Python bool (CatBoost внутри сам приведёт к 0/1)
            row[col] = bool(v) if v is not None else False
        else:
            # Числовые — float; None → NaN (CatBoost понимает NaN)
            row[col] = float(v) if v is not None else np.nan

    df = pd.DataFrame([row], columns=feature_columns)
    # CatBoost категориальные требует object/string dtype
    for col in cat_features:
        df[col] = df[col].astype("object")
    return df


@router.post(
    "/severity",
    response_model=SeverityPredictResponse,
    summary="Предсказание тяжести ДТП (4-class CatBoost + isotonic)",
    responses={
        503: {"description": "Модель не загружена"},
        422: {"description": "Невалидные признаки (отсутствует колонка)"},
    },
)
async def predict_severity(
    payload: SeverityPredictRequest,
    models: Annotated[ModelRegistry, Depends(get_models)],
) -> SeverityPredictResponse:
    """Принимает 34 признака, возвращает калиброванные вероятности 4 классов.

    Калибровка через `IsotonicCalibratedClassifier` — снижает
    ECE для класса dead в 20 раз по сравнению с сырыми CatBoost-выходами.
    """
    if models.catboost_calibrated is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Calibrated CatBoost model not loaded",
        )
    if not models.catboost_feature_columns:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Feature schema (catboost_features.pkl) not loaded",
        )

    df = _request_to_dataframe(
        payload, models.catboost_feature_columns, models.catboost_cat_features
    )

    try:
        proba = models.catboost_calibrated.predict_proba(df)[0]  # (n_classes,)
    except Exception as e:
        logger.exception("predict_proba failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Calibrated predict error: {e}"
        ) from e

    classes = list(models.catboost_calibrated.classes_)
    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    predicted_class = max(probabilities, key=probabilities.get)

    return SeverityPredictResponse(
        predicted_class=predicted_class,  # type: ignore[arg-type]
        probabilities=probabilities,
        calibration_method="isotonic_per_class",
        model_version="catboost_severity_v2 + isotonic_v1",
    )


@router.post(
    "/severity_counterfactual",
    response_model=CounterfactualResponse,
    summary="Counterfactual: baseline + N сценариев → дельты вероятностей",
    responses={
        503: {"description": "Модель не загружена"},
        422: {"description": "Невалидный override (поле отсутствует в feature_columns)"},
    },
)
async def predict_severity_counterfactual(
    payload: CounterfactualRequest,
    models: Annotated[ModelRegistry, Depends(get_models)],
) -> CounterfactualResponse:
    """Per-instance counterfactual через калиброванный CatBoost.

    Один request → 1 baseline predict_proba + N modified-сценариев.
    Каждый сценарий это override-словарь по подмножеству признаков.
    Возвращает baseline_proba + per-scenario modified_proba и
    delta_proba (modified − baseline).

    Производительность: ~50 мс/предсказание × (1 + N) сценариев.
    7 сценариев ≈ 350 мс, приемлемо для UI-симуляции.

    Защита от out-of-distribution на стороне фронта (UI ограничивает
    inputs в p05..p95 train-выборки через numeric_ranges из
    catboost_form_schema.json).
    """
    if models.catboost_calibrated is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Calibrated CatBoost model not loaded",
        )
    if not models.catboost_feature_columns:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Feature schema not loaded",
        )

    feature_columns = models.catboost_feature_columns
    cat_features = models.catboost_cat_features

    # Baseline predict
    base_df = _request_to_dataframe(payload.baseline, feature_columns, cat_features)

    try:
        base_proba_raw = models.catboost_calibrated.predict_proba(base_df)[0]
    except Exception as e:
        logger.exception("baseline predict_proba failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Baseline predict error: {e}"
        ) from e

    classes = list(models.catboost_calibrated.classes_)
    baseline_proba = {cls: float(p) for cls, p in zip(classes, base_proba_raw)}
    baseline_predicted_class = max(baseline_proba, key=baseline_proba.get)

    # Per-scenario predicts
    valid_features = set(feature_columns)
    base_payload = payload.baseline.model_dump()

    scenario_results: list[CounterfactualResult] = []
    for sc in payload.scenarios:
        # Валидация: все ключи overrides должны быть в feature_columns
        unknown = [k for k in sc.overrides.keys() if k not in valid_features]
        if unknown:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Сценарий «{sc.name}»: неизвестные признаки {unknown}",
            )

        # Создаём модифицированный SeverityPredictRequest
        modified_payload = {**base_payload, **sc.overrides}
        try:
            mod_req = SeverityPredictRequest.model_validate(modified_payload)
        except Exception as e:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Сценарий «{sc.name}»: валидация провалена: {e}",
            ) from e

        mod_df = _request_to_dataframe(mod_req, feature_columns, cat_features)
        try:
            mod_proba_raw = models.catboost_calibrated.predict_proba(mod_df)[0]
        except Exception as e:
            logger.exception("scenario %s predict_proba failed", sc.name)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Scenario «{sc.name}» predict error: {e}",
            ) from e

        modified_proba = {cls: float(p) for cls, p in zip(classes, mod_proba_raw)}
        delta_proba = {cls: modified_proba[cls] - baseline_proba[cls] for cls in classes}
        # Конвенция: «dead» — самый интересный класс. Δ в п.п.
        delta_dead_pct_points = float(
            (modified_proba.get("dead", 0.0) - baseline_proba.get("dead", 0.0)) * 100.0
        )
        scenario_results.append(
            CounterfactualResult(
                name=sc.name,
                modified_proba=modified_proba,
                delta_proba=delta_proba,
                delta_dead_pct_points=delta_dead_pct_points,
            )
        )

    return CounterfactualResponse(
        baseline_proba=baseline_proba,
        baseline_predicted_class=baseline_predicted_class,  # type: ignore[arg-type]
        scenarios=scenario_results,
        model_version="catboost_severity_v2 + isotonic_v1",
    )
