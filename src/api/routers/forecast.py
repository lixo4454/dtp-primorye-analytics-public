"""
Router /forecast — Prophet-прогноз количества ДТП.

Эндпоинты:
- GET /forecast/monthly?periods=12 — прогноз на N месяцев вперёд

Особенности:
- Prophet.predict() требует DataFrame с колонкой `ds` (datetime64) +
  опциональный регрессор `covid_lockdown` (binary, 1 в марте-мае 2020).
  Для будущих дат всегда 0.
- Месячная агрегация: суммируем yhat / yhat_lower / yhat_upper по месяцам
  через pd.Grouper(freq='MS').
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import ModelRegistry, get_db, get_models
from src.api.schemas import MonthlyForecastPoint, MonthlyForecastResponse

# Источник истины для COVID-окна — модуль обучения. Регрессор
# с именем `covid_lockdown` зашит в сериализованной модели, даты должны
# совпадать дословно. Дублировать константы здесь = риск тихого рассинхрона.
from src.ml.forecasting import add_covid_regressor_column

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["forecast"])


def _build_future_frame(start_date: pd.Timestamp, periods_days: int) -> pd.DataFrame:
    """Дневной DataFrame для Prophet.predict с регрессором covid_lockdown.

    Делегирует расстановку флага в `add_covid_regressor_column` из
    src.ml.forecasting — единый источник имени колонки и дат окна.
    """
    dates = pd.date_range(start=start_date, periods=periods_days, freq="D")
    df = pd.DataFrame({"ds": dates})
    return add_covid_regressor_column(df)


@router.get(
    "/monthly",
    response_model=MonthlyForecastResponse,
    summary="Прогноз количества ДТП по месяцам (Prophet)",
    responses={503: {"description": "Prophet-модель не загружена"}},
)
async def monthly_forecast(
    session: Annotated[AsyncSession, Depends(get_db)],
    models: Annotated[ModelRegistry, Depends(get_models)],
    periods: int = Query(12, ge=1, le=36, description="Количество будущих месяцев"),
    start_year: int = Query(2026, ge=2024, le=2030, description="Год начала прогноза"),
    start_month: int = Query(1, ge=1, le=12, description="Месяц начала прогноза (1..12)"),
) -> MonthlyForecastResponse:
    """Месячный прогноз через предзагруженный Prophet.

    Алгоритм:
    1. Строим дневной DataFrame с датами от start_year-start_month +periods месяцев
    2. Prophet.predict — получаем дневные yhat / yhat_lower / yhat_upper
    3. Группируем по месяцам, суммируем
    4. Подмешиваем actual из БД для месяцев, где данные уже есть
    """
    if models.prophet is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Prophet model not loaded")

    start = pd.Timestamp(year=start_year, month=start_month, day=1)
    # Выходной горизонт: с 1-го числа start до последнего дня месяца (start + periods - 1)
    end = (start + pd.DateOffset(months=periods)) - pd.Timedelta(days=1)
    days_count = (end - start).days + 1
    future_df = _build_future_frame(start, days_count)

    try:
        forecast_df = models.prophet.predict(future_df)
    except Exception as e:
        logger.exception("Prophet.predict failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Prophet predict error: {e}"
        ) from e

    forecast_df = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # Месячная агрегация — сумма по месяцам
    monthly = (
        forecast_df.groupby(pd.Grouper(key="ds", freq="MS"))[["yhat", "yhat_lower", "yhat_upper"]]
        .sum()
        .reset_index()
    )

    # Actual из БД — за месяцы, для которых данные уже есть
    actual_sql = text(
        """
        SELECT DATE_TRUNC('month', datetime) AS m, COUNT(*)::int AS y
        FROM accidents
        WHERE datetime >= :start AND datetime <= :end
        GROUP BY m
        ORDER BY m
        """
    )
    actual_rows = (await session.execute(actual_sql, {"start": start, "end": end})).all()
    actual_map: dict[pd.Timestamp, int] = {pd.Timestamp(r.m): int(r.y) for r in actual_rows}

    items = [
        MonthlyForecastPoint(
            ds=row.ds.to_pydatetime(),
            yhat=float(row.yhat),
            yhat_lower=float(row.yhat_lower),
            yhat_upper=float(row.yhat_upper),
            actual=actual_map.get(pd.Timestamp(row.ds)),
        )
        for row in monthly.itertuples(index=False)
    ]

    metadata: dict[str, Any] = {
        "model_path": "models/prophet_dtp.pkl",
        "horizon_days": days_count,
        "start": str(start.date()),
        "end": str(end.date()),
    }
    # Добавляем метрики hold-out 2025 из forecast_summary.json (если есть)
    fs = models.forecast_summary
    if fs:
        metadata["train_range"] = {
            "start": fs.get("train", {}).get("start"),
            "end": fs.get("train", {}).get("end"),
        }
        metadata["holdout_2025_metrics"] = fs.get("holdout_2025", {}).get("metrics_monthly", {})

    return MonthlyForecastResponse(
        horizon_months=len(items),
        items=items,
        metadata=metadata,
    )
