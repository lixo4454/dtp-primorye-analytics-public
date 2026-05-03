"""
Prophet-прогноз количества ДТП по дням.

Что делает: строит дневной временной ряд count(accidents) по дате из
Postgres, обучает Prophet с российскими праздниками и binary-регрессором
для COVID-локдауна (март-май 2020), делает прогноз с доверительными
интервалами и cross-validation.

Зачем нужно: Prophet даёт интерпретируемый прогноз с разложением на
тренд / годовую сезонность / недельную сезонность / эффект праздников —
это и production-практик (предупреждение нагрузки на ГИБДД и скорую),
и сильный нарратив для собеседования (vs ARIMA Prophet проще, robust к
пропускам и outlier'ам, явно моделирует праздники).

Архитектурные решения:
- Дневная частота — weekly seasonality и holidays работают точно.
  Месячный прогноз получаем re-aggregation `forecast` по `pd.Grouper('M')`.
- Train: 2015-01-01 .. 2024-12-31 (10 полных лет).
  Hold-out: 2025-01-01 .. 2025-12-31 (12 полных месяцев) для оценки.
  Forecast: 2026-01-01 .. 2026-12-31 (на будущее, январь 2026 в БД
  есть как single-point sanity-check).
- COVID-локдаун (2020-03-15 .. 2020-05-31) — `add_regressor` с binary
  indicator. Точнее, чем выкидывать 2020 из train: данные остаются,
  но модель «знает» что это extreme event и не учит как сезонность.
- Reindex дат с `fillna(0)` для 6 пропущенных дней (нет ДТП — 0,
  не «пропуск»).
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.make_holidays import make_holidays_df
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# =====================================================================
# Константы периодов (centralized — легко скорректировать)
# =====================================================================

TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END = pd.Timestamp("2025-12-31")
FORECAST_HORIZON_DAYS = 365  # 2026 целиком

# COVID-локдаун в России: первый указ Путина о нерабочих днях — 25 марта 2020,
# самоизоляция в Приморье началась 31 марта 2020, активные ограничения
# до конца мая. Берём окно 2020-03-15 .. 2020-05-31 (учёт упреждающего
# падения активности и постепенного восстановления).
COVID_LOCKDOWN_START = pd.Timestamp("2020-03-15")
COVID_LOCKDOWN_END = pd.Timestamp("2020-05-31")


# =====================================================================
# 1. Подготовка временного ряда
# =====================================================================


def prepare_timeseries(
    session: Session,
    start: pd.Timestamp = TRAIN_START,
    end: pd.Timestamp = pd.Timestamp("2026-12-31"),
    is_in_region_only: bool = False,
) -> pd.DataFrame:
    """Загружает дневной временной ряд count(accidents).

    Returns
    -------
    DataFrame с колонками [`ds` datetime, `y` int]. Все даты от start до end
    включительно (reindex с fillna(0) для дней без ДТП).
    """
    where_extra = "AND is_in_region = TRUE" if is_in_region_only else ""
    sql = text(
        f"""
        SELECT DATE(datetime) AS ds, COUNT(*)::int AS y
        FROM accidents
        WHERE datetime >= :start AND datetime <= :end
        {where_extra}
        GROUP BY DATE(datetime)
        ORDER BY ds
        """
    )
    rows = session.execute(sql, {"start": start, "end": end}).all()
    df = pd.DataFrame(rows, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"])

    # Reindex полным диапазоном дней — заполняем пропуски нулями
    full_range = pd.date_range(start=start, end=end, freq="D")
    df = df.set_index("ds").reindex(full_range, fill_value=0).rename_axis("ds").reset_index()
    df["y"] = df["y"].astype(int)
    logger.info(
        "prepare_timeseries: %d дней (%s .. %s), сумма y=%d, дней с y=0: %d",
        len(df),
        df["ds"].min().date(),
        df["ds"].max().date(),
        df["y"].sum(),
        (df["y"] == 0).sum(),
    )
    return df


# =====================================================================
# 2. Праздники РФ
# =====================================================================


def build_holidays_df(years: list[int]) -> pd.DataFrame:
    """Возвращает DataFrame российских праздников для Prophet.

    Источник — встроенные holidays Prophet (`country_list=['RU']`).
    Включает: Новогодние каникулы, 23 февраля, 8 марта, 1-9 мая, 12 июня,
    4 ноября.
    """
    df = make_holidays_df(year_list=years, country="RU")
    logger.info("build_holidays_df: %d праздников за %d лет", len(df), len(years))
    return df


# =====================================================================
# 3. COVID-регрессор
# =====================================================================


def add_covid_regressor_column(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку `covid_lockdown` (0/1) к DataFrame с `ds`.

    1 = дата попадает в окно локдауна 2020-03-15 .. 2020-05-31.
    """
    df = df.copy()
    df["covid_lockdown"] = (
        (df["ds"] >= COVID_LOCKDOWN_START) & (df["ds"] <= COVID_LOCKDOWN_END)
    ).astype(int)
    return df


# =====================================================================
# 4. Обучение Prophet
# =====================================================================


def train_prophet(
    df_train: pd.DataFrame,
    holidays_df: Optional[pd.DataFrame] = None,
    use_covid_regressor: bool = True,
) -> Prophet:
    """Обучает Prophet с holidays и опциональным COVID-регрессором.

    Включены: yearly_seasonality, weekly_seasonality, daily_seasonality=False
    (дневная частота — внутри-дня динамики не моделируем).

    Параметры:
    - changepoint_prior_scale=0.05 (default — гибкость тренда)
    - seasonality_mode='additive' (default — сезонность пропорциональна
      уровню; multiplicative дал бы 'процент', но при count'ах additive
      обычно стабильнее).
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )

    if use_covid_regressor:
        if "covid_lockdown" not in df_train.columns:
            df_train = add_covid_regressor_column(df_train)
        model.add_regressor("covid_lockdown", mode="additive")

    logger.info("train_prophet: fit на %d точках", len(df_train))
    model.fit(df_train)
    return model


# =====================================================================
# 5. Прогноз
# =====================================================================


def make_future_dataframe(
    model: Prophet,
    df_train: pd.DataFrame,
    periods: int,
    use_covid_regressor: bool = True,
) -> pd.DataFrame:
    """Строит future DataFrame с регрессорами для прогноза.

    `make_future_dataframe` Prophet'а возвращает только `ds` — нужно
    самим заполнить `covid_lockdown` (всё в будущем = 0).
    """
    future = model.make_future_dataframe(periods=periods, freq="D")
    if use_covid_regressor:
        future = add_covid_regressor_column(future)
    return future


def forecast(
    model: Prophet,
    df_train: pd.DataFrame,
    periods: int,
    use_covid_regressor: bool = True,
) -> pd.DataFrame:
    """Делает прогноз на `periods` дней вперёд.

    Returns
    -------
    Полный forecast DataFrame Prophet — ds, yhat, yhat_lower, yhat_upper,
    trend, yearly, weekly, holidays и т.д.
    """
    future = make_future_dataframe(model, df_train, periods, use_covid_regressor)
    fcst = model.predict(future)
    logger.info(
        "forecast: %d точек (%s .. %s)",
        len(fcst),
        fcst["ds"].min().date(),
        fcst["ds"].max().date(),
    )
    return fcst


# =====================================================================
# 6. Метрики качества
# =====================================================================


def evaluate(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Считает MAE, RMSE, MAPE между actual и predicted.

    Все три метрики на одинаковом index'е.

    MAPE считаем при `actual > 0` (защита от деления на 0 в дни без ДТП).
    """
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    err = actual - predicted

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    nonzero = actual > 0
    if nonzero.any():
        mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100)
    else:
        mape = float("nan")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 4),
        "n_points": int(len(actual)),
    }


def aggregate_to_monthly(df: pd.DataFrame, value_col: str = "y") -> pd.DataFrame:
    """Aggregate дневной DataFrame в месячный (sum по месяцам)."""
    out = (
        df.set_index("ds")[[value_col]]
        .resample("MS")  # начало месяца
        .sum()
        .reset_index()
    )
    return out


# =====================================================================
# 7. Cross-validation
# =====================================================================


def run_cross_validation(
    model: Prophet,
    initial_days: int = 1825,  # 5 лет
    period_days: int = 180,  # шаг между fold'ами — 6 мес
    horizon_days: int = 365,  # длина каждого прогноза — 1 год
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prophet diagnostic CV — sliding window.

    initial=5 лет, period=6 мес, horizon=1 год даёт ~10-11 fold'ов
    на train 2015-2024.

    Returns
    -------
    (cv_df, perf_df) — cross_validation результаты + performance_metrics
    с MAE/RMSE/MAPE по горизонту.
    """
    logger.info(
        "run_cross_validation: initial=%dd, period=%dd, horizon=%dd",
        initial_days,
        period_days,
        horizon_days,
    )
    cv_df = cross_validation(
        model,
        initial=f"{initial_days} days",
        period=f"{period_days} days",
        horizon=f"{horizon_days} days",
        parallel=None,  # на Windows parallel может ломать — оставляем sequential
        disable_tqdm=True,
    )
    perf_df = performance_metrics(cv_df, rolling_window=1.0)
    return cv_df, perf_df


# =====================================================================
# 8. Сохранение / загрузка модели
# =====================================================================


def save_model(model: Prophet, path: Path | str) -> None:
    """Pickle Prophet модели.

    Идемпотентно: pickle файл перезаписывается атомарно через временный
    файл + rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(model, f)
    tmp.replace(path)
    logger.info("save_model: %s (%.1f MB)", path, path.stat().st_size / 1024 / 1024)


def load_model(path: Path | str) -> Prophet:
    """Загружает pickle Prophet."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_summary(summary: dict, path: Path | str) -> None:
    """Сохраняет JSON-summary с метриками и метаданными."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("save_summary: %s", path)


def _json_default(obj):
    """JSON encoder для timestamp/numpy-типов."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
