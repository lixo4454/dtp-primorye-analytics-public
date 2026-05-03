"""
Оркестратор обучения Prophet-модели для прогноза ДТП в Приморье.

Что делает:
1. Загружает дневной временной ряд из БД (2015-01-01 .. 2026-12-31,
   с reindex и fillna(0) для пропусков).
2. Делит на train (2015-2024), hold-out (2025), forecast-пустоту (2026).
3. Декомпозиция statsmodels на месячных агрегатах — для отчёта.
4. Обучает Prophet с праздниками РФ + COVID-регрессором.
5. Прогноз на 731 день вперёд (2025+2026).
6. Sliding-window cross-validation (initial=5y, period=6mo, horizon=1y).
7. Считает метрики: hold-out 2025 (день+месяц), single-point Jan-2026.
8. Сохраняет модель в `models/prophet_dtp.pkl` и summary в
   `data/processed/forecast_summary.json`.

Зачем нужно: централизованный воспроизводимый pipeline. Повторный
запуск идемпотентно перезаписывает модель и summary (атомарный rename).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# Добавим корень проекта в sys.path, чтобы запускать как `python scripts/...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prophet  # noqa: E402

from src.database.session import get_session  # noqa: E402
from src.ml.forecasting import (  # noqa: E402
    FORECAST_HORIZON_DAYS,
    HOLDOUT_END,
    HOLDOUT_START,
    TRAIN_END,
    TRAIN_START,
    add_covid_regressor_column,
    aggregate_to_monthly,
    build_holidays_df,
    evaluate,
    forecast,
    prepare_timeseries,
    run_cross_validation,
    save_model,
    save_summary,
    train_prophet,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Заглушаем шумные cmdstanpy-логи
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logger = logging.getLogger("train_prophet")


MODEL_PATH = ROOT / "models" / "prophet_dtp.pkl"
SUMMARY_PATH = ROOT / "data" / "processed" / "forecast_summary.json"
DECOMPOSE_PATH = ROOT / "data" / "processed" / "decomposition_summary.json"


def main() -> None:
    started_at = datetime.now()
    logger.info("=" * 70)
    logger.info("Prophet-прогноз ДТП Приморского края. Старт.")
    logger.info("=" * 70)

    # ---- 1. Загрузка ряда ----
    with get_session() as session:
        df_full = prepare_timeseries(session)

    logger.info(
        "Полный ряд: %d дней, total y=%d (sanity: должно быть 29413 ДТП)",
        len(df_full),
        df_full["y"].sum(),
    )
    assert df_full["y"].sum() == 29413, "Total ДТП должно быть 29413 — КРИТИЧЕСКАЯ проверка"

    # ---- 2. Сплиты ----
    df_train = df_full[(df_full["ds"] >= TRAIN_START) & (df_full["ds"] <= TRAIN_END)].copy()
    df_holdout = df_full[(df_full["ds"] >= HOLDOUT_START) & (df_full["ds"] <= HOLDOUT_END)].copy()
    df_jan2026 = df_full[
        (df_full["ds"] >= pd.Timestamp("2026-01-01"))
        & (df_full["ds"] <= pd.Timestamp("2026-01-31"))
    ].copy()
    logger.info(
        "Train: %d дней (%s..%s), сумма=%d",
        len(df_train),
        df_train["ds"].min().date(),
        df_train["ds"].max().date(),
        df_train["y"].sum(),
    )
    logger.info(
        "Hold-out 2025: %d дней (%s..%s), сумма=%d",
        len(df_holdout),
        df_holdout["ds"].min().date(),
        df_holdout["ds"].max().date(),
        df_holdout["y"].sum(),
    )
    logger.info(
        "Январь 2026 (sanity-check): %d дней, сумма=%d",
        len(df_jan2026),
        df_jan2026["y"].sum(),
    )

    # ---- 3. Декомпозиция (месячная) ----
    monthly_train = aggregate_to_monthly(df_train, "y")
    decomp = seasonal_decompose(monthly_train.set_index("ds")["y"], model="additive", period=12)
    trend_first = float(decomp.trend.dropna().iloc[0])
    trend_last = float(decomp.trend.dropna().iloc[-1])
    seasonal_amplitude = float(decomp.seasonal.max() - decomp.seasonal.min())
    logger.info(
        "Декомпозиция (месячная, additive, period=12): trend %.0f → %.0f (Δ %+.1f%%), "
        "seasonal amplitude=%.0f",
        trend_first,
        trend_last,
        100 * (trend_last - trend_first) / trend_first,
        seasonal_amplitude,
    )

    decomp_summary = {
        "method": "statsmodels.seasonal_decompose(model='additive', period=12)",
        "input_frequency": "monthly_sum",
        "trend_first_value": trend_first,
        "trend_last_value": trend_last,
        "trend_change_pct": round(100 * (trend_last - trend_first) / trend_first, 2),
        "seasonal_amplitude": seasonal_amplitude,
        "seasonal_peak_month": int(
            decomp.seasonal.dropna().groupby(decomp.seasonal.dropna().index.month).mean().idxmax()
        ),
        "seasonal_low_month": int(
            decomp.seasonal.dropna().groupby(decomp.seasonal.dropna().index.month).mean().idxmin()
        ),
    }
    save_summary(decomp_summary, DECOMPOSE_PATH)

    # ---- 4. Праздники + COVID-регрессор ----
    holiday_years = list(range(TRAIN_START.year, 2027))  # 2015..2026
    holidays_df = build_holidays_df(holiday_years)
    df_train = add_covid_regressor_column(df_train)

    # ---- 5. Обучение ----
    logger.info("Обучение Prophet...")
    model = train_prophet(df_train, holidays_df=holidays_df, use_covid_regressor=True)

    # ---- 6. Прогноз: 731 день вперёд (2025 + 2026) ----
    periods = (pd.Timestamp("2026-12-31") - TRAIN_END).days  # 731
    logger.info("Прогноз на %d дней (2025-01-01 .. 2026-12-31)", periods)
    fcst = forecast(model, df_train, periods=periods, use_covid_regressor=True)

    # ---- 7. Метрики hold-out 2025 (день + месяц) ----
    fcst_holdout = fcst[(fcst["ds"] >= HOLDOUT_START) & (fcst["ds"] <= HOLDOUT_END)][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].reset_index(drop=True)
    holdout_metrics_daily = evaluate(df_holdout["y"].values, fcst_holdout["yhat"].values)

    # Месячные агрегаты
    holdout_actual_m = aggregate_to_monthly(df_holdout, "y").rename(columns={"y": "actual"})
    holdout_pred_m = (
        fcst_holdout.rename(columns={"yhat": "y"})
        .pipe(aggregate_to_monthly, "y")
        .rename(columns={"y": "predicted"})
    )
    holdout_monthly = holdout_actual_m.merge(holdout_pred_m, on="ds")
    holdout_metrics_monthly = evaluate(
        holdout_monthly["actual"].values, holdout_monthly["predicted"].values
    )
    logger.info("Hold-out 2025 daily: %s", holdout_metrics_daily)
    logger.info("Hold-out 2025 monthly: %s", holdout_metrics_monthly)
    logger.info(
        "2025: actual=%d, predicted=%.0f (отклонение %+.1f%%)",
        df_holdout["y"].sum(),
        fcst_holdout["yhat"].sum(),
        100 * (fcst_holdout["yhat"].sum() - df_holdout["y"].sum()) / df_holdout["y"].sum(),
    )

    # ---- 8. Single-point Jan 2026 ----
    fcst_jan26 = fcst[
        (fcst["ds"] >= pd.Timestamp("2026-01-01")) & (fcst["ds"] <= pd.Timestamp("2026-01-31"))
    ]
    jan2026_actual = int(df_jan2026["y"].sum())
    jan2026_predicted = float(fcst_jan26["yhat"].sum())
    jan2026_lower = float(fcst_jan26["yhat_lower"].sum())
    jan2026_upper = float(fcst_jan26["yhat_upper"].sum())
    jan2026_in_ci = jan2026_lower <= jan2026_actual <= jan2026_upper
    logger.info(
        "Январь 2026: actual=%d, predicted=%.0f (CI %.0f..%.0f), в CI: %s",
        jan2026_actual,
        jan2026_predicted,
        jan2026_lower,
        jan2026_upper,
        jan2026_in_ci,
    )

    # ---- 9. Прогноз 2026 целиком ----
    fcst_2026 = fcst[
        (fcst["ds"] >= pd.Timestamp("2026-01-01")) & (fcst["ds"] <= pd.Timestamp("2026-12-31"))
    ].copy()
    forecast_2026_total = float(fcst_2026["yhat"].sum())
    forecast_2026_lower = float(fcst_2026["yhat_lower"].sum())
    forecast_2026_upper = float(fcst_2026["yhat_upper"].sum())
    forecast_2026_monthly = (
        fcst_2026.rename(columns={"yhat": "y"})
        .pipe(aggregate_to_monthly, "y")
        .assign(month=lambda d: d["ds"].dt.strftime("%Y-%m"))
    )
    logger.info(
        "Прогноз 2026: total=%.0f ДТП (CI %.0f..%.0f)",
        forecast_2026_total,
        forecast_2026_lower,
        forecast_2026_upper,
    )

    # ---- 10. Cross-validation ----
    logger.info("Cross-validation (initial=5y, period=6mo, horizon=1y)...")
    cv_df, perf_df = run_cross_validation(
        model, initial_days=1825, period_days=180, horizon_days=365
    )
    n_folds = cv_df["cutoff"].nunique()
    logger.info("CV perf columns: %s", list(perf_df.columns))
    # `performance_metrics` не всегда возвращает 'mape' (зависит от наличия y=0
    # в окнах); используем 'mdape' (median absolute % error) или считаем сами.
    cv_mean_mae = float(perf_df["mae"].mean())
    cv_mean_rmse = float(perf_df["rmse"].mean())
    cv_mean_mape = (
        float(perf_df["mape"].mean()) * 100 if "mape" in perf_df.columns else float("nan")
    )
    cv_mean_mdape = (
        float(perf_df["mdape"].mean()) * 100 if "mdape" in perf_df.columns else float("nan")
    )
    logger.info(
        "CV: %d fold'ов, mean MAE=%.2f, mean RMSE=%.2f, MAPE=%.2f%%, MdAPE=%.2f%%",
        n_folds,
        cv_mean_mae,
        cv_mean_rmse,
        cv_mean_mape,
        cv_mean_mdape,
    )

    # ---- 11. Save model + summary ----
    save_model(model, MODEL_PATH)

    # COVID effect: коэффициент регрессора
    covid_effect = float(
        model.params["beta"][:, list(model.extra_regressors.keys()).index("covid_lockdown")].mean()
    )
    logger.info("COVID-регрессор beta: %.3f (отрицательный = падение ДТП в локдаун)", covid_effect)

    summary = {
        "trained_at": started_at.isoformat(),
        "duration_seconds": (datetime.now() - started_at).total_seconds(),
        "prophet_version": prophet.__version__,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        # Сплиты
        "train": {
            "start": TRAIN_START.isoformat(),
            "end": TRAIN_END.isoformat(),
            "n_days": int(len(df_train)),
            "total_y": int(df_train["y"].sum()),
        },
        "holdout_2025": {
            "start": HOLDOUT_START.isoformat(),
            "end": HOLDOUT_END.isoformat(),
            "n_days": int(len(df_holdout)),
            "actual_total": int(df_holdout["y"].sum()),
            "predicted_total": float(fcst_holdout["yhat"].sum()),
            "deviation_pct": round(
                100 * (fcst_holdout["yhat"].sum() - df_holdout["y"].sum()) / df_holdout["y"].sum(),
                2,
            ),
            "metrics_daily": holdout_metrics_daily,
            "metrics_monthly": holdout_metrics_monthly,
            "monthly_table": holdout_monthly.to_dict(orient="records"),
        },
        "jan_2026_sanity": {
            "actual": jan2026_actual,
            "predicted": round(jan2026_predicted, 1),
            "ci_lower": round(jan2026_lower, 1),
            "ci_upper": round(jan2026_upper, 1),
            "actual_in_ci": bool(jan2026_in_ci),
            "deviation_pct": round(100 * (jan2026_predicted - jan2026_actual) / jan2026_actual, 2),
        },
        "forecast_2026": {
            "total": round(forecast_2026_total, 0),
            "ci_lower": round(forecast_2026_lower, 0),
            "ci_upper": round(forecast_2026_upper, 0),
            "monthly": forecast_2026_monthly[["month", "y"]].to_dict(orient="records"),
        },
        "cross_validation": {
            "initial_days": 1825,
            "period_days": 180,
            "horizon_days": 365,
            "n_folds": int(n_folds),
            "mean_mae": round(cv_mean_mae, 2),
            "mean_rmse": round(cv_mean_rmse, 2),
            "mean_mape_pct": (round(cv_mean_mape, 2) if cv_mean_mape == cv_mean_mape else None),
            "mean_mdape_pct": (round(cv_mean_mdape, 2) if cv_mean_mdape == cv_mean_mdape else None),
            "perf_columns": list(perf_df.columns),
        },
        "regressors": {
            "covid_lockdown_beta": round(covid_effect, 3),
            "covid_lockdown_window": ["2020-03-15", "2020-05-31"],
        },
        "decomposition": decomp_summary,
    }
    save_summary(summary, SUMMARY_PATH)

    logger.info("=" * 70)
    logger.info("ГОТОВО. Длительность: %.1f сек", summary["duration_seconds"])
    logger.info("Модель: %s", MODEL_PATH)
    logger.info("Summary: %s", SUMMARY_PATH)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
