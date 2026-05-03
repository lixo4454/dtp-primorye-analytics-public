"""
Ежемесячный retrain Prophet.

Что делает:
1. Загружает дневной ряд ДТП из БД на момент запуска (динамические
   границы: train_end = последняя полная неделя в БД, holdout = 12
   последних месяцев перед train_end).
2. Sanity-check на >2σ всплеск в последний год — если найден, исключаем
   аномальные дни из train (защита от разовых событий, см. план7).
3. Тренирует Prophet с праздниками + COVID-регрессором.
4. Cross-validation (initial=5y, period=6mo, horizon=1y) → метрики.
5. Сохраняет в `models/prophet_dtp_<UTC_TS>.pkl`, atomic-replace alias
   `models/prophet_dtp.pkl`, регистрирует в model_versions.

Sanity gate: если cross_validation MAPE > 30% — НЕ переключаем
is_current и пишем error в task_runs (кто-то решит руками).

Idempotency: повторный запуск создаёт новый файл с другим timestamp'ом
и просто становится новой is_current версией.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.database.session import get_session
from src.ml.forecasting import (
    TRAIN_START,
    add_covid_regressor_column,
    build_holidays_df,
    evaluate,
    forecast,
    prepare_timeseries,
    run_cross_validation,
    save_model,
    train_prophet,
)
from src.tasks.model_registry import (
    MODELS_DIR,
    model_swap,
    timestamped_filename,
    try_reload_api,
)
from src.tasks.runner import logged_task

MODEL_NAME = "prophet_dtp"
ALIAS_PATH = MODELS_DIR / "prophet_dtp.pkl"

# Sanity-gate: cross-val MAPE свыше этого — блокируем swap
MAPE_BLOCKING_THRESHOLD = 30.0
# >2σ детектор всплеска (см. план7, риск "Re-train Prophet с свежими COVID")
ANOMALY_SIGMA_K = 2.0


def _detect_and_filter_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Маркирует дни с y > mean(year) + 2σ как аномалии.

    Возвращает (df_filtered, list_of_dropped_dates). Делается на
    последние 365 дней train'а — там где «свежие COVID-подобные» события.
    Старые годы оставляем как есть (Prophet справляется с устойчивыми
    historical-аномалиями через regressor + holiday).
    """
    if df.empty:
        return df, []

    last_year_start = df["ds"].max() - pd.Timedelta(days=365)
    mask_recent = df["ds"] >= last_year_start
    recent = df.loc[mask_recent, "y"]
    if len(recent) < 30:
        return df, []

    threshold = recent.mean() + ANOMALY_SIGMA_K * recent.std()
    anomaly_mask = mask_recent & (df["y"] > threshold)
    dropped_dates = [d.strftime("%Y-%m-%d") for d in df.loc[anomaly_mask, "ds"]]

    if not dropped_dates:
        return df, []

    logger.warning(
        f"[retrain_prophet] обнаружено {len(dropped_dates)} >2σ-аномалий "
        f"в последний год, threshold={threshold:.1f}, исключаем из train: {dropped_dates}"
    )
    df_filtered = df.loc[~anomaly_mask].copy()
    return df_filtered, dropped_dates


HOLDOUT_DAYS = 365


def _train_with_dynamic_split() -> tuple[Any, dict, int]:
    """Retrain Prophet с движущимся окном:
    - Train: TRAIN_START .. (NOW - HOLDOUT_DAYS)
    - Holdout: последние HOLDOUT_DAYS дней (для метрик)

    Это критично для monthly retrain'а: иначе train замораживается на
    статическом TRAIN_END=2024-12-31 из src.ml.forecasting и новые
    данные никогда не входят в обучающую выборку.
    """
    end_ts = pd.Timestamp.utcnow().tz_localize(None).normalize()
    holdout_start = end_ts - pd.Timedelta(days=HOLDOUT_DAYS)
    train_end = holdout_start - pd.Timedelta(days=1)

    with get_session() as s:
        df_full = prepare_timeseries(s, start=TRAIN_START, end=end_ts)

    total_y = int(df_full["y"].sum())
    logger.info(
        f"[retrain_prophet] full series: n_days={len(df_full)} total_y={total_y}, "
        f"train .. {train_end.date()}, holdout {holdout_start.date()} .. {end_ts.date()}"
    )

    df_train = df_full[(df_full["ds"] >= TRAIN_START) & (df_full["ds"] <= train_end)].copy()
    df_holdout = df_full[(df_full["ds"] >= holdout_start) & (df_full["ds"] <= end_ts)].copy()

    df_train, dropped = _detect_and_filter_anomalies(df_train)

    holiday_years = list(range(TRAIN_START.year, end_ts.year + 1))
    holidays_df = build_holidays_df(holiday_years)
    df_train = add_covid_regressor_column(df_train)

    model = train_prophet(df_train, holidays_df=holidays_df, use_covid_regressor=True)

    periods_to_holdout = (end_ts - df_train["ds"].max()).days + 1
    fcst = forecast(model, df_train, periods=periods_to_holdout, use_covid_regressor=True)

    fcst_holdout = fcst[(fcst["ds"] >= holdout_start) & (fcst["ds"] <= end_ts)][
        ["ds", "yhat"]
    ].reset_index(drop=True)
    if not fcst_holdout.empty and not df_holdout.empty:
        n = min(len(df_holdout), len(fcst_holdout))
        holdout_metrics = evaluate(df_holdout["y"].values[:n], fcst_holdout["yhat"].values[:n])
    else:
        holdout_metrics = {"mae": None, "rmse": None, "mape_pct": None}

    logger.info("[retrain_prophet] cross_validation...")
    _, perf_df = run_cross_validation(model, initial_days=1825, period_days=180, horizon_days=365)
    cv_mape = float(perf_df["mape"].mean()) * 100 if "mape" in perf_df.columns else float("nan")
    cv_mae = float(perf_df["mae"].mean())
    cv_rmse = float(perf_df["rmse"].mean())

    summary = {
        "trained_at_utc": end_ts.isoformat(),
        "train": {
            "start": TRAIN_START.isoformat(),
            "end": train_end.isoformat(),
            "n_days": int(len(df_train)),
            "total_y": int(df_train["y"].sum()),
        },
        "holdout": {
            "start": holdout_start.isoformat(),
            "end": end_ts.isoformat(),
            "n_days": int(len(df_holdout)),
            "metrics": holdout_metrics,
        },
        "cross_validation": {
            "n_folds": int(perf_df.shape[0]) if perf_df is not None else 0,
            "mape_pct": round(cv_mape, 2) if not np.isnan(cv_mape) else None,
            "mae": round(cv_mae, 2),
            "rmse": round(cv_rmse, 2),
        },
        "anomalies_dropped": dropped,
        "sigma_k_used": ANOMALY_SIGMA_K,
    }
    return model, summary, int(df_train["y"].sum())


@logged_task(name="src.tasks.retrain_prophet.retrain_prophet")
def retrain_prophet() -> dict[str, Any]:
    model, summary, train_total = _train_with_dynamic_split()

    cv_mape = summary["cross_validation"]["mape_pct"]
    if cv_mape is not None and cv_mape > MAPE_BLOCKING_THRESHOLD:
        # Сохраняем файл и регистрируем версию НЕ-current — для аудита
        # и возможного manual-rollout. Но alias не переключаем.
        version_path = timestamped_filename(MODEL_NAME, "pkl")
        save_model(model, version_path)
        from src.tasks.model_registry import register_version

        register_version(
            MODEL_NAME,
            version_path,
            metadata={
                **summary,
                "blocked_reason": f"MAPE {cv_mape:.1f}% > {MAPE_BLOCKING_THRESHOLD}%",
            },
            train_size=train_total,
            make_current=False,
        )
        raise RuntimeError(
            f"Prophet sanity-gate failed: cross-val MAPE = {cv_mape:.1f}% > "
            f"threshold {MAPE_BLOCKING_THRESHOLD}%. Версия сохранена, но НЕ активирована."
        )

    # OK — saving + atomic swap
    version_path = timestamped_filename(MODEL_NAME, "pkl")
    save_model(model, version_path)

    with model_swap(
        model_name=MODEL_NAME,
        version_path=version_path,
        alias_path=ALIAS_PATH,
        metadata=summary,
        train_size=train_total,
    ):
        try_reload_api(label="retrain_prophet")

    return {
        "version_path": str(version_path.relative_to(MODELS_DIR.parent)).replace("\\", "/"),
        "cv_mape_pct": cv_mape,
        "anomalies_dropped": summary["anomalies_dropped"],
        "train_size": train_total,
    }
