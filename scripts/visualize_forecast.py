"""
Визуализация Prophet-прогноза ДТП в Приморье.

Что делает: загружает обученную модель из `models/prophet_dtp.pkl`,
перестраивает прогноз на 731 день вперёд (2025+2026), и сохраняет три PNG:

1. `data/processed/forecast_2026.png` — история (дневной ряд) + прогноз
   с 95% доверительными интервалами + вертикальная линия train/holdout.
   Bonus-панель — месячные агрегаты (более читаемо).

2. `data/processed/forecast_components.png` — декомпозиция Prophet
   (тренд / годовая сезонность / недельная сезонность / праздники /
   COVID-регрессор).

3. `data/processed/forecast_validation_2025.png` — hold-out: predicted
   vs actual на 2025 год, дневная и месячная панели + январь 2026
   single-point sanity-check.

Зачем нужно: ключевые артефакты для презентации и собеседования.
PNG'и идут в Git, поэтому фиксированный размер и DPI.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.database.session import get_session  # noqa: E402
from src.ml.forecasting import (  # noqa: E402
    HOLDOUT_END,
    HOLDOUT_START,
    TRAIN_END,
    TRAIN_START,
    add_covid_regressor_column,
    aggregate_to_monthly,
    forecast,
    load_model,
    prepare_timeseries,
)

matplotlib.use("Agg")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visualize_forecast")

MODEL_PATH = ROOT / "models" / "prophet_dtp.pkl"
OUT_DIR = ROOT / "data" / "processed"

# Цвета — TUE-палитра (синий + бордовый), ЧБ-friendly
COLOR_ACTUAL = "#1f3a93"  # тёмно-синий
COLOR_FORECAST = "#c0392b"  # бордовый
COLOR_CI = "#e67e22"  # оранжевый (для CI-полосы)


def main() -> None:
    logger.info("Загрузка модели: %s", MODEL_PATH)
    model = load_model(MODEL_PATH)

    # Перестроим прогноз на 731 день (2025 + 2026)
    with get_session() as session:
        df_full = prepare_timeseries(session)

    df_train = df_full[(df_full["ds"] >= TRAIN_START) & (df_full["ds"] <= TRAIN_END)].copy()
    df_train = add_covid_regressor_column(df_train)
    periods = (pd.Timestamp("2026-12-31") - TRAIN_END).days
    fcst = forecast(model, df_train, periods=periods, use_covid_regressor=True)

    # ===== 1. forecast_2026.png — дневная + месячная панели =====
    plot_forecast_overview(df_full, fcst)

    # ===== 2. forecast_components.png =====
    plot_components(model, fcst)

    # ===== 3. forecast_validation_2025.png =====
    plot_validation(df_full, fcst)

    logger.info("Все 3 графика сохранены в %s", OUT_DIR)


# =====================================================================
# 1. forecast_2026.png
# =====================================================================


def plot_forecast_overview(df_full: pd.DataFrame, fcst: pd.DataFrame) -> None:
    """История + прогноз. Две панели: дневная (вверху), месячная (внизу)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # ---- Дневная панель ----
    ax = axes[0]
    train = df_full[df_full["ds"] <= TRAIN_END]
    rest = df_full[df_full["ds"] > TRAIN_END]
    ax.plot(
        train["ds"],
        train["y"],
        color=COLOR_ACTUAL,
        alpha=0.35,
        linewidth=0.4,
        label="Train (2015-2024)",
    )
    ax.plot(
        rest["ds"],
        rest["y"],
        color="#27ae60",  # зелёный — факт после train
        alpha=0.5,
        linewidth=0.4,
        label="Hold-out 2025 + Январь 2026 (факт)",
    )
    fcst_future = fcst[fcst["ds"] > TRAIN_END]
    ax.plot(
        fcst_future["ds"],
        fcst_future["yhat"],
        color=COLOR_FORECAST,
        linewidth=1.2,
        label="Прогноз Prophet",
    )
    ax.fill_between(
        fcst_future["ds"],
        fcst_future["yhat_lower"],
        fcst_future["yhat_upper"],
        color=COLOR_FORECAST,
        alpha=0.15,
        label="95% доверительный интервал",
    )
    ax.axvline(TRAIN_END, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        TRAIN_END,
        ax.get_ylim()[1] * 0.95,
        " train | hold-out + forecast",
        fontsize=8,
        alpha=0.7,
    )
    ax.set_title(
        "Прогноз ДТП Приморья — дневная серия (2015-2026)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("ДТП в день")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.2)

    # ---- Месячная панель ----
    ax = axes[1]
    actual_m = aggregate_to_monthly(df_full, "y")
    fcst_m = (
        fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .set_index("ds")
        .resample("MS")
        .sum()
        .reset_index()
    )
    train_m = actual_m[actual_m["ds"] <= TRAIN_END]
    rest_m = actual_m[actual_m["ds"] > TRAIN_END]

    ax.plot(
        train_m["ds"],
        train_m["y"],
        color=COLOR_ACTUAL,
        marker="o",
        markersize=3,
        linewidth=1.0,
        label="Train (факт)",
    )
    ax.plot(
        rest_m["ds"],
        rest_m["y"],
        color="#27ae60",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="Hold-out 2025 + Янв 2026 (факт)",
    )
    fcst_m_future = fcst_m[fcst_m["ds"] > TRAIN_END]
    ax.plot(
        fcst_m_future["ds"],
        fcst_m_future["yhat"],
        color=COLOR_FORECAST,
        marker="s",
        markersize=3,
        linewidth=1.2,
        label="Прогноз Prophet (мес.)",
    )
    ax.fill_between(
        fcst_m_future["ds"],
        fcst_m_future["yhat_lower"],
        fcst_m_future["yhat_upper"],
        color=COLOR_FORECAST,
        alpha=0.15,
        label="95% CI",
    )
    ax.axvline(TRAIN_END, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(
        "Прогноз ДТП Приморья — месячные агрегаты",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("ДТП в месяц")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = OUT_DIR / "forecast_2026.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ %s", out)


# =====================================================================
# 2. forecast_components.png
# =====================================================================


def plot_components(model, fcst: pd.DataFrame) -> None:
    """Prophet plot_components: trend / yearly / weekly / holidays / regressor."""
    from prophet.plot import plot_components_plotly  # noqa: F401  (only matplotlib used)

    fig = model.plot_components(fcst, figsize=(11, 12))
    fig.suptitle(
        "Декомпозиция Prophet: тренд, сезонности, праздники, COVID",
        fontsize=12,
        fontweight="bold",
        y=1.005,
    )
    out = OUT_DIR / "forecast_components.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ %s", out)


# =====================================================================
# 3. forecast_validation_2025.png
# =====================================================================


def plot_validation(df_full: pd.DataFrame, fcst: pd.DataFrame) -> None:
    """Hold-out 2025 + январь 2026."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    df_h = df_full[(df_full["ds"] >= HOLDOUT_START) & (df_full["ds"] <= HOLDOUT_END)]
    fcst_h = fcst[(fcst["ds"] >= HOLDOUT_START) & (fcst["ds"] <= HOLDOUT_END)]

    # ---- Дневная hold-out ----
    ax = axes[0]
    ax.plot(df_h["ds"], df_h["y"], color=COLOR_ACTUAL, linewidth=0.7, alpha=0.9, label="Факт 2025")
    ax.plot(
        fcst_h["ds"],
        fcst_h["yhat"],
        color=COLOR_FORECAST,
        linewidth=1.0,
        label="Прогноз 2025",
    )
    ax.fill_between(
        fcst_h["ds"],
        fcst_h["yhat_lower"],
        fcst_h["yhat_upper"],
        color=COLOR_FORECAST,
        alpha=0.15,
        label="95% CI",
    )
    ax.set_title(
        f"Hold-out 2025: факт vs прогноз (дневная серия), "
        f"всего факт={int(df_h['y'].sum())}, прогноз={fcst_h['yhat'].sum():.0f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("ДТП в день")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ---- Месячная hold-out + январь 2026 ----
    ax = axes[1]
    actual_m = aggregate_to_monthly(df_full[df_full["ds"] >= HOLDOUT_START], "y")
    fcst_m = (
        fcst[fcst["ds"] >= HOLDOUT_START][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .set_index("ds")
        .resample("MS")
        .sum()
        .reset_index()
    )

    width = 12
    ax.bar(
        actual_m["ds"],
        actual_m["y"],
        width=width,
        color=COLOR_ACTUAL,
        alpha=0.7,
        label="Факт",
    )
    ax.errorbar(
        fcst_m["ds"],
        fcst_m["yhat"],
        yerr=[fcst_m["yhat"] - fcst_m["yhat_lower"], fcst_m["yhat_upper"] - fcst_m["yhat"]],
        fmt="s",
        color=COLOR_FORECAST,
        markersize=6,
        capsize=4,
        label="Прогноз ± 95% CI",
    )
    ax.axvline(
        pd.Timestamp("2026-01-01"),
        color="black",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )
    ax.text(
        pd.Timestamp("2026-01-01"),
        ax.get_ylim()[1] * 0.95,
        " 2026",
        fontsize=8,
        alpha=0.7,
    )
    ax.set_title(
        "Месячная валидация: hold-out 2025 + январь 2026 (sanity-check)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("ДТП в месяц")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2, axis="y")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    out = OUT_DIR / "forecast_validation_2025.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ %s", out)


if __name__ == "__main__":
    main()
