# -*- coding: utf-8 -*-
"""
Визуализация результатов анализа RHD vs LHD.

Создаёт график 2x2 с 4 этапами анализа:
- Этап 1: Базовое сравнение
- Этап 2: По категории ТС
- Этап 3: По возрасту ТС
- Этап 4: По типу местности

Сохраняет в data/processed/rhd_lhd_severity_analysis.png

Запуск:
    python -m src.analysis.rhd_lhd_visualize
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database import SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rhd_lhd_severity_analysis.png"

WHERE_CLASSIFIED = (
    "v.is_right_hand_drive IS NOT NULL " "AND v.steering_confidence IN ('high', 'medium')"
)

# Цвета: RHD = синий (как Япония), LHD = красный (как Европа)
COLOR_RHD = "#1f77b4"
COLOR_LHD = "#d62728"


def fetch_dataframe(query: str) -> pd.DataFrame:
    with SessionLocal() as session:
        result = session.execute(text(query))
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=result.keys())


def calculate_dead_pct(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Группирует по (group_col, steering, severity) и считает долю смертельных.
    Возвращает DataFrame с колонками: group_col, steering, total, dead, dead_pct.
    """
    pivot = df.pivot_table(
        index=[group_col, "steering"], columns="severity", values="cnt", fill_value=0
    ).reset_index()
    severity_cols = [c for c in pivot.columns if c not in [group_col, "steering"]]
    pivot["total"] = pivot[severity_cols].sum(axis=1)
    pivot["dead"] = pivot.get("dead", 0)
    pivot["dead_pct"] = pivot["dead"] / pivot["total"] * 100
    return pivot[[group_col, "steering", "total", "dead", "dead_pct"]]


def plot_grouped_bars(
    ax,
    df: pd.DataFrame,
    group_col: str,
    group_order: list,
    title: str,
    ylabel: str = "Доля смертельных, %",
) -> None:
    """Универсальная функция: рисует сгруппированные bar-чарты RHD vs LHD."""
    x = np.arange(len(group_order))
    width = 0.38

    rhd_vals = []
    lhd_vals = []
    rhd_n = []
    lhd_n = []

    for grp in group_order:
        rhd_row = df[(df[group_col] == grp) & (df["steering"] == "RHD")]
        lhd_row = df[(df[group_col] == grp) & (df["steering"] == "LHD")]
        rhd_vals.append(rhd_row["dead_pct"].iloc[0] if not rhd_row.empty else 0)
        lhd_vals.append(lhd_row["dead_pct"].iloc[0] if not lhd_row.empty else 0)
        rhd_n.append(int(rhd_row["total"].iloc[0]) if not rhd_row.empty else 0)
        lhd_n.append(int(lhd_row["total"].iloc[0]) if not lhd_row.empty else 0)

    bars1 = ax.bar(
        x - width / 2,
        rhd_vals,
        width,
        label="RHD (правый руль)",
        color=COLOR_RHD,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        lhd_vals,
        width,
        label="LHD (левый руль)",
        color=COLOR_LHD,
        edgecolor="black",
        linewidth=0.5,
    )

    # Значения над столбцами
    for bar, val, n in zip(bars1, rhd_vals, rhd_n):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.3,
            f"{val:.1f}%\n(n={n})",
            ha="center",
            fontsize=8,
        )
    for bar, val, n in zip(bars2, lhd_vals, lhd_n):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.3,
            f"{val:.1f}%\n(n={n})",
            ha="center",
            fontsize=8,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(max(rhd_vals), max(lhd_vals)) * 1.30)


def main() -> None:
    logger.info("Подготовка данных для визуализации...")

    # ---------- Запросы для всех 4 этапов ----------
    q_global = f"""
    SELECT
      CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
      a.severity, COUNT(*) AS cnt
    FROM vehicles v JOIN accidents a ON a.id = v.accident_id
    WHERE {WHERE_CLASSIFIED}
    GROUP BY steering, a.severity
    """
    df_global = fetch_dataframe(q_global)
    df_global["group"] = "Всего"

    q_prod = f"""
    SELECT
      CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
      CASE
        WHEN LOWER(v.prod_type) LIKE '%класс%' OR LOWER(v.prod_type) LIKE '%легков%'
          OR LOWER(v.prod_type) LIKE '%минивэн%' THEN 'Легковые'
        WHEN LOWER(v.prod_type) LIKE '%грузов%' OR LOWER(v.prod_type) LIKE '%тягач%'
          OR LOWER(v.prod_type) LIKE '%фургон%' THEN 'Грузовые'
        WHEN LOWER(v.prod_type) LIKE '%мото%' OR LOWER(v.prod_type) LIKE '%мопед%'
          THEN 'Мото'
        ELSE NULL
      END AS category,
      a.severity, COUNT(*) AS cnt
    FROM vehicles v JOIN accidents a ON a.id = v.accident_id
    WHERE {WHERE_CLASSIFIED} AND v.prod_type IS NOT NULL
    GROUP BY steering, category, a.severity
    HAVING CASE
        WHEN LOWER(v.prod_type) LIKE '%класс%' OR LOWER(v.prod_type) LIKE '%легков%'
          OR LOWER(v.prod_type) LIKE '%минивэн%' THEN 'Легковые'
        WHEN LOWER(v.prod_type) LIKE '%грузов%' OR LOWER(v.prod_type) LIKE '%тягач%'
          OR LOWER(v.prod_type) LIKE '%фургон%' THEN 'Грузовые'
        WHEN LOWER(v.prod_type) LIKE '%мото%' OR LOWER(v.prod_type) LIKE '%мопед%'
          THEN 'Мото'
        ELSE NULL
      END IS NOT NULL
    """
    df_prod = fetch_dataframe(q_prod)

    q_age = f"""
    SELECT
      CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
      CASE
        WHEN v.vehicle_year IS NULL THEN NULL
        WHEN v.vehicle_year < 2000 THEN '<2000'
        WHEN v.vehicle_year < 2010 THEN '2000-09'
        WHEN v.vehicle_year < 2020 THEN '2010-19'
        ELSE '2020+'
      END AS age_group,
      a.severity, COUNT(*) AS cnt
    FROM vehicles v JOIN accidents a ON a.id = v.accident_id
    WHERE {WHERE_CLASSIFIED} AND v.vehicle_year IS NOT NULL
    GROUP BY steering, age_group, a.severity
    """
    df_age = fetch_dataframe(q_age)

    q_loc = f"""
    SELECT
      CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
      CASE
        WHEN a.place IN ('Первореченский', 'Ленинский', 'Советский',
                          'Первомайский', 'Фрунзенский') THEN 'Владивосток'
        WHEN a.place IN ('Уссурийский ГО', 'Находкинский ГО', 'Артемовский ГО')
          THEN 'Крупн.города'
        WHEN a.place LIKE '%ГО' THEN 'Малые ГО'
        WHEN a.place LIKE '%МР' OR a.place LIKE '%МО' OR a.place LIKE '%РО'
          THEN 'Сельские'
        ELSE 'Прочее'
      END AS loc,
      a.severity, COUNT(*) AS cnt
    FROM vehicles v JOIN accidents a ON a.id = v.accident_id
    WHERE {WHERE_CLASSIFIED}
    GROUP BY steering, loc, a.severity
    """
    df_loc = fetch_dataframe(q_loc)

    # ---------- Преобразуем в dead_pct таблицы ----------
    s1 = calculate_dead_pct(df_global, "group")
    s2 = calculate_dead_pct(df_prod, "category")
    s3 = calculate_dead_pct(df_age, "age_group")
    s4 = calculate_dead_pct(df_loc, "loc")

    logger.info("Рисуем график 2x2...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Анализ тяжести ДТП: правый vs левый руль\n"
        "Приморский край, 2015-2026, 41362 ТС в 29413 ДТП",
        fontsize=14,
        fontweight="bold",
    )

    plot_grouped_bars(
        axes[0, 0],
        s1,
        "group",
        ["Всего"],
        "Этап 1. Базовое сравнение (без контроля)",
    )
    plot_grouped_bars(
        axes[0, 1],
        s2,
        "category",
        ["Легковые", "Грузовые", "Мото"],
        "Этап 2. С контролем на категорию ТС",
    )
    plot_grouped_bars(
        axes[1, 0],
        s3,
        "age_group",
        ["<2000", "2000-09", "2010-19", "2020+"],
        "Этап 3. С контролем на возраст ТС",
    )
    plot_grouped_bars(
        axes[1, 1],
        s4,
        "loc",
        ["Владивосток", "Крупн.города", "Малые ГО", "Сельские"],
        "Этап 4. С контролем на тип местности",
    )

    fig.text(
        0.5,
        0.01,
        "Гипотеза МВД о повышенной опасности правого руля (4.9x) на данных "
        "Приморского края не подтверждается.\n"
        "Во всех контролях RHD показывает долю смертельных ДТП ниже LHD.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#444",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.success(f"График сохранён: {OUTPUT_PATH}")
    logger.info(f"Размер файла: {OUTPUT_PATH.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
