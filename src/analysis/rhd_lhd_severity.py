# -*- coding: utf-8 -*-
"""
Анализ тяжести ДТП в зависимости от типа руля (правый/левый).

Цель: проверить гипотезу МВД о том, что праворульные ТС опаснее.
Для каждой группы (RHD/LHD) считаем долю смертельных ДТП и тяжёлых
случаев с контролем на ключевые переменные.

Запуск:
    python -m src.analysis.rhd_lhd_severity
"""

from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database import SessionLocal

# ВАЖНО: для анализа берём ТОЛЬКО машины с явной классификацией руля.
# - high и medium confidence включаем
# - unknown (нет в справочнике) и not_applicable (велосипеды/прицепы) ИСКЛЮЧАЕМ
WHERE_CLASSIFIED = (
    "v.is_right_hand_drive IS NOT NULL " "AND v.steering_confidence IN ('high', 'medium')"
)


def fetch_dataframe(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Хелпер: выполнить SQL и вернуть pandas DataFrame."""
    with SessionLocal() as session:
        result = session.execute(text(query), params or {})
        rows = result.fetchall()
        cols = result.keys()
        return pd.DataFrame(rows, columns=cols)


def section_header(title: str) -> None:
    """Красивый разделитель в логах."""
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)


def stage_1_global_picture() -> None:
    """
    Базовая картина БЕЗ контроля переменных.
    Просто: тяжесть ДТП у RHD vs LHD машин.
    """
    section_header("ЭТАП 1. БАЗОВОЕ СРАВНЕНИЕ RHD vs LHD (без контроля)")

    query = f"""
    SELECT
      CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
      a.severity,
      COUNT(*) AS cnt
    FROM vehicles v
    JOIN accidents a ON a.id = v.accident_id
    WHERE {WHERE_CLASSIFIED}
    GROUP BY steering, a.severity
    ORDER BY steering, a.severity
    """
    df = fetch_dataframe(query)

    # Превращаем в таблицу: строки = steering, колонки = severity
    pivot = df.pivot_table(index="steering", columns="severity", values="cnt", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)

    # Считаем доли (%)
    severity_cols = [c for c in pivot.columns if c != "total"]
    for col in severity_cols:
        pivot[f"{col}_pct"] = pivot[col] / pivot["total"] * 100

    logger.info("Распределение тяжести ДТП по типу руля:")
    logger.info(f"\n{pivot.to_string(float_format=lambda x: f'{x:.2f}')}")

    # Главный результат: доля смертельных
    if "dead" in pivot.columns:
        rhd_dead_pct = pivot.loc["RHD", "dead"] / pivot.loc["RHD", "total"] * 100
        lhd_dead_pct = pivot.loc["LHD", "dead"] / pivot.loc["LHD", "total"] * 100
        ratio = rhd_dead_pct / lhd_dead_pct if lhd_dead_pct > 0 else float("inf")

        logger.info("")
        logger.info("ГЛАВНЫЙ РЕЗУЛЬТАТ (доля смертельных ДТП):")
        logger.info(f"  RHD: {rhd_dead_pct:.2f}%  (всего {pivot.loc['RHD', 'total']} ДТП)")
        logger.info(f"  LHD: {lhd_dead_pct:.2f}%  (всего {pivot.loc['LHD', 'total']} ДТП)")
        logger.info(f"  Соотношение RHD/LHD: {ratio:.2f}x")
        logger.info("")
        logger.info("Сравнение с гипотезой МВД (RHD в 4.9 раза опаснее):")
        if ratio < 1:
            logger.info(
                f"  ❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДЕНА: RHD на {(1-ratio)*100:.1f}% БЕЗОПАСНЕЕ LHD"
            )
        elif ratio < 1.5:
            logger.info(
                f"  ⚠️  Различие незначительное: RHD в {ratio:.2f}x опаснее LHD "
                f"(не 4.9x как утверждает МВД)"
            )
        else:
            logger.info(f"  ⚠️  RHD действительно опаснее в {ratio:.2f}x, но не в 4.9x")


def stage_2_by_prod_type() -> None:
    """
    Контроль на категорию ТС (prod_type).
    Чтобы не сравнивать грузовик-LHD с легковушкой-RHD.
    """
    section_header("ЭТАП 2. С КОНТРОЛЕМ НА КАТЕГОРИЮ ТС")

    query = f"""
    SELECT
      CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
      v.prod_type,
      a.severity,
      COUNT(*) AS cnt
    FROM vehicles v
    JOIN accidents a ON a.id = v.accident_id
    WHERE {WHERE_CLASSIFIED}
      AND v.prod_type IS NOT NULL
    GROUP BY steering, v.prod_type, a.severity
    """
    df = fetch_dataframe(query)

    # Группируем категории в укрупнённые группы для читаемости
    def categorize(prod: str) -> str:
        if prod is None:
            return "Прочее"
        p = prod.lower()
        if "класс" in p or "легков" in p or "минивэн" in p:
            return "Легковые"
        if "грузов" in p or "тягач" in p or "фургон" in p:
            return "Грузовые"
        if "мото" in p or "мопед" in p:
            return "Мото"
        if "автобус" in p:
            return "Автобусы"
        return "Прочее"

    df["category"] = df["prod_type"].apply(categorize)

    # Агрегируем по укрупнённым категориям
    cat_summary = df.groupby(["category", "steering", "severity"])["cnt"].sum().reset_index()

    for category in ["Легковые", "Грузовые", "Мото", "Автобусы"]:
        sub = cat_summary[cat_summary["category"] == category]
        if sub.empty:
            continue

        pivot = sub.pivot_table(index="steering", columns="severity", values="cnt", fill_value=0)
        if pivot.empty:
            continue

        pivot["total"] = pivot.sum(axis=1)

        if "dead" not in pivot.columns:
            continue

        logger.info(f"\nКатегория: {category}")
        for steering in pivot.index:
            total = pivot.loc[steering, "total"]
            dead = pivot.loc[steering, "dead"]
            severe = pivot.loc[steering, "severe"] if "severe" in pivot.columns else 0
            dead_pct = dead / total * 100 if total > 0 else 0
            severe_pct = severe / total * 100 if total > 0 else 0
            logger.info(
                f"  {steering}: всего {total:>5} | "
                f"dead {dead:>4} ({dead_pct:.1f}%) | "
                f"severe {severe:>4} ({severe_pct:.1f}%)"
            )

        # Сравнение
        if "RHD" in pivot.index and "LHD" in pivot.index:
            rhd_dead_pct = pivot.loc["RHD", "dead"] / pivot.loc["RHD", "total"] * 100
            lhd_dead_pct = pivot.loc["LHD", "dead"] / pivot.loc["LHD", "total"] * 100
            if lhd_dead_pct > 0:
                ratio = rhd_dead_pct / lhd_dead_pct
                logger.info(f"  → Доля смертельных RHD/LHD = {ratio:.2f}x")


def stage_3_by_vehicle_age() -> None:
    """
    Контроль на возраст автомобиля.
    Старые японки (90-х) могут давать смещение.
    """
    section_header("ЭТАП 3. С КОНТРОЛЕМ НА ВОЗРАСТ ТС")

    query = f"""
    SELECT
      CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
      CASE
        WHEN v.vehicle_year IS NULL THEN 'unknown'
        WHEN v.vehicle_year < 2000 THEN '<2000 (>20 лет)'
        WHEN v.vehicle_year < 2010 THEN '2000-2009 (10-20 лет)'
        WHEN v.vehicle_year < 2020 THEN '2010-2019 (5-15 лет)'
        ELSE '2020+ (новые)'
      END AS age_group,
      a.severity,
      COUNT(*) AS cnt
    FROM vehicles v
    JOIN accidents a ON a.id = v.accident_id
    WHERE {WHERE_CLASSIFIED}
    GROUP BY steering, age_group, a.severity
    """
    df = fetch_dataframe(query)

    age_order = [
        "<2000 (>20 лет)",
        "2000-2009 (10-20 лет)",
        "2010-2019 (5-15 лет)",
        "2020+ (новые)",
        "unknown",
    ]

    for age in age_order:
        sub = df[df["age_group"] == age]
        if sub.empty:
            continue

        pivot = sub.pivot_table(index="steering", columns="severity", values="cnt", fill_value=0)
        if pivot.empty or "dead" not in pivot.columns:
            continue

        pivot["total"] = pivot.sum(axis=1)

        logger.info(f"\nВозраст: {age}")
        for steering in pivot.index:
            total = pivot.loc[steering, "total"]
            dead = pivot.loc[steering, "dead"]
            dead_pct = dead / total * 100 if total > 0 else 0
            logger.info(f"  {steering}: всего {total:>5} | " f"dead {dead:>4} ({dead_pct:.1f}%)")


def stage_4_by_location() -> None:
    """
    Контроль на тип местности.
    Города (ГО, районы Владивостока) vs сельская местность (МР, МО, РО).
    На трассах ожидаемо выше смертность из-за скорости.
    """
    section_header("ЭТАП 4. С КОНТРОЛЕМ НА ТИП МЕСТНОСТИ")

    # Классификация места:
    # - Большие города: Владивосток (5 районов), Уссурийский ГО, Находкинский ГО, Артемовский ГО
    # - Малые города ГО: остальные ГО
    # - Сельские: МР, МО, РО
    # - Спец: ЗАТО, г. Большой Камень
    query = f"""
    SELECT
      CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
      CASE
        WHEN a.place IN (
          'Первореченский', 'Ленинский', 'Советский',
          'Первомайский', 'Фрунзенский'
        ) THEN '1.Владивосток (районы)'
        WHEN a.place IN ('Уссурийский ГО', 'Находкинский ГО', 'Артемовский ГО')
          THEN '2.Крупные города'
        WHEN a.place LIKE '%ГО'
          THEN '3.Малые города (ГО)'
        WHEN a.place LIKE '%МР' OR a.place LIKE '%МО' OR a.place LIKE '%РО'
          THEN '4.Сельские районы'
        ELSE '5.Прочее (ЗАТО и др.)'
      END AS location_type,
      a.severity,
      COUNT(*) AS cnt
    FROM vehicles v
    JOIN accidents a ON a.id = v.accident_id
    WHERE {WHERE_CLASSIFIED}
    GROUP BY steering, location_type, a.severity
    """
    df = fetch_dataframe(query)

    location_order = [
        "1.Владивосток (районы)",
        "2.Крупные города",
        "3.Малые города (ГО)",
        "4.Сельские районы",
        "5.Прочее (ЗАТО и др.)",
    ]

    for loc in location_order:
        sub = df[df["location_type"] == loc]
        if sub.empty:
            continue

        pivot = sub.pivot_table(index="steering", columns="severity", values="cnt", fill_value=0)
        if pivot.empty or "dead" not in pivot.columns:
            continue

        pivot["total"] = pivot.sum(axis=1)

        logger.info(f"\nТип местности: {loc}")
        for steering in pivot.index:
            total = pivot.loc[steering, "total"]
            dead = pivot.loc[steering, "dead"]
            dead_pct = dead / total * 100 if total > 0 else 0
            logger.info(f"  {steering}: всего {total:>6} | " f"dead {dead:>4} ({dead_pct:.1f}%)")

        if "RHD" in pivot.index and "LHD" in pivot.index:
            rhd_dead_pct = pivot.loc["RHD", "dead"] / pivot.loc["RHD", "total"] * 100
            lhd_dead_pct = pivot.loc["LHD", "dead"] / pivot.loc["LHD", "total"] * 100
            if lhd_dead_pct > 0:
                ratio = rhd_dead_pct / lhd_dead_pct
                logger.info(f"  → Доля смертельных RHD/LHD = {ratio:.2f}x")


def main() -> None:
    logger.info("=" * 72)
    logger.info("АНАЛИЗ ТЯЖЕСТИ ДТП: ПРАВЫЙ vs ЛЕВЫЙ РУЛЬ")
    logger.info("=" * 72)
    logger.info("Источник: 29413 ДТП Приморского края (2015-2026), dtp-stat.ru")

    stage_1_global_picture()
    stage_2_by_prod_type()
    stage_3_by_vehicle_age()
    stage_4_by_location()

    logger.success("Анализ завершён.")


if __name__ == "__main__":
    main()
