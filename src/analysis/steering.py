# -*- coding: utf-8 -*-
"""
Модуль применения справочника правый/левый руль к таблице vehicles.

Логика:
1. Загружает CSV-справочник из data/raw/vehicle_brand_steering.csv
2. Для каждой марки в справочнике делает UPDATE vehicles SET ... WHERE mark = ...
3. Для марок, отсутствующих в справочнике, оставляет is_right_hand_drive=NULL,
   steering_confidence='unknown'.
4. Выводит финальную статистику.

Запуск:
    python -m src.analysis.steering
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database import SessionLocal

# Путь к справочнику (project_root/data/raw/vehicle_brand_steering.csv)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE_CSV = PROJECT_ROOT / "data" / "raw" / "vehicle_brand_steering.csv"


def load_brand_reference() -> pd.DataFrame:
    """Загружает справочник марок из CSV."""
    if not REFERENCE_CSV.exists():
        raise FileNotFoundError(
            f"Справочник не найден: {REFERENCE_CSV}\n"
            f"Запусти scripts/build_brand_steering_reference.py для его создания."
        )

    df = pd.read_csv(REFERENCE_CSV, encoding="utf-8")
    logger.info(f"Загружен справочник: {len(df)} марок из {REFERENCE_CSV.name}")

    # Парсим строковое 'true'/'false' в bool
    df["is_right_hand_drive"] = df["is_right_hand_drive"].map(
        {"true": True, "false": False, True: True, False: False}
    )

    # Проверим, что нет пропусков в ключевых колонках
    missing = df[df["brand"].isna() | df["confidence"].isna()]
    if not missing.empty:
        logger.warning(f"В справочнике строки с пропусками:\n{missing}")
        df = df.dropna(subset=["brand", "confidence"])

    return df


def apply_to_vehicles(reference_df: pd.DataFrame) -> dict:
    """
    Применяет справочник к таблице vehicles.

    Стратегия:
    1. Сначала ВСЁ помечается как unknown (на случай повторного запуска).
    2. Затем для каждой марки из справочника делается UPDATE.
    """
    stats = {
        "total_vehicles": 0,
        "marks_in_reference": len(reference_df),
        "marks_updated": 0,
        "rows_updated_total": 0,
        "rows_not_applicable": 0,
        "rows_remaining_unknown": 0,
    }

    with SessionLocal() as session:
        # 1. Считаем сколько всего машин в БД
        result = session.execute(text("SELECT COUNT(*) FROM vehicles"))
        stats["total_vehicles"] = result.scalar()
        logger.info(f"Машин в БД: {stats['total_vehicles']}")

        # 2. Сначала ВСЁ помечаем как unknown.
        # Это идемпотентность: если перезапускаем — старые значения сбросим.
        logger.info("Шаг 1/3: Сбрасываем все машины в unknown...")
        session.execute(
            text(
                "UPDATE vehicles " "SET is_right_hand_drive = NULL, steering_confidence = 'unknown'"
            )
        )
        session.commit()

        # 3. Применяем справочник
        logger.info("Шаг 2/3: Применяем справочник по маркам...")
        for _, row in reference_df.iterrows():
            brand: str = str(row["brand"]).strip()
            is_rhd: Optional[bool] = row["is_right_hand_drive"]
            confidence: str = str(row["confidence"]).strip()

            result = session.execute(
                text(
                    "UPDATE vehicles "
                    "SET is_right_hand_drive = :is_rhd, "
                    "    steering_confidence = :confidence "
                    "WHERE mark = :mark"
                ),
                {"is_rhd": is_rhd, "confidence": confidence, "mark": brand},
            )
            rows_affected = result.rowcount

            if rows_affected > 0:
                stats["marks_updated"] += 1
                stats["rows_updated_total"] += rows_affected
                logger.debug(
                    f"  {brand}: обновлено {rows_affected} записей "
                    f"(rhd={is_rhd}, conf={confidence})"
                )

        session.commit()

        # 3.5. Помечаем велосипеды, прицепы, полуприцепы как not_applicable
        # У них концепция RHD/LHD неприменима в принципе.
        logger.info("Шаг 2.5/3: Помечаем велосипеды/прицепы как not_applicable...")
        result = session.execute(
            text(
                "UPDATE vehicles "
                "SET is_right_hand_drive = NULL, "
                "    steering_confidence = 'not_applicable' "
                "WHERE steering_confidence = 'unknown' "
                "  AND ("
                "    prod_type ILIKE '%велосипед%' "
                "    OR prod_type ILIKE '%прицеп%' "
                "  )"
            )
        )
        stats["rows_not_applicable"] = result.rowcount
        logger.info(f"  Помечено как not_applicable: {result.rowcount}")
        session.commit()

        # 4. Подсчёт оставшихся unknown
        result = session.execute(
            text("SELECT COUNT(*) FROM vehicles WHERE steering_confidence = 'unknown'")
        )
        stats["rows_remaining_unknown"] = result.scalar()

    return stats


def print_final_stats() -> None:
    """Выводит финальную сводку по распределению руля в БД."""
    logger.info("Шаг 3/3: Финальная сводка")

    with SessionLocal() as session:
        # Распределение по типу руля
        result = session.execute(
            text(
                "SELECT "
                "  CASE "
                "    WHEN is_right_hand_drive = true THEN 'RHD (правый)' "
                "    WHEN is_right_hand_drive = false THEN 'LHD (левый)' "
                "    ELSE 'Неизвестно' "
                "  END AS steering, "
                "  steering_confidence AS conf, "
                "  COUNT(*) AS cnt "
                "FROM vehicles "
                "GROUP BY is_right_hand_drive, steering_confidence "
                "ORDER BY cnt DESC"
            )
        )
        rows = result.fetchall()

        logger.info("Распределение по типу руля:")
        total = sum(r.cnt for r in rows)
        for r in rows:
            pct = r.cnt / total * 100
            logger.info(f"  {r.steering:<20} | {r.conf:<10} | {r.cnt:>6} ({pct:.1f}%)")

        # Топ марок, оставшихся без классификации
        # Топ марок, оставшихся без классификации
        result = session.execute(
            text(
                "SELECT COALESCE(mark, '<NULL>') AS mark, COUNT(*) AS cnt "
                "FROM vehicles "
                "WHERE steering_confidence = 'unknown' "
                "GROUP BY mark "
                "ORDER BY cnt DESC "
                "LIMIT 15"
            )
        )
        unknown_rows = result.fetchall()

        if unknown_rows:
            logger.info("Топ марок без классификации (steering_confidence='unknown'):")
            for r in unknown_rows:
                mark_str = r.mark if r.mark is not None else "<NULL>"
                logger.info(f"  {mark_str:<40} | {r.cnt}")


def main() -> None:
    logger.info("=" * 70)
    logger.info("Применение справочника правый/левый руль к таблице vehicles")
    logger.info("=" * 70)

    reference_df = load_brand_reference()
    stats = apply_to_vehicles(reference_df)

    logger.info("=" * 70)
    logger.info("ИТОГИ:")
    logger.info(f"  Всего машин в БД: {stats['total_vehicles']}")
    logger.info(f"  Марок в справочнике: {stats['marks_in_reference']}")
    logger.info(f"  Из них применилось к БД: {stats['marks_updated']}")
    logger.info(f"  Всего записей обновлено: {stats['rows_updated_total']}")
    logger.info(
        f"  Помечено not_applicable (велосипеды/прицепы): {stats['rows_not_applicable']} "
        f"({stats['rows_not_applicable'] / stats['total_vehicles'] * 100:.1f}%)"
    )
    logger.info(
        f"  Осталось unknown: {stats['rows_remaining_unknown']} "
        f"({stats['rows_remaining_unknown'] / stats['total_vehicles'] * 100:.1f}%)"
    )
    logger.info("=" * 70)

    print_final_stats()
    logger.success("Готово!")


if __name__ == "__main__":
    main()
