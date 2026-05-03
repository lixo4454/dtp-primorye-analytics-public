"""
Загрузчик аккумулированных данных dtp-stat.ru в PostgreSQL.

Идемпотентен: если запись с таким external_id уже есть, пропускает.
Батчевая вставка по 500 записей.
"""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.database import SessionLocal
from src.database.models import Accident, DataUpdateLog
from src.loaders.dtp_stat_accumulated_downloader import prepare_data
from src.loaders.dtp_stat_accumulated_parser import parse_record

BATCH_SIZE = 500


def get_existing_external_ids(session) -> set[int]:
    """Возвращает множество external_id, уже существующих в БД."""
    rows = session.execute(select(Accident.external_id)).all()
    return {row[0] for row in rows}


def load_to_db(json_path: Path | None = None, force_redownload: bool = False) -> dict:
    """
    Главная функция загрузки.

    1. Готовит данные (скачивает/распаковывает/извлекает Приморье если нужно)
    2. Читает финальный JSON
    3. Для каждой записи: если её ещё нет в БД — добавляет
    4. Пишет в data_updates_log

    Returns:
        dict со статистикой.
    """
    started_at = datetime.now()

    # 1. Подготовка данных
    if json_path is None:
        json_path = prepare_data(force=force_redownload)

    # 2. Чтение JSON
    logger.info(f"Читаю {json_path.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    logger.info(f"Записей в файле: {total:,}")

    stats = {"total": total, "added": 0, "skipped": 0, "errors": 0}

    with SessionLocal() as session:
        log = DataUpdateLog(
            source="dtp-stat-accumulated",
            status="running",
        )
        session.add(log)
        session.commit()

        try:
            existing_ids = get_existing_external_ids(session)
            logger.info(f"В БД уже {len(existing_ids):,} записей")

            batch: list[Accident] = []

            for i, rec in enumerate(records, start=1):
                ext_id = rec.get("EM_NUMBER")

                if ext_id in existing_ids:
                    stats["skipped"] += 1
                    continue

                accident = parse_record(rec)
                if accident is None:
                    stats["errors"] += 1
                    continue

                batch.append(accident)
                existing_ids.add(ext_id)

                if len(batch) >= BATCH_SIZE:
                    _flush_batch(session, batch, stats)
                    batch = []

                if i % 1000 == 0:
                    logger.info(
                        f"Обработано {i:,}/{total:,} | "
                        f"добавлено={stats['added']:,} | "
                        f"пропущено={stats['skipped']} | "
                        f"ошибок={stats['errors']}"
                    )

            if batch:
                _flush_batch(session, batch, stats)

            log.finished_at = datetime.now()
            log.status = "success"
            log.records_added = stats["added"]
            session.commit()

        except Exception as e:
            log.finished_at = datetime.now()
            log.status = "error"
            log.error_message = str(e)
            session.commit()
            logger.error(f"Загрузка прервана: {e}")
            raise

    duration = (datetime.now() - started_at).total_seconds()
    stats["duration_sec"] = round(duration, 2)
    logger.info(f"=== ИТОГ === {stats}")
    return stats


def _flush_batch(session, batch: list[Accident], stats: dict) -> None:
    """Сохраняет батч в БД и обновляет статистику."""
    try:
        session.add_all(batch)
        session.commit()
        stats["added"] += len(batch)
    except IntegrityError as e:
        session.rollback()
        logger.warning(f"Конфликт в батче: {e}. Пробую по одной...")
        for accident in batch:
            try:
                session.add(accident)
                session.commit()
                stats["added"] += 1
            except IntegrityError:
                session.rollback()
                stats["errors"] += 1


if __name__ == "__main__":
    result = load_to_db()
    print("\n=== РЕЗУЛЬТАТ ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
