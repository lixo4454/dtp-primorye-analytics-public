"""
Snap-to-road для свежих ДТП после weekly-парсинга.

Идемпотентно: WHERE point_snapped IS NULL AND snap_method IS NULL.
После первого batch-прогона `scripts/snap_existing_accidents.py`
обработает все исторические 28 020 ДТП. Эта Celery-задача чистит
«хвост» — те, что появились с последнего запуска.

Inline-версия (snap_new_accidents_inline) вызывается напрямую из
parse_dtp_stat — не ставим её через Celery-queue, чтобы избежать
race с продолжающимся транзакционным контекстом и flag'ом
"данные не загружены полностью".
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.analysis.snap_to_road import snap_batch_sql
from src.database.session import SessionLocal
from src.tasks.runner import logged_task


@logged_task(name="src.tasks.snap_new_accidents.snap_new_accidents")
def snap_new_accidents(batch_size: int = 5000) -> dict[str, Any]:
    """Stand-alone задача (можно дёрнуть руками: celery call)."""
    return snap_new_accidents_inline(batch_size=batch_size)


def snap_new_accidents_inline(batch_size: int = 5000) -> dict[str, Any]:
    """Синхронный inline-вариант для chain'а после parse_dtp_stat."""
    logger.info(f"[snap_new] start, batch_size={batch_size}")
    total_counters = {"osm_road": 0, "unchanged": 0}
    total_processed = 0
    iterations = 0

    while True:
        with SessionLocal() as s:
            counters = snap_batch_sql(s, batch_size=batch_size)
        processed_in_batch = sum(counters.values())
        total_processed += processed_in_batch
        for k in total_counters:
            total_counters[k] += counters.get(k, 0)
        iterations += 1
        logger.info(f"[snap_new] iter={iterations} batch={counters}")
        if processed_in_batch == 0:
            break
        if iterations >= 50:
            # 50 × 5000 = 250k — больше чем ДТП в БД. Если упёрлись —
            # значит в snap_batch_sql ошибка (записи не помечаются после UPDATE'а).
            raise RuntimeError(f"[snap_new] safety cap reached after {iterations} iterations")

    return {
        "iterations": iterations,
        "processed": total_processed,
        "by_method": total_counters,
    }
