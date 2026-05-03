"""
Еженедельный парсер dtp-stat.ru.

Wrapper'ит существующий `src.loaders.dtp_stat_accumulated_loader.load_to_db()`
он уже идемпотентный (ON CONFLICT DO NOTHING / skip по
external_id), так что повторный запуск безопасен.

Что делаем:
1. Скачиваем актуальный архив 2015-202X с dtp-stat.ru (force=True для
   weekly запуска, чтобы получить свежие данные).
2. Распаковка → извлечение Приморского края → load в БД.
3. После успешной загрузки — chain'им snap-to-road для новых записей,
   чтобы заполнить point_snapped/snap_method без отдельного beat-расписания.

Idempotency:
- Парсер: ON CONFLICT DO NOTHING (новые external_id → INSERT,
  существующие → SKIP). См. риск 1 в плане7 — UPSERT не
  делаем сознательно (источник не даёт field-level changelog).
- Snap-to-road: WHERE point_snapped IS NULL — обработает только
  свежие записи.

Возврат: dict с ключами records_added / records_skipped / records_errors /
duration_sec — пишется в task_runs.payload.result.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.tasks.runner import logged_task


@logged_task(name="src.tasks.parse_dtp_stat.parse_dtp_stat")
def parse_dtp_stat(force_redownload: bool = True) -> dict[str, Any]:
    """Скачать + загрузить + (chain) snap-to-road для новых ДТП.

    Loader импортируется лениво: ``ijson`` + ``py7zr`` нужны только
    при фактическом выполнении парсера (heavy deps, не нужны worker'у
    на старте). Если ``ijson``/``py7zr`` не установлены в образе worker'а —
    задача упадёт с ImportError и запишется в task_runs с error;
    тогда добавим их в requirements.txt.
    """
    # Lazy imports: heavy parser-only deps
    from src.loaders.dtp_stat_accumulated_loader import load_to_db

    logger.info("Запуск weekly-парсера dtp-stat.ru (force_redownload=%s)" % force_redownload)

    stats = load_to_db(force_redownload=force_redownload)

    # После загрузки запускаем snap-to-road в той же worker-сессии — это
    # синхронный вызов (~1-2 сек на 100 новых ДТП), не плодим лишний
    # task_run-overhead. Но если snap-to-road упадёт — мы уже зафиксировали
    # parse-успех, так что catch-and-log:
    try:
        from src.tasks.snap_new_accidents import snap_new_accidents_inline

        snap_summary = snap_new_accidents_inline()
        stats["snap"] = snap_summary
    except Exception as exc:
        logger.warning(f"snap-to-road после parse упал: {exc}")
        stats["snap_error"] = str(exc)

    return stats
