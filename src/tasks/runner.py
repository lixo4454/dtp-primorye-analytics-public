"""
Базовая инфраструктура для Celery-задач.

@logged_task — декоратор, который:
1. Создаёт запись TaskRun(status='running') в начале.
2. Запускает обёрнутую функцию.
3. На успехе — UPDATE status='success', finished_at, duration_ms,
   payload (если функция вернула dict).
4. На исключении — UPDATE status='error', error_message с traceback,
   и пробрасывает исключение наружу (Celery его залогирует и retry-логика
   сработает по обычным правилам).

Зачем не `app.tasks` хук Celery: нам нужна структурированная запись
в нашей БД (task_runs), а не только в Celery result backend (Redis).
Дашборд читает task_runs для footer'а «последнее обновление...».
"""

from __future__ import annotations

import functools
import time
import traceback
from datetime import datetime
from typing import Any, Callable

from loguru import logger

from src.database import SessionLocal, TaskRun
from src.tasks.celery_app import app


def logged_task(name: str | None = None, **task_kwargs: Any) -> Callable:
    """Декоратор: оборачивает функцию в @app.task с TaskRun-логированием.

    Использование:

        @logged_task(name="src.tasks.parse_dtp_stat.parse_dtp_stat")
        def parse_dtp_stat() -> dict:
            ...
            return {"records_added": 12, "duration_ms_inside": 4231}

    Возвращаемый dict пишется в `task_runs.payload`.
    Если функция выкинула исключение — статус `error`, payload IS NULL,
    error_message = traceback.
    """

    def decorator(fn: Callable) -> Callable:
        task_name = name or f"{fn.__module__}.{fn.__name__}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run_id = _start_task_run(task_name, payload={"args": list(args), "kwargs": kwargs})
            t0 = time.perf_counter()
            try:
                logger.info(f"[task] start {task_name} (run_id={run_id})")
                result = fn(*args, **kwargs)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                _finish_task_run(
                    run_id,
                    status="success",
                    duration_ms=duration_ms,
                    payload_result=result if isinstance(result, dict) else None,
                )
                logger.info(f"[task] success {task_name} ({duration_ms} ms)")
                return result
            except Exception as exc:
                duration_ms = int((time.perf_counter() - t0) * 1000)
                tb = traceback.format_exc()
                _finish_task_run(run_id, status="error", duration_ms=duration_ms, error_message=tb)
                logger.error(f"[task] error {task_name}: {exc}\n{tb}")
                raise

        return app.task(name=task_name, **task_kwargs)(wrapper)

    return decorator


def _start_task_run(task_name: str, payload: dict | None = None) -> int:
    """Открывает TaskRun-запись со status='running'. Возвращает id."""
    with SessionLocal() as s:
        run = TaskRun(
            task_name=task_name,
            status="running",
            started_at=datetime.utcnow(),
            payload=payload,
        )
        s.add(run)
        s.commit()
        s.refresh(run)
        return run.id


def _finish_task_run(
    run_id: int,
    status: str,
    duration_ms: int,
    payload_result: dict | None = None,
    error_message: str | None = None,
) -> None:
    with SessionLocal() as s:
        run = s.get(TaskRun, run_id)
        if run is None:  # очень редкий race — БД без TaskRun
            logger.warning(f"TaskRun {run_id} disappeared")
            return
        run.status = status
        run.finished_at = datetime.utcnow()
        run.duration_ms = duration_ms
        if error_message is not None:
            run.error_message = error_message[:8000]  # safety cap
        if payload_result is not None:
            # mergeим payload_result в существующий payload (был args/kwargs)
            base = dict(run.payload or {})
            base["result"] = payload_result
            run.payload = base
        s.commit()
