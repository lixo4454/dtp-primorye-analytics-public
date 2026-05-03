"""
Daily health-check task.

Проверяет:
1. PostgreSQL connection: SELECT 1.
2. Redis connection: PING.
3. accidents-count в БД (smoke).
4. Текущие model_versions (что is_current не пусто).
5. Самые свежие записи в task_runs (нет ли error за последние сутки).

Возвращает dict — он пишется в task_runs.payload.result.
Если что-то критично — выкидываем RuntimeError, статус → error.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import redis
from sqlalchemy import func, select

from src.database import ModelVersion, SessionLocal, TaskRun
from src.database.models import Accident
from src.tasks.celery_app import REDIS_URL
from src.tasks.runner import logged_task


@logged_task(name="src.tasks.health_check.health_check")
def health_check() -> dict:
    summary: dict = {}

    # 1. БД + accidents-count
    with SessionLocal() as s:
        n_accidents = s.scalar(select(func.count(Accident.id))) or 0
        n_in_region = (
            s.scalar(select(func.count(Accident.id)).where(Accident.is_in_region.is_(True))) or 0
        )
        summary["accidents_total"] = int(n_accidents)
        summary["accidents_in_region"] = int(n_in_region)
        if n_accidents == 0:
            raise RuntimeError("accidents-таблица пуста — критическая аномалия")

        # 2. Активные версии моделей
        rows = s.execute(
            select(
                ModelVersion.model_name, ModelVersion.version_path, ModelVersion.trained_at
            ).where(ModelVersion.is_current.is_(True))
        ).all()
        summary["current_models"] = [
            {"name": r.model_name, "path": r.version_path, "trained_at": r.trained_at.isoformat()}
            for r in rows
        ]

        # 3. Ошибки задач за последние 24 ч (исключая саму health-check —
        # её сбои фиксируются отдельно через статус самого Celery-job'а)
        since = datetime.utcnow() - timedelta(hours=24)
        n_errors = (
            s.scalar(
                select(func.count(TaskRun.id)).where(
                    TaskRun.status == "error",
                    TaskRun.started_at >= since,
                    TaskRun.task_name != "src.tasks.health_check.health_check",
                )
            )
            or 0
        )
        summary["errors_last_24h"] = int(n_errors)

    # 4. Redis ping
    r = redis.from_url(REDIS_URL, socket_connect_timeout=3, socket_timeout=3)
    try:
        pong = r.ping()
    except Exception as exc:
        raise RuntimeError(f"Redis недоступен: {exc}") from exc
    summary["redis_ping"] = bool(pong)

    return summary
