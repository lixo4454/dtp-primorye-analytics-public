"""
Admin-эндпоинты: hot-reload моделей, метаданные текущих
версий, последние Celery-запуски.

Защита: header X-Admin-Token. Сравниваем с ENV-переменной
``ADMIN_RELOAD_TOKEN``. Если не задана в окружении — endpoint всё равно
доступен (dev-режим), но в production обязательно задавать токен.
Это компромисс: для dev-удобства позволяем дергать без токена, для
prod — обязательная аутентификация.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from sqlalchemy import desc, select

from src.api.dependencies import load_model_registry
from src.database.async_session import get_async_session
from src.database.models import ModelVersion, TaskRun

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _check_token(provided: str | None) -> None:
    expected = os.getenv("ADMIN_RELOAD_TOKEN", "").strip()
    if not expected:
        # dev-режим: токен не задан в окружении — пропускаем
        return
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid X-Admin-Token",
        )


@router.post(
    "/reload_models",
    summary="Перезагрузить ML-модели из disk (после retrain)",
)
async def reload_models(
    request: Request,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict[str, Any]:
    """Заново читает все артефакты и подменяет ``app.state.models``.

    Используется Celery-задачами retrain_prophet / retrain_catboost после
    atomic-swap'а alias-файлов. Без этого reload'а API будет держать
    старую модель в памяти до своего рестарта.

    Race-condition: load_model_registry() читает все файлы и собирает
    новый ModelRegistry; присваивание app.state.models = new_reg —
    атомарное в Python (один pointer-swap). Текущие in-flight
    запросы дочитают на старом регистре, новые — на новом.
    """
    _check_token(x_admin_token)
    logger.info("[/admin/reload_models] reloading…")
    new_reg = load_model_registry()
    request.app.state.models = new_reg
    return {
        "status": "reloaded",
        "models_loaded": {
            "catboost_v2": new_reg.catboost_v2 is not None,
            "catboost_calibrated": new_reg.catboost_calibrated is not None,
            "prophet": new_reg.prophet is not None,
            "bertopic": new_reg.bertopic is not None,
        },
    }


@router.get(
    "/model_versions",
    summary="Активные версии всех моделей (для UI footer'а)",
)
async def get_active_model_versions() -> list[dict[str, Any]]:
    """Возвращает все ``model_versions`` с is_current=TRUE."""
    async for session in get_async_session():
        rows = (
            (
                await session.execute(
                    select(ModelVersion)
                    .where(ModelVersion.is_current.is_(True))
                    .order_by(ModelVersion.model_name)
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "model_name": r.model_name,
                "version_path": r.version_path,
                "trained_at": r.trained_at.isoformat() if r.trained_at else None,
                "train_size": r.train_size,
                "metadata": r.metadata_json,
            }
            for r in rows
        ]
    return []


@router.get(
    "/task_runs",
    summary="Последние Celery-запуски (журнал)",
)
async def get_recent_task_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Последние ``limit`` записей TaskRun, отсортированных по started_at desc."""
    if limit < 1 or limit > 200:
        raise HTTPException(400, "limit must be in [1, 200]")
    async for session in get_async_session():
        rows = (
            (await session.execute(select(TaskRun).order_by(desc(TaskRun.started_at)).limit(limit)))
            .scalars()
            .all()
        )
        return [
            {
                "id": r.id,
                "task_name": r.task_name,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "duration_ms": r.duration_ms,
                "error_message": (r.error_message or "")[:500] if r.error_message else None,
                "payload": r.payload,
            }
            for r in rows
        ]
    return []
