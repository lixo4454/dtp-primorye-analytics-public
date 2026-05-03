"""
FastAPI-приложение проекта «Анализ ДТП Приморского края».

Назначение: REST-сервис, который потребляют Streamlit-дашборд (Дни 14-15)
и потенциально внешние клиенты. Обслуживает:
- Аналитика: список ДТП с фильтрами, агрегаты, очаги аварийности
- ML-инференс: предсказание тяжести (CatBoost+isotonic), прогноз на месяц (Prophet)
- NLP: BERTopic-темы и связь Telegram-постов с записями ДТП

Архитектурные решения:
- Async stack: asyncpg + AsyncSession + selectinload против N+1
- Lifespan-протокол: модели грузятся один раз при старте (`@asynccontextmanager`)
- Pydantic v2 schemas с Field(description=...) — OpenAPI auto-doc
- CORSMiddleware: localhost:8501 (Streamlit), localhost:5173 (потенциальный React)
- Healthcheck `/health` — для Docker healthcheck и production-мониторинга
- Структура роутеров: один файл = одна тематическая группа эндпоинтов
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from src.api.dependencies import ModelRegistry, load_model_registry
from src.api.routers import accidents, admin, clusters, forecast, nlp, predict, recommendations
from src.api.schemas import HealthResponse
from src.database.async_session import async_engine

logger = logging.getLogger(__name__)

API_VERSION = "0.1.0"
API_TITLE = "DTP Primorye Analytics API"
API_DESCRIPTION = """\
REST API для аналитики ДТП Приморского края.

**Источники данных:**
- 29 413 ДТП из dtp-stat.ru (2015-2026)
- 28 020 с координатами в сухопутной границе Приморья
- 2 122 Telegram-поста @prim_police с тематической кластеризацией

**ML-модели:**
- CatBoost v2 (severity 4-class, F1-macro 0.51, ROC-AUC dead 0.73)
- IsotonicCalibratedClassifier (per-class изотонная калибровка вероятностей, ECE<0.05)
- Prophet (дневной forecast, monthly MAPE 5.4%)
- BERTopic (7 тем + шум, MiniLM-L12-v2 + UMAP-10 + HDBSCAN)

**Связанная документация:**
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
"""

# CORS-список — добавляй сюда новые фронты
ALLOWED_ORIGINS = [
    "http://localhost:8501",  # Streamlit
    "http://127.0.0.1:8501",
    "http://localhost:5173",  # Vite/React (на будущее)
    "http://127.0.0.1:5173",
]


# =====================================================================
# Lifespan: загрузка моделей на старте, dispose engine при shutdown
# =====================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Жизненный цикл приложения.

    Startup:
    - Загружаем 4 ML-модели и 6 артефактов в ModelRegistry (~5-7 сек)
    - Кладём в app.state.models — общая ссылка для всех роутеров
    - Проверяем коннект к БД одним SELECT 1
    Shutdown:
    - dispose() async engine — закрывает pool
    """
    logger.info("=== FastAPI startup: loading models ===")
    app.state.models = load_model_registry()

    # БД-sanity check
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connectivity: OK")
    except Exception as e:
        logger.error("Database connectivity failed at startup: %s", e)

    logger.info("=== FastAPI ready ===")

    yield

    logger.info("=== FastAPI shutdown: disposing engine ===")
    await async_engine.dispose()


# =====================================================================
# App factory
# =====================================================================


def create_app() -> FastAPI:
    """Создаёт FastAPI-приложение. Отдельная функция — для pytest и Uvicorn factory-mode."""
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Health
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["meta"],
        summary="Sanity-check: БД доступна, модели загружены",
    )
    async def health() -> HealthResponse:
        # БД-проверка
        db_ok = False
        accidents_count = 0
        try:
            async with async_engine.connect() as conn:
                r = await conn.execute(text("SELECT COUNT(*) FROM accidents"))
                accidents_count = int(r.scalar() or 0)
                db_ok = True
        except Exception as e:
            logger.error("/health DB check failed: %s", e)

        reg: ModelRegistry = getattr(app.state, "models", ModelRegistry())
        models_loaded = {
            "catboost_v2": reg.catboost_v2 is not None,
            "catboost_calibrated": reg.catboost_calibrated is not None,
            "prophet": reg.prophet is not None,
            "bertopic": reg.bertopic is not None,
            "topic_assignments": bool(reg.topic_assignments),
            "hotspots_summary": bool(reg.hotspots_summary),
        }
        all_ok = db_ok and all(models_loaded.values())
        return HealthResponse(
            status="ok" if all_ok else "degraded",
            db_connected=db_ok,
            accidents_count=accidents_count,
            models_loaded=models_loaded,
            version=API_VERSION,
        )

    @app.get("/", tags=["meta"], summary="Корневой эндпоинт — ссылка на /docs")
    async def root() -> dict[str, str]:
        return {
            "service": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "health": "/health",
        }

    # Тематические роутеры
    app.include_router(accidents.router)
    app.include_router(clusters.router)
    app.include_router(predict.router)
    app.include_router(forecast.router)
    app.include_router(nlp.router)
    app.include_router(recommendations.router)
    app.include_router(admin.router)

    return app


# Uvicorn-точка входа: src.api.main:app
app = create_app()
