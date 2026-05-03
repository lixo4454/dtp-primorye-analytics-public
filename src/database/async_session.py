"""
Async-подключение к PostgreSQL для FastAPI.

Параллельная синхронной session.py инфраструктура:
- AsyncEngine на драйвере asyncpg (НЕ psycopg) — это другой бинарь,
  не конфликтует с sync-engine'ом из session.py
- async_sessionmaker фабрика
- get_async_session() — FastAPI dependency, AsyncIterator[AsyncSession]

Зачем отдельно от session.py: парсеры/загрузчики/трейнеры остаются на
sync, потому что это batch-скрипты (один раз в час/день). FastAPI же
обслуживает много параллельных запросов и async + asyncpg даёт
существенно меньшую latency на блокирующих SQL-вызовах.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

load_dotenv()


def build_async_database_url() -> str:
    """Async URL: postgresql+asyncpg:// (отличается от sync postgresql+psycopg://)."""
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")

    if not all([user, password, db]):
        raise RuntimeError(
            "Не заданы POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB — проверь .env"
        )

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


ASYNC_DATABASE_URL = build_async_database_url()

# pool_size=5 + max_overflow=10 — стандарт для read-heavy API на 1 instance.
# При горизонтальном масштабировании (>1 worker) — суммируется на воркер.
async_engine: AsyncEngine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """FastAPI-зависимость: открывает AsyncSession, гарантированно закрывает.

    На исключении делает rollback. Endpoint'ы только читают (SELECT) —
    commit'ы не нужны, но если в будущем добавятся write-эндпоинты,
    они должны явно вызывать `await session.commit()`.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
