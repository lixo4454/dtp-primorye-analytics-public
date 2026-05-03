"""
Управление подключением к PostgreSQL через SQLAlchemy.

Читает настройки из .env, создаёт engine и фабрику сессий.
Используется во всех модулях, которые работают с БД.
"""

import os
from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Загружаем переменные из .env
load_dotenv()


def build_database_url() -> str:
    """Собирает URL подключения к PostgreSQL из переменных окружения."""
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")

    if not all([user, password, db]):
        raise RuntimeError(
            "Не заданы переменные окружения POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB. "
            "Проверь файл .env."
        )

    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


# Создаём движок подключения один раз на весь процесс
DATABASE_URL = build_database_url()
engine: Engine = create_engine(
    DATABASE_URL,
    echo=False,  # True = логировать все SQL-запросы (полезно для отладки)
    pool_size=10,  # размер пула соединений
    max_overflow=20,  # сколько ещё можно поверх pool_size при пиковой нагрузке
    pool_pre_ping=True,  # проверять, что соединение живое, перед использованием
)

# Фабрика сессий
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@contextmanager
def get_session() -> Iterator[Session]:
    """
    Контекстный менеджер для безопасной работы с сессией.
    Автоматически коммитит при успехе, откатывает при ошибке, закрывает в любом случае.

    Пример использования:
        with get_session() as session:
            session.add(accident)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
