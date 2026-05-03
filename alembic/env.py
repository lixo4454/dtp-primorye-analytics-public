"""
Alembic configuration.

Подгружает URL подключения и метаданные моделей из нашего проекта,
чтобы Alembic мог автоматически генерировать миграции на основе моделей SQLAlchemy.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Добавляем корень проекта в sys.path, чтобы можно было импортировать src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Импортируем модели и URL подключения из нашего проекта
from src.database import (  # noqa: E402
    DATABASE_URL,
    Base,
    models,
)  # noqa: F401, E402  # импорт нужен, чтобы Base.metadata знал обо всех таблицах

# Стандартная конфигурация Alembic
config = context.config

# Подставляем URL из .env вместо того, что в alembic.ini
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# Логирование
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Метаданные моделей — Alembic будет сравнивать их с реальной БД
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Запуск миграций в offline-режиме (генерируется SQL без подключения к БД)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Включаем поддержку PostGIS-типов (Geometry)
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Запуск миграций в online-режиме (с реальным подключением к БД)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


def include_object(object_, name, type_, reflected, compare_to):
    """
    Фильтр объектов для autogenerate.

    Игнорируем системные таблицы PostGIS (tiger, topology, spatial_ref_sys, etc.),
    которые создаются автоматически при установке расширения postgis.
    """
    if type_ == "table" and name in (
        "spatial_ref_sys",
        "geography_columns",
        "geometry_columns",
        "raster_columns",
        "raster_overviews",
    ):
        return False

    # Игнорируем все объекты в схемах tiger, tiger_data, topology
    if hasattr(object_, "schema") and object_.schema in (
        "tiger",
        "tiger_data",
        "topology",
    ):
        return False

    return True


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
