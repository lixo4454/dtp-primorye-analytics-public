"""
Базовый класс для всех ORM-моделей.
Все модели наследуются от Base.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Базовый класс для всех моделей SQLAlchemy."""

    pass
