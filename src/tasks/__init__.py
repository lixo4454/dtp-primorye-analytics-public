"""
Celery-задачи для автообновления данных и моделей.

Структура:
- celery_app.py — Celery app + beat schedule (расписание)
- runner.py — base task с автоматическим логированием в task_runs
- parse_dtp_stat.py — еженедельная подгрузка свежих ДТП с dtp-stat.ru
- retrain_prophet.py — ежемесячный retrain Prophet с timestamp-версионированием
- retrain_catboost.py — ежемесячный retrain CatBoost v2 + isotonic
- snap_new_accidents.py — snap-to-road для новых записей (после parse)
- health_check.py — ежедневная проверка БД, Redis, моделей

Идемпотентность всех задач — обязательное требование архитектуры:
повторный запуск не ломает данные. Парсер использует ON CONFLICT DO
NOTHING, snap-to-road работает только над WHERE point_snapped IS NULL,
retrain пишет новый файл и атомарно переключает is_current.
"""

from src.tasks.celery_app import app

__all__ = ["app"]
