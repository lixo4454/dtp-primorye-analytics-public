"""
Конфигурация Celery + расписание Beat для проекта.

Зачем:
- Парсер dtp-stat.ru — раз в неделю (понедельник 03:00 UTC+10 / 20:00 пятница UTC).
- Retrain Prophet — раз в месяц (1-го числа 04:00 UTC+10).
- Retrain CatBoost — раз в месяц (1-го числа 05:00 UTC+10).
- Health-check — каждый день 02:00 UTC+10.

UTC vs локальное время: Celery beat по умолчанию работает в UTC.
Приморский край = UTC+10. Чтобы расписание `weekly_parse` запускалось
именно в понедельник 03:00 по местному времени, в crontab указываем
`hour=17, day_of_week='sun'` (= 03:00 пн UTC+10). Документировано
в каждой beat-записи.

Idempotency:
- Парсер: ON CONFLICT DO NOTHING на external_id (Дни 2/4).
- Retrain: пишет файл с timestamp'ом, swap is_current — atomic UPDATE.
- Snap-to-road: WHERE point_snapped IS NULL.
- Health-check: read-only.

Каждая задача оборачивается @logged_task декоратором
(см. src/tasks/runner.py) — TaskRun-запись создаётся автоматически.
"""

from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

load_dotenv()


def _build_redis_url() -> str:
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_DB", "0")
    return f"redis://{host}:{port}/{db}"


REDIS_URL = _build_redis_url()


app = Celery(
    "dtp",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "src.tasks.parse_dtp_stat",
        "src.tasks.retrain_prophet",
        "src.tasks.retrain_catboost",
        "src.tasks.snap_new_accidents",
        "src.tasks.health_check",
    ],
)

app.conf.update(
    timezone="UTC",
    enable_utc=True,
    # ровно одна попытка — задачи длительные (parse 60-120с, retrain 30с-5мин).
    # При сбое лучше зафиксировать error в task_runs и разобраться вручную,
    # чем повторно дёргать парсер и плодить странные дубликаты.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # heavy tasks → не префетчим лишнее
    task_track_started=True,
    result_expires=3600 * 24 * 7,  # храним результаты 7 дней
    # Сериализация — JSON (а не pickle), безопаснее на запуске worker'ов
    # с разной версией кода
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)


# =====================================================================
# Beat schedule — расписание периодических задач
# =====================================================================
#
# Приморье: UTC+10 круглый год (нет DST с 2014). Все hour'ы ниже даны
# в UTC; рядом — комментарий с локальным временем.
app.conf.beat_schedule = {
    # Каждый понедельник 03:00 локального → 17:00 sunday UTC
    "weekly-parse-dtp-stat": {
        "task": "src.tasks.parse_dtp_stat.parse_dtp_stat",
        "schedule": crontab(hour=17, minute=0, day_of_week="sun"),
        "options": {"expires": 60 * 60 * 6},  # если worker лежал 6 ч — пропускаем
    },
    # Каждое 1-е число месяца 04:00 локального → 18:00 предыдущего дня UTC
    # (≈ конец месяца UTC, но это OK — задача ежемесячная без жёсткого календаря)
    "monthly-retrain-prophet": {
        "task": "src.tasks.retrain_prophet.retrain_prophet",
        "schedule": crontab(hour=18, minute=0, day_of_month="1"),
        "options": {"expires": 60 * 60 * 12},
    },
    # Каждое 1-е число 05:00 локального → 19:00 UTC предыдущего дня
    # (фактически Celery интерпретирует day_of_month=1 как 1-е число
    # текущей даты UTC — близкая дата к prophet-retrain)
    "monthly-retrain-catboost": {
        "task": "src.tasks.retrain_catboost.retrain_catboost",
        "schedule": crontab(hour=19, minute=0, day_of_month="1"),
        "options": {"expires": 60 * 60 * 12},
    },
    # Каждый день 02:00 локального → 16:00 предыдущего дня UTC
    "daily-health-check": {
        "task": "src.tasks.health_check.health_check",
        "schedule": crontab(hour=16, minute=0),
        "options": {"expires": 60 * 60 * 2},
    },
}


# Запуск worker'а:
#   celery -A src.tasks.celery_app worker --loglevel=info --pool=solo
# (--pool=solo — для Windows и dev; в Linux/Docker → --pool=prefork --concurrency=2)
#
# Запуск beat:
#   celery -A src.tasks.celery_app beat --loglevel=info
#
# Inspect:
#   celery -A src.tasks.celery_app inspect active
#   celery -A src.tasks.celery_app inspect scheduled
