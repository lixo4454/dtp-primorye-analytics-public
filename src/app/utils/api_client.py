"""Sync httpx-клиент к FastAPI FastAPI с кешированием Streamlit.

Streamlit-скрипты выполняются синхронно, поэтому используем sync
``httpx.Client``. Один клиент на процесс через ``@st.cache_resource``
(переиспользуется keep-alive). Ответы JSON — через ``@st.cache_data``
с TTL: ``/health`` 60 сек (live-индикатор), агрегаты 1 час.

Базовый URL берётся из ``DTP_API_URL`` (default ``http://localhost:8000``);
это позволяет переопределить на хост контейнера в docker-compose.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("DTP_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=3.0)
# CatBoost predict — изотонная калибровка + 4 предсказания, ~50 мс на single
# запрос, но cold-start cache_resource внутри FastAPI до 1 сек. 15 сек запас.

_PREDICT_TIMEOUT = httpx.Timeout(20.0, connect=3.0)


@st.cache_resource(show_spinner=False)
def get_client() -> httpx.Client:
    """HTTP-клиент один на процесс (cache_resource = singleton, не копируется)."""
    return httpx.Client(base_url=API_BASE_URL, timeout=DEFAULT_TIMEOUT)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_health() -> dict | None:
    """Health-check FastAPI. None при недоступности — для индикатора в sidebar."""
    try:
        r = get_client().get("/health")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stats() -> dict:
    """``/accidents/stats`` — агрегаты для KPI и тренда. Ошибки пробрасываются."""
    r = get_client().get("/accidents/stats")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hotspots(limit: int = 30) -> dict:
    """``/clusters/hotspots`` — топ очагов аварийности (DBSCAN из шага кластеризации)."""
    r = get_client().get("/clusters/hotspots", params={"limit": limit})
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast(periods: int, start_year: int, start_month: int) -> dict:
    """``/forecast/monthly`` — Prophet-прогноз ДТП по месяцам.

    ``periods``: 1..36 (валидация на бэке). ``start_year``: 2024..2030.

    Кешируем по triple (periods, start_year, start_month) — каждый
    rerun с теми же параметрами не дёргает Prophet.predict повторно.
    """
    r = get_client().get(
        "/forecast/monthly",
        params={
            "periods": periods,
            "start_year": start_year,
            "start_month": start_month,
        },
    )
    r.raise_for_status()
    return r.json()


def predict_severity(payload: dict[str, Any]) -> dict:
    """``POST /predict/severity`` — 4 калиброванные вероятности.

    Не кешируем: пользователь меняет вход, каждый submit это уникальный
    кейс. Используем отдельный timeout (cold start калибратора ~1 сек).
    """
    with httpx.Client(base_url=API_BASE_URL, timeout=_PREDICT_TIMEOUT) as client:
        r = client.post("/predict/severity", json=payload)
        r.raise_for_status()
        return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_topics(include_noise: bool = True) -> dict:
    """``/nlp/topics`` — список 7 BERTopic-тем + шум."""
    r = get_client().get(
        "/nlp/topics",
        params={"include_noise": str(include_noise).lower()},
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_recommendations_hotspot(cluster_id: int, top_k: int = 5) -> dict:
    """``/recommendations/hotspot/{cluster_id}`` — рекомендации для DBSCAN-очага.

    Кеш 10 мин: rule engine детерминирован, методология меняется редко.
    """
    r = get_client().get(
        f"/recommendations/hotspot/{cluster_id}",
        params={"top_k": top_k},
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_recommendations_point(
    lat: float,
    lon: float,
    radius: int,
    top_k: int = 5,
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict:
    """``/recommendations/point?lat=&lon=&radius=&top_k=&year_min=&year_max=``.

    Кеш 10 мин. Координаты округляем до 5 знаков, чтобы случайные пиксельные
    сдвиги клика не ломали кеш. ``year_min/year_max`` фильтруют ДТП по году.
    """
    lat_r = round(lat, 5)
    lon_r = round(lon, 5)
    params: dict[str, Any] = {
        "lat": lat_r,
        "lon": lon_r,
        "radius": radius,
        "top_k": top_k,
    }
    if year_min is not None:
        params["year_min"] = year_min
    if year_max is not None:
        params["year_max"] = year_max
    r = get_client().get("/recommendations/point", params=params)
    r.raise_for_status()
    return r.json()


def predict_severity_counterfactual(payload: dict[str, Any]) -> dict:
    """``POST /predict/severity_counterfactual`` — counterfactual через CatBoost.

    Не кешируем (каждый сценарий — уникальный, пользователь может
    менять чекбоксы). Используем такой же 20-сек timeout что и
    /predict/severity (cold-start калибратора).
    """
    with httpx.Client(base_url=API_BASE_URL, timeout=_PREDICT_TIMEOUT) as client:
        r = client.post("/predict/severity_counterfactual", json=payload)
        r.raise_for_status()
        return r.json()


# =====================================================================
# admin endpoints (model_versions, task_runs)
# =====================================================================


@st.cache_data(ttl=300, show_spinner=False)
def fetch_active_model_versions() -> list[dict[str, Any]]:
    """``GET /admin/model_versions`` — текущие активные версии моделей.

    Кеш 5 мин: retrain происходит ежемесячно, чаще запрашивать не нужно.
    Используется в footer'е для timestamp'а «последнее обновление модели».
    """
    try:
        r = get_client().get("/admin/model_versions")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return []


@st.cache_data(ttl=120, show_spinner=False)
def fetch_recent_task_runs(limit: int = 20) -> list[dict[str, Any]]:
    """``GET /admin/task_runs?limit=N`` — последние Celery-запуски."""
    try:
        r = get_client().get("/admin/task_runs", params={"limit": limit})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return []
