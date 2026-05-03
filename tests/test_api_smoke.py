"""
Smoke-тесты FastAPI.

Проверяют только status-код 200 + базовую структуру response — детальное
покрытие через pytest + GitHub Actions.

Архитектура:
- httpx.AsyncClient + ASGITransport(app=...) — позволяет тестировать
  async-эндпоинты без поднятия uvicorn (без бинарного порта)
- Реальная dev-БД (Postgres healthy) — это smoke, не unit. Mock-БД на

- Lifespan активируется через `LifespanManager` или `httpx.AsyncClient`
  context — без него `app.state.models` пустой, и эндпоинты возвращают 503.
"""

from __future__ import annotations

import asyncio
import sys

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app

# На Windows asyncpg несовместим с ProactorEventLoop (default Python 3.8+).
# Под uvicorn это решается автоматически (--loop=asyncio + SelectorEventLoopPolicy),
# но в pytest-asyncio нужно прописать явно. Иначе тесты с DB-запросами
# падают с `AttributeError: 'NoneType' object has no attribute 'send'`
# при cleanup пула.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def client():
    """Поднимает FastAPI с lifespan через ASGI-transport.

    Порядок: сначала lifespan (грузит модели в app.state), потом
    AsyncClient внутри. AsyncClient + ASGITransport не триггерит
    lifespan автоматически — нужен явный `lifespan_context`.

    Session-scope (см. pytest.ini asyncio_default_*_loop_scope=session):
    модели загружаются один раз на весь набор тестов — иначе ~5 сек * N тестов.
    """
    app = create_app()
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# =====================================================================
# Meta
# =====================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_health_endpoint(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["accidents_count"], int)
    assert body["accidents_count"] > 0
    assert isinstance(body["models_loaded"], dict)
    assert "catboost_v2" in body["models_loaded"]


@pytest.mark.asyncio(loop_scope="session")
async def test_root_endpoint(client: AsyncClient):
    r = await client.get("/")
    assert r.status_code == 200
    assert r.json()["service"]


@pytest.mark.asyncio(loop_scope="session")
async def test_openapi_docs(client: AsyncClient):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    paths = schema["paths"]
    # Все 14 эндпоинтов должны быть в OpenAPI
    expected = [
        "/health",
        "/",
        "/accidents",
        "/accidents/{accident_id}",
        "/accidents/stats",
        "/clusters/hotspots",
        "/predict/severity",
        "/predict/severity_counterfactual",
        "/forecast/monthly",
        "/nlp/topics",
        "/nlp/posts/{tg_id}",
        "/nlp/search",
        "/recommendations/hotspot/{cluster_id}",
        "/recommendations/point",
    ]
    for p in expected:
        assert p in paths, f"endpoint {p} missing from OpenAPI"


# =====================================================================
# Accidents
# =====================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_accidents_list(client: AsyncClient):
    r = await client.get("/accidents", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert "total" in body and body["total"] > 0
    assert len(body["items"]) == 5
    item = body["items"][0]
    # Pydantic-схема обещает id, datetime, severity
    assert "id" in item
    assert "datetime" in item
    assert "severity" in item


@pytest.mark.asyncio(loop_scope="session")
async def test_accidents_filter_by_year(client: AsyncClient):
    r = await client.get("/accidents", params={"year": 2024, "limit": 3})
    assert r.status_code == 200
    body = r.json()
    for item in body["items"]:
        assert item["datetime"].startswith("2024-")


@pytest.mark.asyncio(loop_scope="session")
async def test_accidents_stats(client: AsyncClient):
    r = await client.get("/accidents/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    assert body["in_region"] > 0
    assert "light" in body["by_severity"]
    assert len(body["by_year"]) >= 5  # минимум 5 лет данных


@pytest.mark.asyncio(loop_scope="session")
async def test_accidents_detail(client: AsyncClient):
    r = await client.get("/accidents/1")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1
    assert "vehicles" in body
    assert "pedestrians" in body
    assert isinstance(body["vehicles"], list)


@pytest.mark.asyncio(loop_scope="session")
async def test_accidents_detail_404(client: AsyncClient):
    r = await client.get("/accidents/999999999")
    assert r.status_code == 404


# =====================================================================
# Clusters
# =====================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_clusters_hotspots(client: AsyncClient):
    r = await client.get("/clusters/hotspots", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert "params" in body
    assert "stats" in body
    assert len(body["items"]) <= 5
    if body["items"]:
        h = body["items"][0]
        assert "centroid" in h
        assert -90 <= h["centroid"]["lat"] <= 90
        assert -180 <= h["centroid"]["lon"] <= 180


# =====================================================================
# Predict
# =====================================================================


SAMPLE_PREDICT = {
    "hour": 22,
    "dow": 5,
    "month": 11,
    "year": 2026,
    "lat": 43.115,
    "lon": 131.886,
    "veh_amount": 1,
    "veh_count_actual": 1,
    "rhd_share": 1.0,
    "classified_veh": 1.0,
    "avg_vehicle_year": 2008.0,
    "part_count": 2,
    "drunk_share": 0.0,
    "med_known_count": 1.0,
    "unbelted_share": 0.5,
    "avg_age_from_tg": None,
    "ped_count": 1,
    "avg_ped_age_from_tg": None,
    "light_type": "В темное время суток, освещение не включено",
    "traffic_area_state": "Сухое",
    "mt_rate": "В населенном пункте",
    "clouds_top": "Ясно",
    "em_type": "Наезд на пешехода",
    "np_top": "г Владивосток",
    "mark_top": "TOYOTA",
    "is_weekend": True,
    "is_holiday": False,
    "is_highway": False,
    "is_in_region": True,
    "has_defect": False,
    "has_moto": False,
    "has_truck_or_bus": False,
    "has_known_age": False,
    "has_known_ped_age": False,
}


@pytest.mark.asyncio(loop_scope="session")
async def test_predict_severity(client: AsyncClient):
    r = await client.post("/predict/severity", json=SAMPLE_PREDICT)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in {"light", "severe", "severe_multiple", "dead"}
    assert len(body["probabilities"]) == 4
    proba_sum = sum(body["probabilities"].values())
    # Изотонная нормализация — сумма точно 1.0 ± эпсилон
    assert abs(proba_sum - 1.0) < 1e-6


@pytest.mark.asyncio(loop_scope="session")
async def test_predict_severity_validation_error(client: AsyncClient):
    """Отсутствует обязательное поле — Pydantic должен вернуть 422."""
    r = await client.post("/predict/severity", json={"hour": 12})
    assert r.status_code == 422


# =====================================================================
# Forecast
# =====================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_forecast_monthly(client: AsyncClient):
    r = await client.get(
        "/forecast/monthly", params={"periods": 3, "start_year": 2026, "start_month": 1}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_months"] == 3
    assert len(body["items"]) == 3
    point = body["items"][0]
    assert point["yhat"] > 0
    # CI: lower <= yhat <= upper
    assert point["yhat_lower"] <= point["yhat"] <= point["yhat_upper"]


# =====================================================================
# NLP
# =====================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_nlp_topics(client: AsyncClient):
    r = await client.get("/nlp/topics")
    assert r.status_code == 200
    body = r.json()
    # 7 реальных тем + 1 шумовая
    assert body["n_topics"] == 7
    assert body["n_posts"] == 2122
    assert len(body["items"]) >= 7


@pytest.mark.asyncio(loop_scope="session")
async def test_nlp_post_lookup(client: AsyncClient):
    r = await client.get("/nlp/posts/11")
    assert r.status_code == 200
    body = r.json()
    assert body["tg_id"] == 11
    assert body["topic_id"] is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_nlp_post_404(client: AsyncClient):
    r = await client.get("/nlp/posts/99999999")
    assert r.status_code == 404


# =====================================================================
# Recommendations
# =====================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_recommendations_hotspot(client: AsyncClient):
    """Топ-K рекомендаций для DBSCAN-очага."""
    r = await client.get("/recommendations/hotspot/2", params={"top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "profile" in body
    assert "items" in body
    profile = body["profile"]
    assert profile["source"] == "dbscan"
    assert profile["cluster_id"] == 2
    assert profile["n_points"] > 0
    # Items должны быть Recommendation-структуры с полным контрактом
    for rec in body["items"]:
        assert "rule_id" in rec and rec["rule_id"].startswith("R")
        assert rec["priority"] in (1, 2, 3)
        assert "title" in rec
        assert "expected_effect" in rec
        eff = rec["expected_effect"]
        # CMF должен быть отрицательным (мера снижает риск)
        assert eff["point_estimate"] < 0
        assert "evidence_basis" in rec and len(rec["evidence_basis"]) >= 1
        for cit in rec["evidence_basis"]:
            assert cit["url"].startswith("http")
        assert 0.0 <= rec["confidence"] <= 1.0
        assert rec["score"] > 0


@pytest.mark.asyncio(loop_scope="session")
async def test_recommendations_hotspot_404(client: AsyncClient):
    r = await client.get("/recommendations/hotspot/99999")
    assert r.status_code == 404


@pytest.mark.asyncio(loop_scope="session")
async def test_recommendations_point_dynamic(client: AsyncClient):
    """Динамический очаг через PostGIS ST_DWithin."""
    # Центр Владивостока, радиус 300 м — гарантированно ≥ 10 ДТП
    r = await client.get(
        "/recommendations/point",
        params={"lat": 43.115, "lon": 131.886, "radius": 300, "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    profile = body["profile"]
    assert profile["source"] == "dynamic_radius"
    assert profile["radius_query_m"] == 300
    assert profile["n_points"] >= 10  # достаточный sample
    # Должен быть top_em_type (хотя бы какой-то)
    assert profile["top_em_type"] is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_recommendations_point_insufficient_sample(client: AsyncClient):
    """Маленький радиус → пустой items + note."""
    # Точка где-то в океане — 0 ДТП
    r = await client.get(
        "/recommendations/point",
        params={"lat": 42.0, "lon": 130.0, "radius": 30, "top_k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["note"] is not None
    assert "увеличь радиус" in body["note"].lower() or "увеличь" in body["note"].lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_recommendations_point_year_filter(client: AsyncClient):
    """year_min/year_max сужают выборку: full > 2023-2025 > 2025."""
    base = {"lat": 43.115, "lon": 131.886, "radius": 300, "top_k": 3}
    r_full = await client.get("/recommendations/point", params=base)
    r_period = await client.get(
        "/recommendations/point",
        params={**base, "year_min": 2023, "year_max": 2025},
    )
    r_one = await client.get(
        "/recommendations/point",
        params={**base, "year_min": 2025, "year_max": 2025},
    )
    assert r_full.status_code == 200
    assert r_period.status_code == 200
    assert r_one.status_code == 200
    n_full = r_full.json()["profile"]["n_points"]
    n_period = r_period.json()["profile"]["n_points"]
    n_one = r_one.json()["profile"]["n_points"]
    assert n_full >= n_period >= n_one


# =====================================================================
# Counterfactual
# =====================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_severity_counterfactual(client: AsyncClient):
    """POST /predict/severity_counterfactual с 2 сценариями."""
    payload = {
        "baseline": {
            "hour": 22,
            "dow": 5,
            "month": 11,
            "year": 2026,
            "lat": 43.115,
            "lon": 131.886,
            "veh_amount": 1,
            "veh_count_actual": 1,
            "rhd_share": 1.0,
            "classified_veh": 1.0,
            "avg_vehicle_year": 2008.0,
            "part_count": 2,
            "drunk_share": 0.5,
            "med_known_count": 1.0,
            "unbelted_share": 1.0,
            "ped_count": 1,
            "light_type": "В темное время суток, освещение не включено",
            "traffic_area_state": "Сухое",
            "mt_rate": "Режим движения не изменялся",
            "clouds_top": "Ясно",
            "em_type": "Наезд на пешехода",
            "np_top": "г Владивосток",
            "mark_top": "TOYOTA",
            "is_weekend": True,
            "is_holiday": False,
            "is_highway": False,
            "is_in_region": True,
            "has_defect": False,
            "has_moto": False,
            "has_truck_or_bus": False,
            "has_known_age": False,
            "has_known_ped_age": False,
        },
        "scenarios": [
            {"name": "Все пристёгнуты", "overrides": {"unbelted_share": 0.0}},
            {"name": "Никто не пьян", "overrides": {"drunk_share": 0.0}},
        ],
    }
    r = await client.post("/predict/severity_counterfactual", json=payload)
    assert r.status_code == 200
    body = r.json()
    # Baseline должен быть валидной 4-class распределением
    bp = body["baseline_proba"]
    assert set(bp.keys()) == {"light", "severe", "severe_multiple", "dead"}
    total = sum(bp.values())
    assert 0.99 < total < 1.01
    assert body["baseline_predicted_class"] in {"light", "severe", "severe_multiple", "dead"}
    # Сценариев = 2
    assert len(body["scenarios"]) == 2
    for sc in body["scenarios"]:
        assert "name" in sc
        assert "modified_proba" in sc
        assert "delta_proba" in sc
        assert "delta_dead_pct_points" in sc
        # Σ modified_proba ≈ 1
        mp_total = sum(sc["modified_proba"].values())
        assert 0.99 < mp_total < 1.01
        # Σ delta = 0 (закон сохранения вероятности)
        delta_total = sum(sc["delta_proba"].values())
        assert abs(delta_total) < 1e-6


@pytest.mark.asyncio(loop_scope="session")
async def test_severity_counterfactual_unknown_override_422(client: AsyncClient):
    """Override на несуществующий признак → 422."""
    payload = {
        "baseline": {
            "hour": 14,
            "dow": 3,
            "month": 7,
            "year": 2019,
            "lat": 43.31,
            "lon": 131.99,
            "veh_amount": 1,
            "veh_count_actual": 1,
            "rhd_share": 1.0,
            "classified_veh": 1.0,
            "avg_vehicle_year": 2004.0,
            "part_count": 2,
            "drunk_share": 0.0,
            "med_known_count": 0.0,
            "unbelted_share": 0.0,
            "ped_count": 0,
            "light_type": "Светлое время суток",
            "traffic_area_state": "Сухое",
            "mt_rate": "Режим движения не изменялся",
            "clouds_top": "Ясно",
            "em_type": "Столкновение",
            "np_top": "г Владивосток",
            "mark_top": "TOYOTA",
            "is_weekend": False,
            "is_holiday": False,
            "is_highway": False,
            "is_in_region": True,
            "has_defect": False,
            "has_moto": False,
            "has_truck_or_bus": False,
            "has_known_age": False,
            "has_known_ped_age": False,
        },
        "scenarios": [
            {"name": "Bad override", "overrides": {"nonexistent_feature": 42}},
        ],
    }
    r = await client.post("/predict/severity_counterfactual", json=payload)
    assert r.status_code == 422
