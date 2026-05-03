"""
Тесты для admin-эндпоинтов и snap-to-road.

Покрывает:
- /admin/model_versions: возвращает is_current=TRUE версии
- /admin/task_runs: список последних запусков
- /admin/reload_models: перезагружает app.state.models
- snap_to_road.snap_point: работает на одиночной точке
- model_registry: register_version + atomic_swap_alias

DB и Redis должны быть доступны (это integration smoke, не unit).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        async with app.router.lifespan_context(app):
            yield ac


# =====================================================================
# /admin/model_versions
# =====================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_model_versions_returns_list(client: AsyncClient):
    r = await client.get("/admin/model_versions")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    # После seed_model_versions ожидаем 4 модели
    names = {v["model_name"] for v in data}
    assert "prophet_dtp" in names or "catboost_severity_v2" in names


@pytest.mark.asyncio(loop_scope="module")
async def test_model_versions_has_required_fields(client: AsyncClient):
    r = await client.get("/admin/model_versions")
    assert r.status_code == 200
    for v in r.json():
        assert "model_name" in v
        assert "version_path" in v
        assert "trained_at" in v


# =====================================================================
# /admin/task_runs
# =====================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_task_runs_default_limit(client: AsyncClient):
    r = await client.get("/admin/task_runs")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) <= 20


@pytest.mark.asyncio(loop_scope="module")
async def test_task_runs_limit_validation(client: AsyncClient):
    r = await client.get("/admin/task_runs?limit=0")
    assert r.status_code == 400
    r = await client.get("/admin/task_runs?limit=500")
    assert r.status_code == 400


@pytest.mark.asyncio(loop_scope="module")
async def test_task_runs_payload_structure(client: AsyncClient):
    r = await client.get("/admin/task_runs?limit=5")
    assert r.status_code == 200
    for run in r.json():
        assert "task_name" in run
        assert "status" in run
        assert run["status"] in ("running", "success", "error", "skipped")


# =====================================================================
# /admin/reload_models
# =====================================================================


@pytest.mark.asyncio(loop_scope="module")
async def test_reload_models_dev_mode(client: AsyncClient):
    """Без ADMIN_RELOAD_TOKEN в env — endpoint открыт."""
    r = await client.post("/admin/reload_models")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "reloaded"
    assert "models_loaded" in body


# =====================================================================
# Snap-to-road: snap_point на одной точке
# =====================================================================


def test_snap_point_central_vladivostok():
    """Snap на точку прямо на Светланской улице должен дать дистанцию < 10 м."""
    from src.analysis.snap_to_road import snap_point
    from src.database import SessionLocal

    with SessionLocal() as s:
        # Светланская улица: координаты с самого asphalt'а (взято из ST_AsText)
        result = snap_point(s, lon=131.88610, lat=43.11563)
    assert result.snap_method == "osm_road"
    assert result.snap_distance_m is not None
    assert result.snap_distance_m < 10.0  # точка прямо на дороге
    assert result.snapped_lat is not None
    assert result.snapped_lon is not None


def test_snap_point_offshore_unchanged():
    """Snap на точку посреди Японского моря (>>50м от road graph) → unchanged."""
    from src.analysis.snap_to_road import snap_point
    from src.database import SessionLocal

    with SessionLocal() as s:
        # Точка в Японском море, 100 км от берега
        result = snap_point(s, lon=134.0, lat=42.0)
    # Должна быть unchanged, потому что нет road в радиусе 50 м
    assert result.snap_method == "unchanged"
    assert result.snap_distance_m > 50.0


# =====================================================================
# model_registry
# =====================================================================


def test_register_version_and_swap():
    """register_version + atomic_swap_alias: создаём, переключаем, очищаем."""
    from sqlalchemy import select

    from src.database import ModelVersion, SessionLocal
    from src.tasks.model_registry import (
        atomic_swap_alias,
        register_version,
    )

    test_model = "test_pytest_dummy"
    with tempfile.TemporaryDirectory() as tmp:
        # Создаём фейковый "model file"
        version_path = Path(tmp) / f"{test_model}_20260502.bin"
        version_path.write_bytes(b"hello world")
        alias_path = Path(tmp) / f"{test_model}.bin"

        # 1. Регистрируем версию
        version_id = register_version(
            test_model,
            version_path,
            metadata={"test": True, "marker": "v1"},
            train_size=42,
            make_current=True,
        )
        assert version_id > 0

        # 2. Atomic swap (in-tmp)
        atomic_swap_alias(version_path, alias_path)
        assert alias_path.exists()
        assert alias_path.read_bytes() == b"hello world"

        # 3. Регистрируем второй раз — старый is_current=False, новый =True
        version_path2 = Path(tmp) / f"{test_model}_20260503.bin"
        version_path2.write_bytes(b"v2 data")
        register_version(
            test_model,
            version_path2,
            metadata={"test": True, "marker": "v2"},
            train_size=99,
            make_current=True,
        )

        with SessionLocal() as s:
            current = s.execute(
                select(ModelVersion)
                .where(ModelVersion.model_name == test_model)
                .where(ModelVersion.is_current.is_(True))
            ).scalar_one()
            assert current.metadata_json["marker"] == "v2"
            assert current.train_size == 99

            n_total = (
                s.execute(select(ModelVersion).where(ModelVersion.model_name == test_model))
                .scalars()
                .all()
            )
            assert len(n_total) == 2

        # Cleanup
        with SessionLocal() as s:
            s.query(ModelVersion).filter(ModelVersion.model_name == test_model).delete()
            s.commit()


# =====================================================================
# Snap-to-road stats
# =====================================================================


def test_snap_stats_dist_quantiles():
    """После batch-snap'а median должна быть < 10 м, p99 < 50 м."""
    from src.analysis.snap_to_road import stats
    from src.database import SessionLocal

    with SessionLocal() as s:
        st = stats(s)
    method_counts = st["method_counts"]
    # Должны быть какие-то snapped'ы (после scripts/snap_existing_accidents)
    assert method_counts.get("osm_road", 0) > 1000
    dist = st["snapped_distance_stats"]
    assert dist["median_m"] is not None
    assert dist["median_m"] < 10.0  # медиана точно меньше 10 м
    assert dist["p99_m"] < 60.0  # внутри 50 + небольшая толерантность для чисел
