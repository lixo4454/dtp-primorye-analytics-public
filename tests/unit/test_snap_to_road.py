"""Unit-тесты snap-to-road.

Покрывает чистую часть модуля ``src.analysis.snap_to_road`` без БД:
* dataclass ``SnapResult`` — корректность полей
* константы ``SNAP_METHOD_*`` — стабильные строки для статусов
* константы радиуса/KNN — sane defaults

Integration-проверка (реальная snap_point/snap_batch_sql на Postgres+PostGIS)
делается через ``tests/test_day17_admin.py`` и ручную SQL-проверку.
"""

from __future__ import annotations

from src.analysis.snap_to_road import (
    KNN_CANDIDATES,
    MAX_SEARCH_RADIUS_M,
    SNAP_METHOD_FAILED,
    SNAP_METHOD_OSM,
    SNAP_METHOD_UNCHANGED,
    SnapResult,
)

# =====================================================================
# Константы статусов — фиксированы как enum-like
# =====================================================================


def test_snap_method_constants():
    """Строки snap_method не должны меняться (попадают в БД и API)."""
    assert SNAP_METHOD_OSM == "osm_road"
    assert SNAP_METHOD_UNCHANGED == "unchanged"
    assert SNAP_METHOD_FAILED == "failed"


def test_max_search_radius_sane():
    """50 м — баланс между ошибочным снапом и непокрытием."""
    assert MAX_SEARCH_RADIUS_M == 50.0


def test_knn_candidates_positive_int():
    assert isinstance(KNN_CANDIDATES, int)
    assert KNN_CANDIDATES >= 1


# =====================================================================
# SnapResult dataclass
# =====================================================================


def test_snap_result_osm_road():
    """Результат успешного snap'а — все поля заполнены."""
    r = SnapResult(
        snap_method=SNAP_METHOD_OSM,
        snap_distance_m=12.5,
        snap_road_id=42,
        snapped_lon=131.95,
        snapped_lat=43.12,
    )
    assert r.snap_method == "osm_road"
    assert r.snap_distance_m == 12.5
    assert r.snap_road_id == 42
    assert r.snapped_lon == 131.95
    assert r.snapped_lat == 43.12


def test_snap_result_unchanged_no_snapped_coords():
    """Off-road кейс: snap_method=unchanged, координаты None."""
    r = SnapResult(
        snap_method=SNAP_METHOD_UNCHANGED,
        snap_distance_m=120.0,
        snap_road_id=99,
        snapped_lon=None,
        snapped_lat=None,
    )
    assert r.snap_method == "unchanged"
    assert r.snapped_lon is None
    assert r.snapped_lat is None


def test_snap_result_failed_all_none():
    """Failed: road graph пуст в регионе, всё None."""
    r = SnapResult(
        snap_method=SNAP_METHOD_FAILED,
        snap_distance_m=None,
        snap_road_id=None,
        snapped_lon=None,
        snapped_lat=None,
    )
    assert r.snap_method == "failed"
    assert r.snap_distance_m is None
    assert r.snap_road_id is None
