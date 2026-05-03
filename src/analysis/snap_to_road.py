"""
Snap-to-road: привязка raw-точек ДТП к ближайшему ребру OSM road graph
.

Зачем нужно: ~5-10% точек dtp-stat.ru снапаются на полигоны зданий
вместо точки на дороге (известная проблема геокодирования источника,
обнаружена в4 при drill-down). Privязка к ближайшей дороге
— чистое улучшение и для UX (точки на карте лежат на дорогах), и
для будущих ML-моделей с пространственными признаками.

Алгоритм для одной точки:
1. Найти кандидатов: рёбра osm_roads в радиусе MAX_SEARCH_RADIUS_M
   (50 м по плану — баланс между «снапнули случайно на параллельную
   дорогу» и «не снапнули на нужную»). KNN-ограничение через
   `<#>` distance оператор PostGIS / ORDER BY ST_Distance ... LIMIT.
2. Среди кандидатов — выбрать минимум по ST_Distance(geom::geography,
   point::geography). Это даёт metric distance в метрах.
3. Если расстояние ≤ 50 м → переносим точку через ST_ClosestPoint
   на эту LineString. snap_method = 'osm_road'.
4. Если > 50 м → snap_method = 'unchanged' (точка реально вне road
   graph: парковка, площадь, off-road ДТП).
5. Если в osm_roads вообще нет данных в этом регионе или geom NULL:
   snap_method = 'failed'.

Sanity-check (ручная калибровка cutoff'а 50 м): на 100 точек
случайных ДТП в Приморье — медиана snap_distance ~10-15 м, p90 ~30-50 м.
Точки > 50 м (≈5%) — реально off-road.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

MAX_SEARCH_RADIUS_M = 50.0
KNN_CANDIDATES = 10  # ограничение на kandid'оров для оценки
SNAP_METHOD_OSM = "osm_road"
SNAP_METHOD_UNCHANGED = "unchanged"
SNAP_METHOD_FAILED = "failed"


@dataclass
class SnapResult:
    snap_method: str
    snap_distance_m: Optional[float]
    snap_road_id: Optional[int]
    snapped_lon: Optional[float]
    snapped_lat: Optional[float]


def snap_point(
    session: Session, lon: float, lat: float, max_radius_m: float = MAX_SEARCH_RADIUS_M
) -> SnapResult:
    """Snap одной точки. Используется в Celery-задаче для новых записей."""
    sql = text("""
        WITH candidates AS (
            SELECT id, geom,
                   ST_Distance(geom::geography,
                               ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS dist_m
            FROM osm_roads
            ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
            LIMIT :knn
        ),
        best AS (
            SELECT id, geom, dist_m FROM candidates ORDER BY dist_m LIMIT 1
        )
        SELECT
            id,
            dist_m,
            ST_X(ST_ClosestPoint(geom, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)))::float AS sx,
            ST_Y(ST_ClosestPoint(geom, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)))::float AS sy
        FROM best
    """)
    row = session.execute(sql, {"lon": lon, "lat": lat, "knn": KNN_CANDIDATES}).one_or_none()
    if row is None:
        return SnapResult(SNAP_METHOD_FAILED, None, None, None, None)
    rid, dist_m, sx, sy = row
    if dist_m is None:
        return SnapResult(SNAP_METHOD_FAILED, None, None, None, None)
    if dist_m > max_radius_m:
        return SnapResult(SNAP_METHOD_UNCHANGED, float(dist_m), int(rid), None, None)
    return SnapResult(SNAP_METHOD_OSM, float(dist_m), int(rid), float(sx), float(sy))


def snap_batch_sql(
    session: Session, batch_size: int = 1000, max_radius_m: float = MAX_SEARCH_RADIUS_M
) -> dict:
    """Идемпотентно обрабатывает все ДТП с point IS NOT NULL и
    point_snapped IS NULL. Один большой UPDATE с CTE — быстрее, чем
    обходить 28k точек через Python.

    Возвращает dict со счётчиками: osm_road / unchanged.

    Логика:
    - Для каждой точки CROSS JOIN LATERAL находит KNN ближайших road'ов
      (по KNN=10), DISTINCT ON выбирает минимальный по дистанции.
    - Если осталось расстояние ≤ max_radius_m → snap_method='osm_road'
      + point_snapped = ST_ClosestPoint.
    - Иначе → snap_method='unchanged'.

    Batch не выставляет 'failed': при пустой osm_roads CROSS JOIN LATERAL
    не вернёт строк, и UPDATE просто пропустит цикл (записи останутся
    с snap_method=NULL). 'failed' существует только в одиночном
    snap_point() — там используется когда road graph пуст.
    """
    sql = text("""
        WITH targets AS (
            SELECT id, point
            FROM accidents
            WHERE point IS NOT NULL
              AND point_snapped IS NULL
              AND snap_method IS NULL
            LIMIT :batch_size
        ),
        nearest AS (
            SELECT
                t.id AS accident_id,
                r.id AS road_id,
                r.geom AS road_geom,
                t.point AS raw_point,
                ST_Distance(r.geom::geography, t.point::geography) AS dist_m
            FROM targets t
            CROSS JOIN LATERAL (
                SELECT id, geom
                FROM osm_roads
                ORDER BY geom <-> t.point
                LIMIT :knn
            ) r
        ),
        best AS (
            SELECT DISTINCT ON (accident_id)
                   accident_id, road_id, road_geom, raw_point, dist_m
            FROM nearest
            ORDER BY accident_id, dist_m
        )
        UPDATE accidents a SET
            snap_distance_m = b.dist_m,
            snap_road_id = CASE WHEN b.dist_m <= :max_r THEN b.road_id ELSE NULL END,
            snap_method = CASE WHEN b.dist_m <= :max_r THEN 'osm_road' ELSE 'unchanged' END,
            point_snapped = CASE WHEN b.dist_m <= :max_r
                                 THEN ST_ClosestPoint(b.road_geom, b.raw_point)
                                 ELSE NULL END
        FROM best b
        WHERE a.id = b.accident_id
        RETURNING a.id, a.snap_method, a.snap_distance_m
    """)
    rows = session.execute(
        sql, {"batch_size": batch_size, "knn": KNN_CANDIDATES, "max_r": max_radius_m}
    ).fetchall()
    session.commit()

    counters = {"osm_road": 0, "unchanged": 0}
    for _id, method, _d in rows:
        counters[method] = counters.get(method, 0) + 1
    return counters


def stats(session: Session) -> dict:
    """Распределение snap_method и snap_distance_m по всем accidents."""
    method_rows = session.execute(
        text("""
        SELECT snap_method, COUNT(*) AS cnt
        FROM accidents
        WHERE point IS NOT NULL
        GROUP BY snap_method
        ORDER BY cnt DESC
    """)
    ).all()
    methods = {(m or "NULL"): int(c) for m, c in method_rows}

    quantile_rows = session.execute(
        text("""
        SELECT
            COUNT(*) AS n,
            ROUND(AVG(snap_distance_m)::numeric, 2) AS mean_m,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY snap_distance_m) AS median_m,
            PERCENTILE_CONT(0.9)  WITHIN GROUP (ORDER BY snap_distance_m) AS p90_m,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY snap_distance_m) AS p99_m,
            MAX(snap_distance_m) AS max_m
        FROM accidents
        WHERE snap_method = 'osm_road'
    """)
    ).one()

    return {
        "method_counts": methods,
        "snapped_distance_stats": {
            "n": int(quantile_rows.n or 0),
            "mean_m": float(quantile_rows.mean_m) if quantile_rows.mean_m else None,
            "median_m": float(quantile_rows.median_m) if quantile_rows.median_m else None,
            "p90_m": float(quantile_rows.p90_m) if quantile_rows.p90_m else None,
            "p99_m": float(quantile_rows.p99_m) if quantile_rows.p99_m else None,
            "max_m": float(quantile_rows.max_m) if quantile_rows.max_m else None,
        },
    }
