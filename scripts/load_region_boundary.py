"""
Idempotent loader for region boundary polygons into PostGIS.

Что делает: читает GeoJSON-файл с MultiPolygon (по умолчанию — сухопутную
границу Приморского края), вычисляет точную площадь через UTM zone 52N
и сохраняет геометрию в таблицу `region_boundaries` через UPSERT по
уникальному `region_name` (повторный запуск обновляет, не дублирует).

Зачем нужно: даёт PostGIS-нативный фильтр `ST_Within(point, geom)` для
отсечения ДТП с координатами в воде (дефект #6 источника dtp-stat.ru)
и для всех последующих геопространственных операций — DBSCAN-кластеризации
очагов аварийности, агрегатов по районам и т.п.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import MultiPolygon, shape
from sqlalchemy import text

from src.database import SessionLocal


def load_region_boundary(
    geojson_path: Path,
    region_name: str,
    display_name: str,
    osm_relation_id: int = 151225,
) -> dict:
    """Загружает MultiPolygon из geojson_path в region_boundaries (UPSERT)."""
    logger.info("Reading GeoJSON: {}", geojson_path)
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    # GeoJSON может быть FeatureCollection (geopandas) или single Geometry
    if gj.get("type") == "FeatureCollection":
        features = gj.get("features", [])
        if len(features) != 1:
            logger.warning("Expected 1 feature in FeatureCollection, got {}", len(features))
        geom = shape(features[0]["geometry"])
    else:
        geom = shape(gj)

    if geom.geom_type == "Polygon":
        geom = MultiPolygon([geom])
    if geom.geom_type != "MultiPolygon":
        raise ValueError(f"Expected MultiPolygon (or Polygon), got {geom.geom_type!r}")

    n_subpoly = len(geom.geoms)
    logger.info("Geometry: MultiPolygon with {} sub-polygons", n_subpoly)

    # Площадь в км² через UTM zone 52N (минимум искажений для Приморья)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    area_km2 = float(gdf.to_crs("EPSG:32652").geometry.iloc[0].area / 1_000_000)
    logger.info("Area (UTM 52N): {:.2f} km^2", area_km2)

    # raw_meta — полный аудит-след источника для воспроизводимости
    raw_meta = {
        "osm_admin_relation_id": osm_relation_id,
        "osm_admin_source_url": (
            f"https://polygons.openstreetmap.fr/get_geojson.py" f"?id={osm_relation_id}&params=0"
        ),
        "osm_land_polygons_source_url": (
            "https://osmdata.openstreetmap.de/download/" "land-polygons-complete-4326.zip"
        ),
        "osm_land_polygons_last_modified": "Tue, 28 Apr 2026 03:29:39 GMT",
        "processing_method": (
            "geopandas.overlay(land_polygons, admin_boundary, how='intersection'); "
            "bbox pre-filter lon[130, 139.5] lat[42, 48.6]"
        ),
        "sub_polygons_count": n_subpoly,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }

    source_descr = (
        f"OSM admin_boundary relation {osm_relation_id} ∩ OSM land_polygons "
        f"(land_polygons last-modified {raw_meta['osm_land_polygons_last_modified']})"
    )

    sql = text(
        """
        INSERT INTO region_boundaries
            (region_name, display_name, source, geom, area_km2, raw_meta)
        VALUES (
            :region_name,
            :display_name,
            :source_descr,
            ST_Multi(ST_GeomFromText(:geom_wkt, 4326)),
            :area_km2,
            CAST(:raw_meta AS JSONB)
        )
        ON CONFLICT (region_name) DO UPDATE
        SET display_name = EXCLUDED.display_name,
            source       = EXCLUDED.source,
            geom         = EXCLUDED.geom,
            area_km2     = EXCLUDED.area_km2,
            raw_meta     = EXCLUDED.raw_meta,
            updated_at   = now()
        RETURNING
            id,
            region_name,
            area_km2,
            ST_GeometryType(geom) AS gtype,
            ST_NumGeometries(geom) AS n_subpoly,
            ST_SRID(geom) AS srid
        """
    )

    with SessionLocal() as session:
        row = session.execute(
            sql,
            {
                "region_name": region_name,
                "display_name": display_name,
                "source_descr": source_descr,
                "geom_wkt": geom.wkt,
                "area_km2": area_km2,
                "raw_meta": json.dumps(raw_meta, ensure_ascii=False),
            },
        ).one()
        session.commit()

    result = {
        "id": row.id,
        "region_name": row.region_name,
        "area_km2": float(row.area_km2),
        "gtype": row.gtype,
        "n_subpoly": row.n_subpoly,
        "srid": row.srid,
    }
    logger.info(
        "UPSERT ok: id={}, region_name={!r}, area_km2={:.2f}, " "gtype={}, sub_polys={}, srid={}",
        result["id"],
        result["region_name"],
        result["area_km2"],
        result["gtype"],
        result["n_subpoly"],
        result["srid"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-name", default="primorye_krai_land")
    parser.add_argument("--display-name", default="Приморский край (суша)")
    parser.add_argument(
        "--geojson-path",
        type=Path,
        default=Path("data/raw/primorye_land_polygons.geojson"),
    )
    parser.add_argument("--osm-relation-id", type=int, default=151225)
    args = parser.parse_args()

    load_region_boundary(
        geojson_path=args.geojson_path,
        region_name=args.region_name,
        display_name=args.display_name,
        osm_relation_id=args.osm_relation_id,
    )


if __name__ == "__main__":
    main()
