"""
Folium map for visual sanity-check of the region boundary filter.

Что делает: строит интерактивную HTML-карту Приморья с тремя слоями —
полигон сухопутной границы (зелёная заливка), 50 случайных «чистых» ДТП
внутри полигона (зелёные точки) и 50 «водных» ДТП дефекта #6 вне
полигона (красные точки с popup-адресом). Карта сохраняется в
`data/processed/region_boundary_check.html`.

Зачем нужно: глазами убедиться что L3-фильтр (OSM admin ∩ land_polygons)
действительно отсекает только точки в воде, не режет ДТП на пляжах,
набережных и в портах. Это последняя проверка перед фиксацией колонки
`is_in_region` и переходом к DBSCAN-кластеризации.
"""

from __future__ import annotations

import json
from pathlib import Path

import folium
from loguru import logger
from sqlalchemy import text

from src.database import SessionLocal


def main() -> None:
    out_path = Path("data/processed/region_boundary_check.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        # 1) Полигон границы (как GeoJSON) для слоя
        polygon_row = session.execute(
            text(
                "SELECT ST_AsGeoJSON(geom) AS gj, area_km2 "
                "FROM region_boundaries WHERE region_name = :rn"
            ),
            {"rn": "primorye_krai_land"},
        ).one()
        polygon_geojson = json.loads(polygon_row.gj)
        logger.info("Loaded boundary geometry, area_km2={}", polygon_row.area_km2)

        # 2) 50 случайных «чистых» ДТП (внутри полигона)
        inside = session.execute(
            text(
                """
                SELECT id, np, street, house, severity,
                       ST_Y(point) AS lat, ST_X(point) AS lon
                FROM accidents
                WHERE point IS NOT NULL
                  AND ST_Within(
                      point,
                      (SELECT geom FROM region_boundaries WHERE region_name = :rn)
                  )
                ORDER BY random()
                LIMIT 50
                """
            ),
            {"rn": "primorye_krai_land"},
        ).all()

        # 3) 50 «водных» ДТП дефекта #6 (внутри Приморье bbox, но не на суше)
        water = session.execute(
            text(
                """
                SELECT a.id, a.np, a.street, a.house, a.severity,
                       ST_Y(point) AS lat, ST_X(point) AS lon,
                       ROUND(
                           ST_Distance(
                               point::geography,
                               (SELECT geom FROM region_boundaries WHERE region_name = :rn)::geography
                           )::numeric, 0
                       ) AS dist_to_land_m
                FROM accidents a
                WHERE point IS NOT NULL
                  AND NOT ST_Within(
                      point,
                      (SELECT geom FROM region_boundaries WHERE region_name = :rn)
                  )
                  AND ST_X(point) BETWEEN 130 AND 140
                  AND ST_Y(point) BETWEEN 42 AND 49
                ORDER BY random()
                LIMIT 50
                """
            ),
            {"rn": "primorye_krai_land"},
        ).all()

    logger.info("Sampled inside={}, water={}", len(inside), len(water))

    # 4) Карта
    # OpenStreetMap нейтральный по attribution. CartoDB Positron как
    # раньше показывал украинский флаг в © — для гос-проекта не
    # подходит. При проблеме 403 от OSM-тайлов открыть HTML через
    # http-сервер (`python -m http.server`), а не file://.
    m = folium.Map(
        location=[44.5, 134.5],  # центр Приморья
        zoom_start=7,
        tiles="OpenStreetMap",
    )

    folium.GeoJson(
        polygon_geojson,
        name="Сухопутная граница Приморья (L3-фильтр)",
        style_function=lambda _: {
            "fillColor": "#2ecc71",
            "color": "#27ae60",
            "weight": 1,
            "fillOpacity": 0.15,
        },
    ).add_to(m)

    fg_inside = folium.FeatureGroup(name="50 ДТП внутри полигона (чистые)")
    for r in inside:
        addr = f"{r.np or ''} {r.street or ''} {r.house or ''}".strip() or "—"
        folium.CircleMarker(
            location=[r.lat, r.lon],
            radius=4,
            color="#27ae60",
            fill=True,
            fill_color="#2ecc71",
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(
                f"<b>id={r.id}</b><br>{addr}<br>severity={r.severity}<br>"
                f"({r.lat:.4f}, {r.lon:.4f})",
                max_width=300,
            ),
        ).add_to(fg_inside)
    fg_inside.add_to(m)

    fg_water = folium.FeatureGroup(name="50 ДТП в воде (дефект #6)")
    for r in water:
        addr = f"{r.np or ''} {r.street or ''} {r.house or ''}".strip() or "—"
        folium.CircleMarker(
            location=[r.lat, r.lon],
            radius=5,
            color="#c0392b",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(
                f"<b>id={r.id}</b><br>{addr}<br>severity={r.severity}<br>"
                f"({r.lat:.4f}, {r.lon:.4f})<br>"
                f"<i>До берега: {r.dist_to_land_m} м</i>",
                max_width=300,
            ),
        ).add_to(fg_water)
    fg_water.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(out_path)
    size_kb = out_path.stat().st_size / 1024
    logger.info("Saved: {} ({:.1f} kB)", out_path, size_kb)


if __name__ == "__main__":
    main()
