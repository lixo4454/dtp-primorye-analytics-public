"""
Folium map of DBSCAN accident hotspots in Primorye.

Что делает: строит интерактивную HTML-карту с тремя слоями —
сухопутный полигон Приморья (зелёная подложка), топ-30 «горячих»
очагов аварийности (яркие круги, радиус ∝ √n_points, цвет ∝
доле смертельных ДТП) и все остальные кластеры (бледные круги).
По клику на круг — popup с топ-5 улицами, типами ДТП и
распределением severity. Карта сохраняется в
`data/processed/map_hotspots.html`.

Зачем нужно: главный визуальный артефакт по очагам:
позволяет сразу увидеть «где в Приморье опаснее всего» и проверить
что найденные очаги совпадают с известными местами (Золотой мост,
Артёмовская объездная, Гоголя, Снеговая, центр Уссурийска,
Находкинский проспект).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
from folium.plugins import MarkerCluster
from loguru import logger
from sqlalchemy import text

from src.database import SessionLocal


# Цветовая шкала по pct_dead (доля смертельных ДТП в кластере)
def color_by_pct_dead(pct: float) -> str:
    """Зелёный → жёлтый → оранжевый → красный → тёмно-красный."""
    if pct < 0.03:
        return "#27ae60"  # зелёный
    if pct < 0.06:
        return "#f1c40f"  # жёлтый
    if pct < 0.10:
        return "#e67e22"  # оранжевый
    if pct < 0.15:
        return "#e74c3c"  # красный
    return "#8e44ad"  # фиолетовый = редко, но смертельно


def radius_by_n_points(n: int, n_max: int) -> float:
    """Радиус маркера в пикселях — sqrt-шкала чтобы не было гигантов."""
    # min 6 px, max 30 px
    if n_max <= 0:
        return 8.0
    import math

    return 6.0 + 24.0 * math.sqrt(n / n_max)


def build_popup_html(props: dict, rank: int | None) -> str:
    """HTML-блок для popup с метаданными кластера."""
    sev = props.get("severity_distribution", {})
    sev_str = " | ".join(f"{k}: {v}" for k, v in sorted(sev.items(), key=lambda kv: -kv[1]))
    em_str = "<br>".join(f"&nbsp;&nbsp;{k}: {v}" for k, v in props.get("top_em_types", []))
    streets_str = "<br>".join(f"&nbsp;&nbsp;{k}: {v}" for k, v in props.get("top_streets", []))
    np_str = ", ".join(f"{k} ({v})" for k, v in props.get("top_np", []))

    rank_block = f"<b>Топ-#{rank}</b><br>" if rank is not None else ""

    return f"""
    <div style="font-family: sans-serif; font-size: 12px; min-width: 320px;">
        {rank_block}
        <b>Кластер #{props['cluster_id']}</b>
        &nbsp;|&nbsp; <b>{props['n_points']}</b> ДТП
        &nbsp;|&nbsp; %dead = <b>{props['pct_dead']:.2%}</b>
        <br>
        Радиус кластера: {props['radius_meters']:.0f} м,
        медиана дистанций: {props['median_distance_meters']:.0f} м
        <br>
        НП: {np_str}
        <br>
        <hr style="margin: 4px 0;">
        <b>Топ улиц:</b><br>{streets_str}
        <hr style="margin: 4px 0;">
        <b>Топ типов ДТП:</b><br>{em_str}
        <hr style="margin: 4px 0;">
        <b>Severity:</b> {sev_str}
    </div>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geojson",
        type=Path,
        default=Path("data/processed/hotspots_eps100.geojson"),
        help="Файл с городскими очагами (eps=100м) — точные перекрёстки",
    )
    parser.add_argument(
        "--highway-geojson",
        type=Path,
        default=Path("data/processed/hotspots_highway.geojson"),
        help="Файл с трассовыми очагами (eps=500м, min_samples=5) — для покрытия всего Приморья",
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/map_hotspots.html"),
    )
    parser.add_argument("--region-name", default="primorye_krai_land")
    args = parser.parse_args()

    # Загрузка GeoJSON-результатов DBSCAN
    if not args.geojson.exists():
        raise FileNotFoundError(
            f"{args.geojson} not found — run scripts/dbscan_sensitivity_analysis.py first"
        )
    geo = json.loads(args.geojson.read_text(encoding="utf-8"))
    features = geo["features"]
    metadata = geo.get("metadata", {})
    logger.info(
        "Loaded {} clusters from {} (eps={}, min_samples={})",
        len(features),
        args.geojson,
        metadata.get("eps_meters"),
        metadata.get("min_samples"),
    )

    # Сортируем по n_points DESC и делим на топ-N и остальные
    features.sort(key=lambda f: -f["properties"]["n_points"])
    top = features[: args.top_n]
    rest = features[args.top_n :]
    n_max = top[0]["properties"]["n_points"] if top else 0

    # Загрузка полигона границы из БД
    with SessionLocal() as session:
        polygon_row = session.execute(
            text("SELECT ST_AsGeoJSON(geom) AS gj FROM region_boundaries WHERE region_name = :rn"),
            {"rn": args.region_name},
        ).one()
        polygon_geojson = json.loads(polygon_row.gj)

    # Карта на OpenStreetMap — нейтральный attribution, без флагов
    # в © (CartoDB Positron как раньше показывал украинский флаг).
    m = folium.Map(
        location=[44.5, 134.5],
        zoom_start=7,
        tiles="OpenStreetMap",
    )

    # Слой 1 — полигон Приморья
    folium.GeoJson(
        polygon_geojson,
        name="Сухопутная граница Приморья",
        style_function=lambda _: {
            "fillColor": "#2ecc71",
            "color": "#27ae60",
            "weight": 1,
            "fillOpacity": 0.08,
        },
    ).add_to(m)

    # Слой 2 — топ-N очагов (яркие)
    fg_top = folium.FeatureGroup(name=f"ТОП-{args.top_n} очагов аварийности")
    for rank, feat in enumerate(top, start=1):
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        color = color_by_pct_dead(props["pct_dead"])
        radius = radius_by_n_points(props["n_points"], n_max)
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#2c3e50",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(build_popup_html(props, rank=rank), max_width=400),
            tooltip=f"#{rank}: {props['n_points']} ДТП, %dead={props['pct_dead']:.1%}",
        ).add_to(fg_top)
    fg_top.add_to(m)

    # Слой 3 — остальные кластеры (бледнее)
    fg_rest = folium.FeatureGroup(name="Остальные городские кластеры (eps=100м)", show=False)
    for feat in rest:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        color = color_by_pct_dead(props["pct_dead"])
        radius = radius_by_n_points(props["n_points"], n_max)
        folium.CircleMarker(
            location=[lat, lon],
            radius=max(4.0, radius * 0.7),
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,
            popup=folium.Popup(build_popup_html(props, rank=None), max_width=400),
            tooltip=f"{props['n_points']} ДТП",
        ).add_to(fg_rest)
    fg_rest.add_to(m)

    # Слой 4 — Трассовые очаги (eps=500м, min_samples=5) — покрытие всего Приморья
    if args.highway_geojson.exists():
        hw_geo = json.loads(args.highway_geojson.read_text(encoding="utf-8"))
        hw_features = hw_geo["features"]
        hw_metadata = hw_geo.get("metadata", {})
        hw_features.sort(key=lambda f: -f["properties"]["n_points"])
        # Top-50 трассовых, исключая дубликаты с городскими (центроид < 1км от любого top-30 городского)
        existing_centers = [
            (f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1]) for f in top
        ]
        hw_unique = []
        for hwf in hw_features[:80]:
            hlon, hlat = hwf["geometry"]["coordinates"]
            # фильтрация — не дублировать городские очаги
            too_close = False
            for clon, clat in existing_centers:
                # ~1 км в градусах: 0.01° lat, 0.014° lon на 43°N
                if abs(hlon - clon) < 0.014 and abs(hlat - clat) < 0.01:
                    too_close = True
                    break
            if not too_close and hwf["properties"]["n_points"] >= 8:
                hw_unique.append(hwf)
            if len(hw_unique) >= 30:
                break

        fg_hw = folium.FeatureGroup(
            name=f"ТОП-30 трассовых очагов (eps={hw_metadata.get('eps_meters'):.0f}м, мягкий)"
        )
        hw_n_max = max(f["properties"]["n_points"] for f in hw_unique) if hw_unique else 1
        for rank, feat in enumerate(hw_unique, start=1):
            props = feat["properties"]
            lon, lat = feat["geometry"]["coordinates"]
            color = color_by_pct_dead(props["pct_dead"])
            radius = radius_by_n_points(props["n_points"], hw_n_max) * 0.9
            folium.RegularPolygonMarker(
                location=[lat, lon],
                number_of_sides=4,  # ромб — отличие от круга городских
                rotation=45,
                radius=radius,
                color="#34495e",
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(build_popup_html(props, rank=rank), max_width=400),
                tooltip=f"Трасса #{rank}: {props['n_points']} ДТП, %dead={props['pct_dead']:.1%}",
            ).add_to(fg_hw)
        fg_hw.add_to(m)
        logger.info(
            "Highway layer: {} clusters added (filtered from top-80 in {})",
            len(hw_unique),
            args.highway_geojson,
        )

    # Легенда (HTML overlay)
    legend_html = f"""
    <div style="
        position: fixed; bottom: 20px; left: 20px; z-index: 9999;
        background: white; padding: 10px 14px; border: 1px solid #888;
        border-radius: 6px; font-family: sans-serif; font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
      <b>Цвет — доля смертельных ДТП в кластере</b><br>
      <span style="color: #27ae60;">●</span> &lt; 3% (норма города)<br>
      <span style="color: #f1c40f;">●</span> 3 – 6% (среднее)<br>
      <span style="color: #e67e22;">●</span> 6 – 10% (повышенный риск)<br>
      <span style="color: #e74c3c;">●</span> 10 – 15% (опасный участок)<br>
      <span style="color: #8e44ad;">●</span> ≥ 15% (катастрофический)<br>
      <hr style="margin:6px 0;">
      <b>Форма маркера:</b><br>
      ● Городские очаги (eps={metadata.get('eps_meters'):.0f}м) — точные перекрёстки<br>
      ◆ Трассовые очаги (eps=500м) — для покрытия всего Приморья<br>
      <hr style="margin:6px 0;">
      Размер ∝ √(N ДТП), top-{args.top_n} городских + top-30 трассовых
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.output))
    logger.info(
        "Saved: {} ({:.1f} kB)",
        args.output,
        args.output.stat().st_size / 1024,
    )

    # Краткая сводка для лога
    logger.info("=== TOP-{} hotspots map summary ===", args.top_n)
    for rank, feat in enumerate(top[:10], start=1):
        p = feat["properties"]
        top_street = p["top_streets"][0][0] if p["top_streets"] else "—"
        logger.info(
            "  #{:2}  n={:>4}  %dead={:5.2%}  ({:.4f},{:.4f})  {}",
            rank,
            p["n_points"],
            p["pct_dead"],
            feat["geometry"]["coordinates"][0],
            feat["geometry"]["coordinates"][1],
            top_street,
        )


if __name__ == "__main__":
    main()
