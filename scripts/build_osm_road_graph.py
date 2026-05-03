"""
Импорт OSM road graph для Приморского края в таблицу `osm_roads`.

Что делает:
1. Запрашивает Overpass API с фильтром highway IN (motorway, trunk,
   primary, secondary, tertiary, unclassified, residential, service)
   в BBox Приморского края (42.0..48.5 lat, 130.0..140.0 lon).
2. Парсит JSON-ответ → list of LineString (osm_way_id, highway_class,
   name, oneway, maxspeed, geom).
3. INSERT в osm_roads. Идемпотентно: TRUNCATE перед загрузкой
   (мы предполагаем full re-build при запуске; partial-update OSM
   через Diff не реализован — это редкая задача).

Источник: Overpass API public mirror (lz4.overpass-api.de). Для
Приморья запрос разбиваем на 4 тайла (BBox делим на 2x2), чтобы
не упереться в timeout/size-limit (по 25 МБ на ответ).

Запуск:
    python -m scripts.build_osm_road_graph

Запускается ОДИН РАЗ при первом deploy (или при необходимости обновить
road graph — раз в год). Потом snap-to-road и предсказания работают
с уже загруженной таблицей osm_roads.

Sanity-check после загрузки:
- Количество ребер по highway_class
- Гео-bbox имеющихся данных (должен лежать в Приморье)
- Спот-проверка: ST_DWithin от центра Владивостока (43.115, 131.886)
  на радиусе 1 км должен дать 100+ ребер
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database.session import SessionLocal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_osm_road_graph")


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Highway-классы для snap-to-road. Исключены footway/cycleway/path —
# на них ДТП быть не может (или это аномалия источника).
HIGHWAY_CLASSES = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "road",
]
HIGHWAY_REGEX = "|".join(HIGHWAY_CLASSES)

# Приморский край: 42.0..48.5 N, 130.0..140.0 E. Делим на 2x2 = 4 тайла.
BBOX = (42.0, 130.0, 48.5, 140.0)
TILES_LAT = 2
TILES_LON = 2

CACHE_DIR = ROOT / "data" / "raw" / "osm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TileBox:
    south: float
    west: float
    north: float
    east: float

    @property
    def name(self) -> str:
        return f"{self.south:.1f}_{self.west:.1f}_{self.north:.1f}_{self.east:.1f}"


def split_bbox(bbox: tuple[float, float, float, float], n_lat: int, n_lon: int) -> list[TileBox]:
    s, w, n, e = bbox
    dlat = (n - s) / n_lat
    dlon = (e - w) / n_lon
    tiles: list[TileBox] = []
    for i in range(n_lat):
        for j in range(n_lon):
            tiles.append(
                TileBox(
                    south=s + i * dlat,
                    west=w + j * dlon,
                    north=s + (i + 1) * dlat,
                    east=w + (j + 1) * dlon,
                )
            )
    return tiles


def fetch_tile(tile: TileBox, retries: int = 3) -> dict[str, Any]:
    cache_path = CACHE_DIR / f"primorye_roads_{tile.name}.json"
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1024 / 1024
        logger.info(f"  [cache] {cache_path.name} ({size_mb:.1f} MB)")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    query = f"""
    [out:json][timeout:600];
    (
      way["highway"~"^({HIGHWAY_REGEX})$"]
        ({tile.south},{tile.west},{tile.north},{tile.east});
    );
    out geom tags;
    """.strip()

    for mirror in OVERPASS_MIRRORS:
        for attempt in range(retries):
            try:
                logger.info(f"  → POST {mirror} tile={tile.name} attempt={attempt+1}")
                r = requests.post(mirror, data={"data": query}, timeout=900)
                if r.status_code == 200:
                    data = r.json()
                    n_ways = len(data.get("elements", []))
                    logger.info(f"  ← {n_ways} ways")
                    cache_path.write_text(json.dumps(data), encoding="utf-8")
                    # Overpass рекомендует ≥1 сек между запросами
                    time.sleep(1.5)
                    return data
                else:
                    logger.warning(f"  ! HTTP {r.status_code} (mirror={mirror})")
            except Exception as exc:
                logger.warning(f"  ! request failed: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Все Overpass-mirrors упали для tile {tile.name}")


def parse_ways(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags") or {}
        hw = tags.get("highway", "")
        oneway = tags.get("oneway") in ("yes", "true", "1")
        maxspeed_raw = tags.get("maxspeed", "").strip()
        try:
            maxspeed = int(maxspeed_raw) if maxspeed_raw and maxspeed_raw.isdigit() else None
        except ValueError:
            maxspeed = None
        # WKT LineString
        coords_str = ", ".join(f"{p['lon']} {p['lat']}" for p in geom)
        wkt = f"LINESTRING({coords_str})"
        rows.append(
            {
                "osm_way_id": el["id"],
                "highway_class": hw,
                "name": tags.get("name"),
                "oneway": oneway,
                "maxspeed": maxspeed,
                "wkt": wkt,
                "tags": tags,
            }
        )
    return rows


def truncate_table() -> None:
    with SessionLocal() as s:
        s.execute(text("TRUNCATE TABLE osm_roads RESTART IDENTITY"))
        s.commit()
    logger.info("osm_roads truncated")


def load_into_db(rows: list[dict[str, Any]], batch_size: int = 1000) -> int:
    """Батчевая вставка через ST_GeomFromText. Возвращает кол-во вставленных."""
    sql = text("""
        INSERT INTO osm_roads (osm_way_id, highway_class, name, oneway, maxspeed, geom, length_m, tags)
        VALUES (:osm_way_id, :highway_class, :name, :oneway, :maxspeed,
                ST_GeomFromText(:wkt, 4326),
                ST_Length(ST_GeomFromText(:wkt, 4326)::geography),
                CAST(:tags AS JSONB))
    """)
    inserted = 0
    with SessionLocal() as s:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            params = [
                {
                    "osm_way_id": r["osm_way_id"],
                    "highway_class": r["highway_class"],
                    "name": r["name"],
                    "oneway": r["oneway"],
                    "maxspeed": r["maxspeed"],
                    "wkt": r["wkt"],
                    "tags": json.dumps(r["tags"], ensure_ascii=False),
                }
                for r in batch
            ]
            s.execute(sql, params)
            s.commit()
            inserted += len(batch)
            if (i // batch_size) % 10 == 0:
                logger.info(f"  inserted {inserted}/{len(rows)}")
    return inserted


def sanity_check() -> None:
    with SessionLocal() as s:
        n = s.execute(text("SELECT COUNT(*) FROM osm_roads")).scalar() or 0
        logger.info(f"[sanity] osm_roads.count = {n}")
        if n == 0:
            raise RuntimeError("osm_roads пуста после загрузки")

        rows = s.execute(
            text("""
            SELECT highway_class, COUNT(*) AS cnt
            FROM osm_roads GROUP BY highway_class ORDER BY cnt DESC
        """)
        ).all()
        for hw, cnt in rows:
            logger.info(f"  [{hw:20s}] {cnt:>7d}")

        # Спот-проверка: ребра в радиусе 1 км от центра Владивостока (43.115, 131.886)
        n_vlad = (
            s.execute(
                text("""
            SELECT COUNT(*) FROM osm_roads
            WHERE ST_DWithin(geom::geography,
                             ST_SetSRID(ST_MakePoint(131.886, 43.115), 4326)::geography,
                             1000)
        """)
            ).scalar()
            or 0
        )
        logger.info(f"[sanity] ребер в 1 км от центра Владивостока: {n_vlad}")
        if n_vlad < 50:
            logger.warning("Подозрительно мало ребер вокруг центра Владивостока")


def main() -> None:
    logger.info("=" * 70)
    logger.info("OSM road graph builder for Приморский край")
    logger.info("=" * 70)

    tiles = split_bbox(BBOX, TILES_LAT, TILES_LON)
    logger.info(f"BBox: {BBOX}, разбит на {len(tiles)} тайлов")

    all_rows: list[dict[str, Any]] = []
    seen_way_ids: set[int] = set()

    for tile in tiles:
        logger.info(f"Tile {tile.name}:")
        data = fetch_tile(tile)
        parsed = parse_ways(data)
        # Дедупликация по osm_way_id (тайлы пересекаются)
        for row in parsed:
            if row["osm_way_id"] in seen_way_ids:
                continue
            seen_way_ids.add(row["osm_way_id"])
            all_rows.append(row)
        logger.info(f"  parsed={len(parsed)}, total_unique={len(all_rows)}")

    logger.info(f"Всего уникальных рёбер: {len(all_rows)}")

    truncate_table()
    inserted = load_into_db(all_rows)
    logger.info(f"Inserted: {inserted}")

    sanity_check()
    logger.info("ГОТОВО")


if __name__ == "__main__":
    main()
