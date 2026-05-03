"""HotspotProfile — единый объект для rule engine рекомендаций.

Профиль агрегирует характеристики выборки ДТП в заданной географической
области (DBSCAN-кластер ИЛИ произвольный круг радиуса r вокруг точки).
Все 18 правил из ``src.analysis.recommendations`` принимают на вход
именно ``HotspotProfile``.

Два факторий-метода:

* :py:meth:`HotspotProfile.from_dbscan` — из словаря очага в
  ``data/processed/hotspots_summary.json`` (артефакт).
* :py:meth:`HotspotProfile.from_dynamic_radius` — из произвольной
  точки + радиуса; считает PostGIS ``ST_DWithin``-запрос.

Поля профиля построены так, чтобы триггеры правил сводились к
проверке нескольких полей (``top_em_type``, ``pct_dead``,
``is_highway``, ``has_night_dominant`` и т.д.) — это упрощает
тесты и делает rule engine легко расширяемым.

Замечания о значениях:

* ``is_highway`` извлекаем из ``roads IS NOT NULL AND TRIM(roads) <> ''``
  (та же логика, что в обучении CatBoost — см.
  ``src.ml.severity_classifier``).
* ``is_in_city`` — комплементарно ``is_highway`` (упрощение: если ДТП
  не на трассе, оно либо в населённом пункте, либо на off-grid).
* ``has_night_dominant`` — доля ``light_type`` категорий с подстрокой
  «темное время суток» либо «Сумерки» >= 50 %.
* ``has_winter_spike`` — доля ДТП в дек-фев >= 35 % (норма для
  4 месяцев из 12 = ~33 %, выше — спайк).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Em-types из БД (см. SQL: SELECT em_type, COUNT(*) FROM accidents)
EM_TYPE_PED = "Наезд на пешехода"
EM_TYPE_COLLISION = "Столкновение"
EM_TYPE_ROADSIDE = "Съезд с дороги"
EM_TYPE_ROLLOVER = "Опрокидывание"
EM_TYPE_OBSTACLE = "Наезд на препятствие"
EM_TYPE_PARKED = "Наезд на стоящее ТС"
EM_TYPE_BIKE = "Наезд на велосипедиста"

# light_type в БД (см. SQL)
LIGHT_DAY = "Светлое время суток"
LIGHT_NIGHT_LIT = "В темное время суток, освещение включено"
LIGHT_NIGHT_NO_LIGHT = "В темное время суток, освещение отсутствует"
LIGHT_NIGHT_NOT_ON = "В темное время суток, освещение не включено"
LIGHT_DUSK = "Сумерки"

# traffic_area_state в БД
STATE_DRY = "Сухое"
STATE_WET = "Мокрое"
STATE_SNOW_PACK = "Со снежным накатом"
STATE_SNOWY = "Заснеженное"
STATE_ICE = "Гололедица"
STATE_TREATED = "Обработанное противогололедными материалами"

NIGHT_LIGHT_TYPES = {LIGHT_NIGHT_LIT, LIGHT_NIGHT_NO_LIGHT, LIGHT_NIGHT_NOT_ON, LIGHT_DUSK}
DARK_NO_LIGHT_TYPES = {LIGHT_NIGHT_NO_LIGHT, LIGHT_NIGHT_NOT_ON}
WINTER_SLIPPERY_STATES = {STATE_SNOW_PACK, STATE_SNOWY, STATE_ICE}


@dataclass
class HotspotProfile:
    """Агрегатные характеристики набора ДТП в географической области.

    Парные поля для топ-N: ``top_em_type`` (str) — самый частый,
    ``top_em_types`` (list[tuple[str, int]]) — топ-3 для отображения.
    """

    n_points: int
    radius_meters: float
    pct_dead: float
    pct_severe_or_dead: float

    top_em_type: str | None
    top_em_types: list[tuple[str, int]] = field(default_factory=list)
    top_np: str | None = None

    dominant_light_type: str | None = None
    dominant_state: str | None = None

    # Дериваты — чаще используются в правилах, чем raw mt_rate
    is_highway: bool = False
    is_in_city: bool = False

    has_night_dominant: bool = False
    has_winter_spike: bool = False

    # Координаты центроида (для попапа)
    centroid_lat: float | None = None
    centroid_lon: float | None = None

    # Происхождение профиля
    source: str = "unknown"  # "dbscan" | "dynamic_radius"
    cluster_id: int | None = None
    radius_query_m: int | None = None

    # ----------- Factory: DBSCAN-кластер -----------------------------

    @classmethod
    def from_dbscan(cls, hotspot: dict[str, Any]) -> "HotspotProfile":
        """Из словаря очага в ``hotspots_summary.json`` (артефакт).

        Очаг содержит:

        - ``n_points``, ``centroid_lat``, ``centroid_lon``,
          ``radius_meters``
        - ``pct_dead``, ``pct_severe_or_dead``
        - ``top_em_types`` — список ``[em_type, count]``
        - ``top_np`` — список ``[np, count]``
        - ``severity_distribution`` — словарь severity → count

        Поля, которых в JSON нет (``light_type``,
        ``traffic_area_state``, ``is_highway``, сезонность), приходится
        восстанавливать через дополнительный SQL — но в это не
        делалось. Поэтому здесь они инициализируются нейтральными
        значениями ("неизвестно"), а у фабрики `from_dynamic_radius`
        они полные. Rule engine безопасно проверяет ``None``-поля
        через :py:meth:`BaseRule.applies_to`.
        """
        n_points = int(hotspot.get("n_points", 0))
        top_em_pairs = [(t[0], int(t[1])) for t in hotspot.get("top_em_types", [])][:5]
        top_em_type = top_em_pairs[0][0] if top_em_pairs else None
        top_np_pairs = [(t[0], int(t[1])) for t in hotspot.get("top_np", [])][:5]
        top_np = top_np_pairs[0][0] if top_np_pairs else None

        # is_highway эвристически: если top_np содержит «-я трасса» / «А-188»
        # либо top_np пустой, считаем highway. Иначе — city.
        # Для DBSCAN-кластеров «городские» очаги почти всегда имеют top_np
        # = «г Владивосток» / «г Находка» — этого достаточно для эвристики.
        is_in_city = bool(top_np and top_np.lower().startswith(("г ", "пгт", "с ", "п ")))
        is_highway = not is_in_city

        return cls(
            n_points=n_points,
            radius_meters=float(hotspot.get("radius_meters", 0.0)),
            pct_dead=float(hotspot.get("pct_dead", 0.0)),
            pct_severe_or_dead=float(hotspot.get("pct_severe_or_dead", 0.0)),
            top_em_type=top_em_type,
            top_em_types=top_em_pairs,
            top_np=top_np,
            dominant_light_type=None,  # JSON-артефакт это не содержит
            dominant_state=None,
            is_highway=is_highway,
            is_in_city=is_in_city,
            has_night_dominant=False,
            has_winter_spike=False,
            centroid_lat=hotspot.get("centroid_lat"),
            centroid_lon=hotspot.get("centroid_lon"),
            source="dbscan",
            cluster_id=hotspot.get("cluster_id"),
            radius_query_m=None,
        )

    # ----------- Factory: dynamic radius (PostGIS ST_DWithin) ---------

    @classmethod
    async def from_dynamic_radius(
        cls,
        lat: float,
        lon: float,
        radius_m: int,
        session: Any,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> "HotspotProfile":
        """Из PostGIS-запроса вокруг произвольной (lat, lon) с опц. year-фильтром.

        Использует GIST-индекс ``idx_accidents_point`` — ``ST_DWithin``
        на geography с радиусом 30..1000 м занимает ~10-30 мс
        (см. ``EXPLAIN ANALYZE`` — должен быть Index Scan, не Seq).

        ``year_min/year_max`` фильтруют ДТП по году datetime'а
        (включительно с обеих сторон). None = без фильтра.

        Если в радиусе **<10 ДТП**, возвращается профиль с ``n_points=N`` и
        дефолтными значениями остальных полей. Rule engine на стороне
        API увидит ``n_points < 10`` и не запустится — UI покажет
        «увеличь радиус».
        """
        from sqlalchemy import text

        params: dict = {"lat": lat, "lon": lon, "radius_m": radius_m}
        year_clause = ""
        if year_min is not None:
            year_clause += " AND EXTRACT(year FROM datetime) >= :y_min"
            params["y_min"] = year_min
        if year_max is not None:
            year_clause += " AND EXTRACT(year FROM datetime) <= :y_max"
            params["y_max"] = year_max

        sql = text(
            f"""
            SELECT
                COUNT(*) AS n_points,
                SUM(CASE WHEN severity = 'dead' THEN 1 ELSE 0 END)::float
                    / NULLIF(COUNT(*), 0) AS pct_dead,
                SUM(CASE WHEN severity IN ('dead','severe','severe_multiple') THEN 1 ELSE 0 END)::float
                    / NULLIF(COUNT(*), 0) AS pct_severe_or_dead,
                MODE() WITHIN GROUP (ORDER BY em_type) FILTER (WHERE em_type IS NOT NULL) AS top_em_type,
                MODE() WITHIN GROUP (ORDER BY light_type) FILTER (WHERE light_type IS NOT NULL) AS dominant_light,
                MODE() WITHIN GROUP (ORDER BY traffic_area_state) FILTER (WHERE traffic_area_state IS NOT NULL) AS dominant_state,
                MODE() WITHIN GROUP (ORDER BY np) FILTER (WHERE np IS NOT NULL AND np <> '') AS top_np,
                SUM(CASE WHEN roads IS NOT NULL AND TRIM(roads) <> '' THEN 1 ELSE 0 END)::float
                    / NULLIF(COUNT(*), 0) AS pct_highway,
                SUM(CASE WHEN light_type IN (
                    'В темное время суток, освещение включено',
                    'В темное время суток, освещение отсутствует',
                    'В темное время суток, освещение не включено',
                    'Сумерки'
                ) THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS pct_night,
                SUM(CASE WHEN EXTRACT(MONTH FROM datetime) IN (12, 1, 2)
                         THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS pct_winter
            FROM accidents
            WHERE ST_DWithin(
                point::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                :radius_m
            )
              AND is_in_region = TRUE
              AND severity IS NOT NULL
              {year_clause}
            """
        )

        result = await session.execute(sql, params)
        row = result.mappings().one()
        n_points = int(row["n_points"] or 0)

        # Для top_em_types топ-3 — отдельный запрос (MODE даёт только 1)
        top_em_types: list[tuple[str, int]] = []
        if n_points > 0:
            sql_top = text(
                f"""
                SELECT em_type, COUNT(*) AS n
                FROM accidents
                WHERE ST_DWithin(
                    point::geography,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                    :radius_m
                )
                  AND is_in_region = TRUE
                  AND em_type IS NOT NULL
                  {year_clause}
                GROUP BY em_type
                ORDER BY n DESC
                LIMIT 3
                """
            )
            r2 = await session.execute(sql_top, params)
            top_em_types = [(rec["em_type"], int(rec["n"])) for rec in r2.mappings()]

        pct_highway = float(row["pct_highway"] or 0.0)
        pct_night = float(row["pct_night"] or 0.0)
        pct_winter = float(row["pct_winter"] or 0.0)

        is_highway = pct_highway >= 0.5
        is_in_city = pct_highway < 0.5

        return cls(
            n_points=n_points,
            radius_meters=float(radius_m),
            pct_dead=float(row["pct_dead"] or 0.0),
            pct_severe_or_dead=float(row["pct_severe_or_dead"] or 0.0),
            top_em_type=row["top_em_type"],
            top_em_types=top_em_types,
            top_np=row["top_np"],
            dominant_light_type=row["dominant_light"],
            dominant_state=row["dominant_state"],
            is_highway=is_highway,
            is_in_city=is_in_city,
            has_night_dominant=pct_night >= 0.5,
            has_winter_spike=pct_winter >= 0.35,
            centroid_lat=lat,
            centroid_lon=lon,
            source="dynamic_radius",
            cluster_id=None,
            radius_query_m=radius_m,
        )

    # ----------- Утилиты для тестов / попапов -------------------------

    def has_em_type(self, em_type: str) -> bool:
        """True если ``em_type`` входит в топ-3 (либо равен top_em_type)."""
        if self.top_em_type == em_type:
            return True
        return any(t[0] == em_type for t in self.top_em_types)

    @classmethod
    def from_em_counts(
        cls,
        em_counts: Counter | dict[str, int],
        n_dead: int,
        n_severe_or_dead: int,
        *,
        is_highway: bool = False,
        is_in_city: bool = True,
        dominant_light_type: str | None = None,
        dominant_state: str | None = None,
        has_night_dominant: bool = False,
        has_winter_spike: bool = False,
        radius_meters: float = 200.0,
        top_np: str | None = None,
    ) -> "HotspotProfile":
        """Утилита для тестов — собрать профиль из счётчика em_type."""
        counter = Counter(em_counts) if not isinstance(em_counts, Counter) else em_counts
        n = sum(counter.values())
        top_pairs = counter.most_common(5)
        top = top_pairs[0][0] if top_pairs else None
        return cls(
            n_points=n,
            radius_meters=radius_meters,
            pct_dead=(n_dead / n) if n else 0.0,
            pct_severe_or_dead=(n_severe_or_dead / n) if n else 0.0,
            top_em_type=top,
            top_em_types=[(t[0], int(t[1])) for t in top_pairs],
            top_np=top_np,
            dominant_light_type=dominant_light_type,
            dominant_state=dominant_state,
            is_highway=is_highway,
            is_in_city=is_in_city,
            has_night_dominant=has_night_dominant,
            has_winter_spike=has_winter_spike,
            source="test",
        )
