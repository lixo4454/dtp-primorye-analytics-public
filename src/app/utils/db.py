"""Прямое подключение к Postgres из Streamlit.

Используется для запросов, которые не имеет смысла гонять через FastAPI:
тяжёлые SQL-агрегаты (cross-tab по em_type × year × severity, ~800 строк).
JSON-roundtrip через API дороже чем SQL-агрегация в Postgres.

Engine — singleton через ``@st.cache_resource``; результаты — DataFrame
через ``@st.cache_data`` (TTL 1 час).

URL строим inline, не через ``src.database.session.build_database_url`` —
импорт ``src.database`` триггерит ``__init__.py``, тянущий ORM-модели и
``geoalchemy2`` + ``shapely``. Дашборду они не нужны (мы делаем raw SQL
без PostGIS-объектов), и Dockerfile.app получает ~30 МБ меньше зависимостей.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()


def _build_db_url() -> str:
    """Конструктор URL для синхронного драйвера psycopg.

    Дублирует логику ``src.database.session.build_database_url`` сознательно
    (см. модульный docstring) — на случай дрейфа стоит держать оба в синке.
    """
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")
    if not all([user, password, db]):
        raise RuntimeError(
            "Не заданы POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB — "
            "проверь .env или environment в docker-compose"
        )
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    """SQLAlchemy Engine один на Streamlit-процесс."""
    return create_engine(
        _build_db_url(),
        pool_size=2,
        max_overflow=2,
        pool_pre_ping=True,
        echo=False,
    )


@st.cache_data(ttl=3600, show_spinner="Загрузка 28 020 координат ДТП...")
def get_accident_coords(
    snap_to_road: bool = False,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    """Координаты ДТП в регионе для drill-down карты.

    ~28 020 строк без фильтра по годам, ~50-80 мс в SQL, после кеша
    мгновенно. Возвращает только то что нужно карте: id, lat, lon,
    severity, year, snap_method, datetime, em_type, lost_amount,
    suffer_amount, np, street, house — для popup'а единичной аварии.

    PostGIS ``geom`` извлекаем через ``ST_X`` / ``ST_Y``. Folium ждёт (lat, lon).

    snap_to_road=True — отдаёт snapped координаты для тех ДТП, где
    snap_method='osm_road' (точка перенесена на ближайшее ребро OSM road
    graph). Для unchanged/failed/NULL — raw point: off-road ДТП всё
    равно должны быть видны.

    year_min/year_max — отрезают по году datetime'а (включительно с
    обеих сторон). None = без фильтра.

    Returns
    -------
    DataFrame: id, lat, lon, severity, year, snap_method, datetime,
    em_type, lost_amount, suffer_amount, np, street, house.
    """
    coord_expr = "COALESCE(point_snapped, point)" if snap_to_road else "point"
    where_year = ""
    params: dict = {}
    if year_min is not None:
        where_year += " AND EXTRACT(year FROM datetime) >= :y_min"
        params["y_min"] = year_min
    if year_max is not None:
        where_year += " AND EXTRACT(year FROM datetime) <= :y_max"
        params["y_max"] = year_max

    sql = text(
        f"""
        SELECT id,
               ST_X({coord_expr})::double precision AS lon,
               ST_Y({coord_expr})::double precision AS lat,
               severity,
               EXTRACT(year FROM datetime)::int AS year,
               snap_method,
               datetime,
               em_type,
               lost_amount,
               suffer_amount,
               np,
               street,
               house
        FROM accidents
        WHERE is_in_region = TRUE
          AND point IS NOT NULL
          AND severity IS NOT NULL
          {where_year}
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    df["year"] = df["year"].astype(int)
    df["id"] = df["id"].astype(int)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_heatmap_grid(precision: int = 3) -> list[list[float]]:
    """28 020 точек ДТП → weighted-grid для HeatMap.

    heatmap.js на сильном zoom пересчитывает screen-projection каждой
    точки при каждом move/zoom. На 28k точек это и есть источник лагов
    (~200-500 мс per frame). Решение — агрегация в ячейки сетки
    с весом = количество ДТП в ячейке.

    precision=3 ⇒ округление до 0.001° ≈ 110 м на lat 43°. Городские
    ДТП плотно бьются в ячейки — 28k → ~5-7k bins. На фоне blur=18 px
    разница в визуале не заметна (blur всё равно «размазывает» кружки),
    но pan/zoom становится плавным.

    Возвращает list[[lat, lon, weight]] — формат, который Folium
    HeatMap принимает напрямую.
    """
    df = get_accident_coords()
    g = (
        df.assign(
            lat_bin=df["lat"].round(precision),
            lon_bin=df["lon"].round(precision),
        )
        .groupby(["lat_bin", "lon_bin"])
        .size()
        .reset_index(name="weight")
    )
    return g[["lat_bin", "lon_bin", "weight"]].astype(float).values.tolist()


@st.cache_data(ttl=3600, show_spinner=False)
def get_monthly_history() -> pd.DataFrame:
    """Месячная история ДТП 2015..текущий месяц для Prophet-визуализации.

    Используется на странице forecast: Plotly рисует историю слева от
    разделительной линии, прогноз справа. Источник для actual — БД,
    источник для forecast — FastAPI ``/forecast/monthly``.

    Returns
    -------
    DataFrame с колонками: ds (datetime, начало месяца), y (int).
    Около 130 строк (11 лет × 12 месяцев + текущие).
    """
    sql = text(
        """
        SELECT DATE_TRUNC('month', datetime)::date AS ds,
               COUNT(*)::int AS y
        FROM accidents
        WHERE datetime IS NOT NULL
        GROUP BY ds
        ORDER BY ds
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_year_range() -> tuple[int, int]:
    """Min/max года в БД (для slider'а на stats-page)."""
    sql = text(
        "SELECT MIN(EXTRACT(year FROM datetime))::int AS y_min, "
        "MAX(EXTRACT(year FROM datetime))::int AS y_max "
        "FROM accidents WHERE is_in_region AND datetime IS NOT NULL"
    )
    with get_engine().connect() as conn:
        r = conn.execute(sql).one()
    return int(r.y_min), int(r.y_max)


@st.cache_data(ttl=3600, show_spinner=False)
def get_hour_dow(year_min: int, year_max: int) -> pd.DataFrame:
    """Heatmap hour × dow (0=Вс..6=Сб).

    Возвращает DataFrame[hour, dow, n].
    """
    sql = text(
        """
        SELECT
            EXTRACT(hour FROM datetime)::int AS hour,
            EXTRACT(dow FROM datetime)::int AS dow,
            COUNT(*)::int AS n
        FROM accidents
        WHERE is_in_region AND datetime IS NOT NULL
          AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        GROUP BY hour, dow
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_monthly_by_year(year_min: int, year_max: int) -> pd.DataFrame:
    """Помесячная динамика по годам.

    Возвращает DataFrame[year, month, n] (year × 12 строк).
    """
    sql = text(
        """
        SELECT
            EXTRACT(year FROM datetime)::int AS year,
            EXTRACT(month FROM datetime)::int AS month,
            COUNT(*)::int AS n
        FROM accidents
        WHERE is_in_region AND datetime IS NOT NULL
          AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        GROUP BY year, month
        ORDER BY year, month
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_severity_by_field(field: str, year_min: int, year_max: int) -> pd.DataFrame:
    """Кросс-таб severity × произвольный категориальный field.

    field — одно из: light_type, traffic_area_state, mt_rate.
    Защита от SQL-инъекции — белый список.
    """
    allowed = {"light_type", "traffic_area_state", "mt_rate"}
    if field not in allowed:
        raise ValueError(f"Unsupported field: {field}")
    sql = text(
        f"""
        SELECT {field} AS category, severity, COUNT(*)::int AS n
        FROM accidents
        WHERE is_in_region AND severity IS NOT NULL AND {field} IS NOT NULL
          AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        GROUP BY category, severity
        ORDER BY 3 DESC
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_severity_by_clouds(year_min: int, year_max: int) -> pd.DataFrame:
    """Severity × clouds (первое значение JSONB-массива)."""
    sql = text(
        """
        SELECT
            COALESCE(clouds->>0, 'не указано') AS category,
            severity,
            COUNT(*)::int AS n
        FROM accidents
        WHERE is_in_region AND severity IS NOT NULL
          AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        GROUP BY category, severity
        ORDER BY 3 DESC
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_severity_by_defects(year_min: int, year_max: int) -> pd.DataFrame:
    """Severity × has_defect (наличие записей в jsonb-массиве)."""
    sql = text(
        """
        SELECT
            CASE
                WHEN defects IS NULL OR jsonb_array_length(defects) = 0
                    THEN 'без дефектов'
                ELSE 'есть дефекты'
            END AS category,
            severity,
            COUNT(*)::int AS n
        FROM accidents
        WHERE is_in_region AND severity IS NOT NULL
          AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        GROUP BY category, severity
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_rhd_lhd_severity(year_min: int, year_max: int) -> pd.DataFrame:
    """Severity по RHD/LHD (только classified high/medium confidence).

    Ключевой аналитический результат: гипотеза МВД 4.9× опровергнута, реальное соотношение 0.65×.
    """
    sql = text(
        """
        SELECT
            CASE WHEN v.is_right_hand_drive THEN 'RHD' ELSE 'LHD' END AS steering,
            a.severity,
            COUNT(*)::int AS n
        FROM vehicles v
        JOIN accidents a ON a.id = v.accident_id
        WHERE v.is_right_hand_drive IS NOT NULL
          AND v.steering_confidence IN ('high', 'medium')
          AND a.severity IS NOT NULL
          AND a.is_in_region
          AND EXTRACT(year FROM a.datetime) BETWEEN :y0 AND :y1
        GROUP BY steering, a.severity
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_np(year_min: int, year_max: int, limit: int = 30) -> pd.DataFrame:
    """Топ-N населённых пунктов по количеству ДТП."""
    sql = text(
        """
        SELECT np, COUNT(*)::int AS n,
               SUM(CASE WHEN severity='dead' THEN 1 ELSE 0 END)::int AS n_dead
        FROM accidents
        WHERE is_in_region AND np IS NOT NULL AND np <> ''
          AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        GROUP BY np
        ORDER BY n DESC
        LIMIT :limit
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max, "limit": limit})


@st.cache_data(ttl=3600, show_spinner=False)
def get_city_vs_highway(year_min: int, year_max: int) -> pd.DataFrame:
    """City vs highway: severity-распределение."""
    sql = text(
        """
        SELECT
            CASE
                WHEN roads IS NOT NULL AND TRIM(roads) <> '' THEN 'highway'
                ELSE 'city'
            END AS area,
            severity,
            COUNT(*)::int AS n
        FROM accidents
        WHERE is_in_region AND severity IS NOT NULL
          AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        GROUP BY area, severity
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_dangerous_roads(year_min: int, year_max: int, min_n: int = 30) -> pd.DataFrame:
    """Топ-10 опасных дорог по pct_dead, при n_points >= min_n."""
    sql = text(
        """
        SELECT
            roads,
            COUNT(*)::int AS n,
            SUM(CASE WHEN severity='dead' THEN 1 ELSE 0 END)::int AS n_dead,
            SUM(CASE WHEN severity='dead' THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS pct_dead
        FROM accidents
        WHERE is_in_region AND roads IS NOT NULL AND TRIM(roads) <> ''
          AND severity IS NOT NULL
          AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        GROUP BY roads
        HAVING COUNT(*) >= :min_n
        ORDER BY pct_dead DESC
        LIMIT 10
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max, "min_n": min_n})


@st.cache_data(ttl=3600, show_spinner=False)
def get_np_em_heatmap(year_min: int, year_max: int) -> pd.DataFrame:
    """Heatmap топ-7 НП × топ-7 em_type."""
    sql = text(
        """
        WITH top_np AS (
            SELECT np FROM accidents
            WHERE is_in_region AND np IS NOT NULL AND np <> ''
              AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
            GROUP BY np ORDER BY COUNT(*) DESC LIMIT 7
        ), top_em AS (
            SELECT em_type FROM accidents
            WHERE is_in_region AND em_type IS NOT NULL
              AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
            GROUP BY em_type ORDER BY COUNT(*) DESC LIMIT 7
        )
        SELECT a.np, a.em_type, COUNT(*)::int AS n
        FROM accidents a
        WHERE a.np IN (SELECT np FROM top_np)
          AND a.em_type IN (SELECT em_type FROM top_em)
          AND a.is_in_region
          AND EXTRACT(year FROM a.datetime) BETWEEN :y0 AND :y1
        GROUP BY a.np, a.em_type
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_drunk_unbelted_severity(year_min: int, year_max: int) -> pd.DataFrame:
    """Severity по условиям: пьяные / непристёгнутые / возраст водителя.

    Возвращает long-format: severity, has_drunk, has_unbelted, avg_driver_age, n.

    avg_driver_age — ТОЛЬКО среди ``part_type = 'Водитель'`` с
    age_from_telegram IS NOT NULL. Если в ДТП несколько водителей,
    берём среднее (обычно 1 водитель = его возраст). Пешеходы и
    пассажиры из avg_age исключены — иначе сигнал «молодые
    водители опаснее» некорректен (Дн 16 fix).

    has_drunk — есть ли среди ВСЕХ участников хотя бы один с
    промилле >= 0.16. has_unbelted — есть ли непристёгнутый.
    """
    sql = text(
        """
        WITH part_summary AS (
            SELECT
                v.accident_id,
                COUNT(*) FILTER (WHERE p.med_result_permille >= 0.16) AS n_drunk,
                COUNT(*) FILTER (WHERE p.safety_belt = 'Нет') AS n_unbelted,
                AVG(p.age_from_telegram) FILTER (
                    WHERE p.part_type = 'Водитель' AND p.age_from_telegram IS NOT NULL
                ) AS avg_driver_age
            FROM vehicles v
            JOIN participants p ON p.vehicle_id = v.id
            GROUP BY v.accident_id
        )
        SELECT
            a.severity,
            COALESCE(ps.n_drunk, 0) > 0 AS has_drunk,
            COALESCE(ps.n_unbelted, 0) > 0 AS has_unbelted,
            ps.avg_driver_age,
            COUNT(*)::int AS n
        FROM accidents a
        LEFT JOIN part_summary ps ON ps.accident_id = a.id
        WHERE a.is_in_region AND a.severity IS NOT NULL
          AND EXTRACT(year FROM a.datetime) BETWEEN :y0 AND :y1
        GROUP BY a.severity, has_drunk, has_unbelted, ps.avg_driver_age
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_holidays_severity(year_min: int, year_max: int) -> pd.DataFrame:
    """Праздники vs будни — severity-распределение.

    Праздники определяются упрощённо: 1, 7, 8 янв, 23 фев, 8 мар, 1, 9 мая,
    12 июн, 4 ноя. Этого достаточно для UI.
    """
    sql = text(
        """
        WITH classified AS (
            SELECT
                CASE
                    WHEN (EXTRACT(month FROM datetime) = 1 AND EXTRACT(day FROM datetime) IN (1, 2, 3, 4, 5, 6, 7, 8))
                      OR (EXTRACT(month FROM datetime) = 2 AND EXTRACT(day FROM datetime) = 23)
                      OR (EXTRACT(month FROM datetime) = 3 AND EXTRACT(day FROM datetime) = 8)
                      OR (EXTRACT(month FROM datetime) = 5 AND EXTRACT(day FROM datetime) IN (1, 9))
                      OR (EXTRACT(month FROM datetime) = 6 AND EXTRACT(day FROM datetime) = 12)
                      OR (EXTRACT(month FROM datetime) = 11 AND EXTRACT(day FROM datetime) = 4)
                    THEN 'праздник'
                    ELSE 'будни'
                END AS category,
                severity
            FROM accidents
            WHERE is_in_region AND datetime IS NOT NULL AND severity IS NOT NULL
              AND EXTRACT(year FROM datetime) BETWEEN :y0 AND :y1
        )
        SELECT category, severity, COUNT(*)::int AS n
        FROM classified
        GROUP BY category, severity
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"y0": year_min, "y1": year_max})


@st.cache_data(ttl=3600, show_spinner=False)
def get_em_type_year_severity() -> pd.DataFrame:
    """Кросс-таб ``em_type × year × severity`` для всех ДТП в регионе.

    Фильтр ``is_in_region = TRUE`` — исключаем 1 393 точки вне сухопутного
    полигона Приморья (Null Island fallback и точки в воде, дефект #7).

    Returns
    -------
    DataFrame с колонками: em_type, year, severity, n. Около 800 строк.
    """
    sql = text(
        """
        SELECT
            em_type,
            EXTRACT(year FROM datetime)::int AS year,
            severity,
            COUNT(*) AS n
        FROM accidents
        WHERE em_type IS NOT NULL
          AND severity IS NOT NULL
          AND is_in_region = TRUE
        GROUP BY em_type, year, severity
        ORDER BY year, em_type, severity
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn)
    df["year"] = df["year"].astype(int)
    df["n"] = df["n"].astype(int)
    return df
