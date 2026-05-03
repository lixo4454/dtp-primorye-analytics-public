"""
DBSCAN clustering of accident hotspots in Primorye Krai.

Что делает: модуль, реализующий поиск географических очагов аварийности
через DBSCAN на координатах ДТП. Для максимальной точности использует
кастомную азимутальную равнопромежуточную проекцию (AEQD), центрированную
в географическом центре Приморья (134.5° E, 45.0° N) — это даёт точность
расстояний ~0.05% по всему региону против ~0.4% у UTM zone 52N на
восточной кромке (Тернейский район, Дальнегорск).

Зачем нужно: производит структурированный список «кластеров аварийности»
(центроид, число ДТП, %смертельных, доминирующий тип, топ улиц), который
далее используется для (1) Folium-карты топ-N очагов, (2) сопоставления
с известными опасными местами Владивостока — Золотой мост, объездная
Артёма, Гоголя, Снеговая, (3) формирования признаков для CatBoost-модели
предсказания тяжести ДТП.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from sqlalchemy import text
from sqlalchemy.orm import Session

# =====================================================================
# Custom AEQD projection — Primorye-centered for maximum accuracy
# =====================================================================
# Azimuthal Equidistant с центром в географическом центре Приморья
# (lat_0=45.0, lon_0=134.5). Расстояния от центра до любой точки региона
# (~500 км максимум) сохраняются с ошибкой <0.05%. Это лучше, чем
# UTM zone 52N (центральный меридиан 129° E, ошибка до 0.4% на
# восточной кромке Приморья) и UTM zone 53N (135° E, лучше для центра,
# но искажает Владивосток).
PRIMORYE_AEQD_PROJ_STRING = (
    "+proj=aeqd +lat_0=45.0 +lon_0=134.5 " "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

# Прямая и обратная проекции — кэшируем на уровне модуля
_FWD = Transformer.from_crs("EPSG:4326", PRIMORYE_AEQD_PROJ_STRING, always_xy=True)
_INV = Transformer.from_crs(PRIMORYE_AEQD_PROJ_STRING, "EPSG:4326", always_xy=True)


# =====================================================================
# Data classes
# =====================================================================


@dataclass(frozen=True)
class Cluster:
    """Один кластер ДТП — географический очаг аварийности."""

    cluster_id: int
    n_points: int

    # Центроид и геометрия в WGS84 — для карты и отчётов
    centroid_lon: float
    centroid_lat: float
    bbox_lon_min: float
    bbox_lon_max: float
    bbox_lat_min: float
    bbox_lat_max: float

    # Дистанции в метрах (вычислены в AEQD)
    radius_meters: float  # max расстояние от центра до точки кластера
    median_distance_meters: float  # медиана — оценка «плотности»

    # Severity-аналитика
    severity_distribution: dict[
        str, int
    ]  # {'light': N, 'severe': M, 'dead': K, 'severe_multiple': J}
    pct_dead: float  # % ДТП с погибшими — главная метрика опасности
    pct_severe_or_dead: float  # % ДТП тяжёлых (dead + severe + severe_multiple)

    # Тематика и адресная привязка (топ-5 — для лучшей интерпретации)
    top_em_types: list[tuple[str, int]]  # топ-5 типов ДТП
    top_np: list[tuple[str, int]]  # топ-5 населённых пунктов
    top_streets: list[tuple[str, int]]  # топ-5 улиц (np+street как ключ)

    # IDs для drill-down
    accident_ids: list[int]


@dataclass(frozen=True)
class HotspotsResult:
    """Полный результат прогона DBSCAN."""

    clusters: list[Cluster]  # отсортированы по n_points DESC
    noise_count: int  # точки с label=-1 (шум, не вошли в кластеры)
    clustered_count: int  # точки в кластерах (всего по всем кластерам)
    total_points: int  # всего входных точек

    # Параметры запуска — для трассировки и sensitivity-анализа
    eps_meters: float
    min_samples: int
    region_name: str
    projection: str = field(default=PRIMORYE_AEQD_PROJ_STRING)

    elapsed_seconds: float = 0.0


# =====================================================================
# Main function
# =====================================================================


def find_hotspots(
    session: Session,
    eps_meters: float = 300.0,
    min_samples: int = 15,
    region_name: str = "primorye_krai_land",
    only_in_region: bool = True,
) -> HotspotsResult:
    """Найти кластеры ДТП через DBSCAN в кастомной AEQD-проекции.

    Args:
        session: SQLAlchemy Session.
        eps_meters: радиус DBSCAN в метрах (300м = ~отрезок улицы).
        min_samples: минимум точек для образования кластера.
        region_name: код региона в region_boundaries (для фильтра is_in_region).
        only_in_region: если True (по умолчанию), кластеризуем только точки
            с is_in_region IS TRUE (отбрасывая дефекты #6 и #7).

    Returns:
        HotspotsResult с отсортированным списком кластеров (по убыванию размера).
    """
    t_start = time.time()
    logger.info(
        "find_hotspots: eps={}m, min_samples={}, region={!r}, only_in_region={}",
        eps_meters,
        min_samples,
        region_name,
        only_in_region,
    )

    # 1) Выгрузка точек из БД (WGS84)
    where = "is_in_region IS TRUE" if only_in_region else "point IS NOT NULL"
    rows = session.execute(
        text(
            f"""
            SELECT
                id,
                severity,
                em_type,
                np,
                street,
                ST_X(point) AS lon,
                ST_Y(point) AS lat
            FROM accidents
            WHERE {where}
            ORDER BY id
            """
        )
    ).all()
    if not rows:
        logger.warning("No accidents matched the filter — empty result")
        return HotspotsResult(
            clusters=[],
            noise_count=0,
            clustered_count=0,
            total_points=0,
            eps_meters=eps_meters,
            min_samples=min_samples,
            region_name=region_name,
            elapsed_seconds=time.time() - t_start,
        )

    df = pd.DataFrame([dict(r._mapping) for r in rows])
    logger.info("Loaded {} points from DB", len(df))

    # 2) Проекция в кастомную AEQD (метры от центра Приморья)
    xs, ys = _FWD.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"] = xs
    df["y"] = ys

    # 3) DBSCAN
    t_dbscan = time.time()
    db = DBSCAN(
        eps=eps_meters,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit(df[["x", "y"]].to_numpy())
    df["cluster_id"] = db.labels_
    logger.info(
        "DBSCAN: {:.2f}s, unique labels: {} (incl. noise=-1)",
        time.time() - t_dbscan,
        df["cluster_id"].nunique(),
    )

    noise_count = int((df["cluster_id"] == -1).sum())
    clustered_df = df[df["cluster_id"] >= 0].copy()
    logger.info(
        "Clustered: {} points in {} clusters | Noise: {} ({:.1%})",
        len(clustered_df),
        clustered_df["cluster_id"].nunique() if len(clustered_df) else 0,
        noise_count,
        noise_count / len(df) if len(df) else 0,
    )

    # 4) Агрегация по кластерам
    clusters: list[Cluster] = []
    for cid, group in clustered_df.groupby("cluster_id"):
        clusters.append(_build_cluster(int(cid), group))

    # Сортируем по убыванию размера — топ-N будут в начале
    clusters.sort(key=lambda c: c.n_points, reverse=True)

    elapsed = time.time() - t_start
    logger.info(
        "find_hotspots done in {:.2f}s — {} clusters, top-1 size={}",
        elapsed,
        len(clusters),
        clusters[0].n_points if clusters else 0,
    )

    return HotspotsResult(
        clusters=clusters,
        noise_count=noise_count,
        clustered_count=len(clustered_df),
        total_points=len(df),
        eps_meters=eps_meters,
        min_samples=min_samples,
        region_name=region_name,
        elapsed_seconds=elapsed,
    )


def _build_cluster(cluster_id: int, group: pd.DataFrame) -> Cluster:
    """Собирает Cluster-dataclass из группы DataFrame с одинаковым cluster_id."""
    n = len(group)
    # Центроид в AEQD-метрах
    cx = group["x"].mean()
    cy = group["y"].mean()
    # Обратная проекция в WGS84
    centroid_lon, centroid_lat = _INV.transform(cx, cy)

    # Расстояния от центроида в метрах
    dx = group["x"].to_numpy() - cx
    dy = group["y"].to_numpy() - cy
    distances = np.sqrt(dx * dx + dy * dy)
    radius_m = float(distances.max())
    median_m = float(np.median(distances))

    # Severity
    sev_counts = group["severity"].value_counts(dropna=False).to_dict()
    sev_distribution = {str(k): int(v) for k, v in sev_counts.items()}
    n_dead = sev_distribution.get("dead", 0)
    n_severe = sev_distribution.get("severe", 0)
    n_severe_mult = sev_distribution.get("severe_multiple", 0)
    pct_dead = n_dead / n if n else 0.0
    pct_severe_or_dead = (n_dead + n_severe + n_severe_mult) / n if n else 0.0

    # Топ-5 типов ДТП
    em_counter = Counter(x for x in group["em_type"].dropna().astype(str).tolist() if x)
    top_em_types = [(k, int(v)) for k, v in em_counter.most_common(5)]

    # Топ-5 населённых пунктов
    np_counter = Counter(x for x in group["np"].dropna().astype(str).tolist() if x)
    top_np = [(k, int(v)) for k, v in np_counter.most_common(5)]

    # Топ-5 улиц (с привязкой к НП — улица «Ленина» в Уссурийске и Находке разные)
    # Через явный itertuples — устойчиво к пустым DataFrame и к NaN-значениям
    street_counter: Counter[str] = Counter()
    for row in group[["np", "street"]].itertuples(index=False):
        np_val = "" if pd.isna(row.np) else str(row.np).strip()
        st_val = "" if pd.isna(row.street) else str(row.street).strip()
        if np_val and st_val:
            street_counter[f"{np_val} / {st_val}"] += 1
        elif np_val:
            street_counter[np_val] += 1
        elif st_val:
            street_counter[st_val] += 1
    top_streets = [(k, int(v)) for k, v in street_counter.most_common(5)]

    return Cluster(
        cluster_id=cluster_id,
        n_points=n,
        centroid_lon=float(centroid_lon),
        centroid_lat=float(centroid_lat),
        bbox_lon_min=float(group["lon"].min()),
        bbox_lon_max=float(group["lon"].max()),
        bbox_lat_min=float(group["lat"].min()),
        bbox_lat_max=float(group["lat"].max()),
        radius_meters=radius_m,
        median_distance_meters=median_m,
        severity_distribution=sev_distribution,
        pct_dead=float(pct_dead),
        pct_severe_or_dead=float(pct_severe_or_dead),
        top_em_types=top_em_types,
        top_np=top_np,
        top_streets=top_streets,
        accident_ids=[int(x) for x in group["id"].tolist()],
    )


# =====================================================================
# Self-check (модуль можно запустить напрямую для smoke-теста)
# =====================================================================


def _smoke_test() -> None:
    """Прогон с дефолтными параметрами на боевой БД — для отладки."""
    from src.database import SessionLocal

    with SessionLocal() as session:
        result = find_hotspots(session, eps_meters=300.0, min_samples=15)
    logger.info("Smoke-test result: {} clusters", len(result.clusters))
    if result.clusters:
        top = result.clusters[0]
        logger.info(
            "Top cluster: cid={}, n={}, centroid=({:.4f}, {:.4f}), pct_dead={:.2%}",
            top.cluster_id,
            top.n_points,
            top.centroid_lon,
            top.centroid_lat,
            top.pct_dead,
        )
        logger.info("Top EM types: {}", top.top_em_types)
        logger.info("Top streets: {}", top.top_streets)


if __name__ == "__main__":
    _smoke_test()
