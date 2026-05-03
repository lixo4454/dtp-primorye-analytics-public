"""
Router /clusters — очаги аварийности (DBSCAN-результаты).

Эндпоинты:
- GET /clusters/hotspots — топ очагов из hotspots_summary.json

Источник данных: предсчитанный артефакт `data/processed/hotspots_summary.json`,
загруженный в ModelRegistry в lifespan. Это снимает нагрузку с БД и
PostGIS-spatial-индекса при каждом запросе (DBSCAN на 28k точках —
~600мс, недопустимо в HTTP-цикле).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import ModelRegistry, get_models
from src.api.schemas import GeoPoint, HotspotOut, HotspotsResponse

router = APIRouter(prefix="/clusters", tags=["clusters"])


@router.get(
    "/hotspots",
    response_model=HotspotsResponse,
    summary="Топ очагов аварийности (DBSCAN)",
    responses={503: {"description": "hotspots_summary не загружен (артефакт отсутствует)"}},
)
async def hotspots(
    models: Annotated[ModelRegistry, Depends(get_models)],
    min_count: int = Query(15, ge=1, description="Минимум ДТП в очаге"),
    limit: int = Query(30, ge=1, le=200, description="Максимум очагов"),
) -> HotspotsResponse:
    """Возвращает топ-N очагов из предсчитанного hotspots_summary.json."""
    summary = models.hotspots_summary
    if not summary:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "hotspots_summary.json не загружен — запусти scripts/run_dbscan_hotspots.py",
        )

    raw_clusters = summary.get("top_clusters", [])
    items: list[HotspotOut] = []
    for c in raw_clusters:
        if c.get("n_points", 0) < min_count:
            continue
        items.append(
            HotspotOut(
                rank=c["rank"],
                cluster_id=c["cluster_id"],
                n_points=c["n_points"],
                centroid=GeoPoint(lat=c["centroid_lat"], lon=c["centroid_lon"]),
                radius_meters=c["radius_meters"],
                median_distance_meters=c["median_distance_meters"],
                pct_dead=c["pct_dead"],
                pct_severe_or_dead=c["pct_severe_or_dead"],
                severity_distribution=c.get("severity_distribution", {}),
                top_em_types=[(t[0], int(t[1])) for t in c.get("top_em_types", [])[:5]],
                top_np=[(t[0], int(t[1])) for t in c.get("top_np", [])[:5]],
            )
        )
        if len(items) >= limit:
            break

    return HotspotsResponse(
        params=summary.get("params", {}),
        stats=summary.get("stats", {}),
        items=items,
    )
