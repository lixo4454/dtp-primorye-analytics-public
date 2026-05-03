"""Router /recommendations — СППР для ГИБДД.

Эндпоинты:

* ``GET /recommendations/hotspot/{cluster_id}`` — топ-K правил для
  DBSCAN-очага (артефакт, ``hotspots_summary.json``).
* ``GET /recommendations/point?lat=&lon=&radius=&top_k=`` — топ-K
  для произвольной точки + радиус. PostGIS ``ST_DWithin`` через
  GIST-индекс ``idx_accidents_point``. Sample size ≥ 10 — иначе
  возвращается пустой ``items`` + ``note``.

Каждая рекомендация включает rule_id, priority, title, trigger_human,
expected_effect (CMF + 95 % CI), evidence_basis (≥1 источник —
Cochrane / FHWA / CMF Clearinghouse / iRAP / TRB), confidence, score.
Score-сортировка: priority_weight × confidence × |effect|.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.analysis.recommendations import (
    Recommendation,
    RuleEngine,
    score_recommendation,
)
from src.analysis.spot_profile import HotspotProfile
from src.api.dependencies import ModelRegistry, get_db, get_models
from src.api.schemas import (
    CitationOut,
    EffectEstimateOut,
    HotspotProfileOut,
    RecommendationOut,
    RecommendationsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Минимальное количество ДТП для запуска rule engine на динамической области.
# При меньшем sample size биномиальный CI для pct_dead очень широкий
# (±30 %), а top_em_type определён шумом. См. methodology раздел 5.3.
MIN_DYNAMIC_SAMPLE = 10
RULE_ENGINE = RuleEngine()


def _profile_to_out(p: HotspotProfile) -> HotspotProfileOut:
    return HotspotProfileOut(
        n_points=p.n_points,
        radius_meters=p.radius_meters,
        pct_dead=p.pct_dead,
        pct_severe_or_dead=p.pct_severe_or_dead,
        top_em_type=p.top_em_type,
        top_em_types=p.top_em_types,
        top_np=p.top_np,
        dominant_light_type=p.dominant_light_type,
        dominant_state=p.dominant_state,
        is_highway=p.is_highway,
        is_in_city=p.is_in_city,
        has_night_dominant=p.has_night_dominant,
        has_winter_spike=p.has_winter_spike,
        centroid_lat=p.centroid_lat,
        centroid_lon=p.centroid_lon,
        source=p.source,
        cluster_id=p.cluster_id,
        radius_query_m=p.radius_query_m,
    )


def _rec_to_out(rec: Recommendation) -> RecommendationOut:
    return RecommendationOut(
        rule_id=rec.rule_id,
        priority=rec.priority,
        title=rec.title,
        icon=rec.icon,
        trigger_human=rec.trigger_human,
        expected_effect=EffectEstimateOut(
            metric=rec.expected_effect.metric,
            point_estimate=rec.expected_effect.point_estimate,
            ci_low=rec.expected_effect.ci_low,
            ci_high=rec.expected_effect.ci_high,
            note=rec.expected_effect.note,
        ),
        evidence_basis=[
            CitationOut(label=c.label, url=c.url, quoted_effect=c.quoted_effect)
            for c in rec.evidence_basis
        ],
        confidence=rec.confidence,
        implementation_cost=rec.implementation_cost,  # type: ignore[arg-type]
        target_severity=list(rec.target_severity),
        target_em_types=list(rec.target_em_types),
        score=score_recommendation(rec),
    )


# =====================================================================
# /recommendations/hotspot/{cluster_id} — для DBSCAN-очага
# =====================================================================


@router.get(
    "/hotspot/{cluster_id}",
    response_model=RecommendationsResponse,
    summary="Топ-K рекомендаций для DBSCAN-очага",
    responses={
        404: {"description": "Cluster not found in hotspots_summary.json"},
        503: {"description": "hotspots_summary.json не загружен"},
    },
)
async def recommendations_for_hotspot(
    cluster_id: int,
    models: Annotated[ModelRegistry, Depends(get_models)],
    top_k: int = Query(5, ge=1, le=18, description="Сколько рекомендаций вернуть"),
) -> RecommendationsResponse:
    """Загружает очаг из артефакта, строит профиль, возвращает топ-K."""
    summary = models.hotspots_summary
    if not summary:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "hotspots_summary.json не загружен — запусти scripts/run_dbscan_hotspots.py",
        )

    raw_clusters = summary.get("top_clusters", [])
    cluster = next((c for c in raw_clusters if c.get("cluster_id") == cluster_id), None)
    if cluster is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Cluster {cluster_id} not found in hotspots_summary",
        )

    profile = HotspotProfile.from_dbscan(cluster)
    recs = RULE_ENGINE.recommend(profile, top_k=top_k)

    note: str | None = None
    if not recs:
        note = (
            "Ни одно правило не сработало для этого очага. "
            "Проверь pct_dead/severe и dominant_light/state — возможно, "
            "профиль на грани триггеров."
        )

    return RecommendationsResponse(
        profile=_profile_to_out(profile),
        items=[_rec_to_out(r) for r in recs],
        note=note,
    )


# =====================================================================
# /recommendations/point — динамический очаг (lat, lon, radius)
# =====================================================================


@router.get(
    "/point",
    response_model=RecommendationsResponse,
    summary="Топ-K рекомендаций для произвольной точки + радиуса",
    responses={
        503: {"description": "БД недоступна"},
    },
)
async def recommendations_for_point(
    db: Annotated[AsyncSession, Depends(get_db)],
    lat: float = Query(..., ge=-90, le=90, description="Широта (WGS84)"),
    lon: float = Query(..., ge=-180, le=180, description="Долгота (WGS84)"),
    radius: int = Query(
        100,
        ge=10,
        le=5000,
        description="Радиус в метрах (можно любое целое 10..5000)",
    ),
    top_k: int = Query(5, ge=1, le=18),
    year_min: int | None = Query(
        None,
        ge=2000,
        le=2100,
        description="Нижняя граница периода (год datetime'а ДТП, включительно)",
    ),
    year_max: int | None = Query(
        None,
        ge=2000,
        le=2100,
        description="Верхняя граница периода (включительно)",
    ),
) -> RecommendationsResponse:
    """Динамический очаг: PostGIS ``ST_DWithin`` → профиль → rule engine.

    ``year_min/year_max`` фильтруют ДТП по году datetime'а — рекомендации
    учитывают только происшествия в выбранном периоде.

    Если в радиусе **<10 ДТП**, ``items`` пустой, ``note`` указывает
    на необходимость увеличить радиус или расширить период.
    """
    try:
        profile = await HotspotProfile.from_dynamic_radius(
            lat=lat,
            lon=lon,
            radius_m=radius,
            session=db,
            year_min=year_min,
            year_max=year_max,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("from_dynamic_radius failed")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"PostGIS query failed: {e}",
        ) from e

    if profile.n_points < MIN_DYNAMIC_SAMPLE:
        period_str = ""
        if year_min is not None or year_max is not None:
            period_str = f" за период {year_min or '…'}–{year_max or '…'}"
        return RecommendationsResponse(
            profile=_profile_to_out(profile),
            items=[],
            note=(
                f"Найдено {profile.n_points} ДТП в радиусе {radius} м{period_str}. "
                f"Для надёжной статистики нужно ≥ {MIN_DYNAMIC_SAMPLE} ДТП — "
                "увеличь радиус или расширь диапазон годов."
            ),
        )

    recs = RULE_ENGINE.recommend(profile, top_k=top_k)

    note: str | None = None
    if not recs:
        note = (
            "Ни одно правило не сработало. Профиль не попадает под "
            "триггеры 18 правил из docs/recommendations_methodology.md."
        )

    return RecommendationsResponse(
        profile=_profile_to_out(profile),
        items=[_rec_to_out(r) for r in recs],
        note=note,
    )
