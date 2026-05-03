"""
Router /accidents — список ДТП с фильтрами, детальная карточка, агрегаты.

Эндпоинты:
- GET /accidents — пагинированный список с фильтрами year/severity/em_type/in_region
- GET /accidents/{id} — детали ДТП с участниками + ТС + пешеходами (selectinload, без N+1)
- GET /accidents/stats — агрегаты для дашборда (by severity / em_type / year)

Особенности:
- AsyncSession + selectinload для loading vehicles → participants и pedestrians
  одним JOIN-batch (избежание классической N+1 в FastAPI/ORM)
- PostGIS Point → GeoPoint{lat, lon} через ST_Y/ST_X в SQL (быстрее чем shapely)
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import get_db
from src.api.schemas import (
    AccidentDetail,
    AccidentsListResponse,
    AccidentsStatsResponse,
    AccidentSummary,
    GeoPoint,
    PedestrianOut,
    SeverityClass,
    VehicleOut,
)
from src.database.models import Accident, Vehicle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/accidents", tags=["accidents"])


# =====================================================================
# Helpers
# =====================================================================


async def _fetch_point(session: AsyncSession, accident_id: int) -> GeoPoint | None:
    """Достаёт координаты через ST_Y/ST_X. Дешевле чем грузить geometry в SQLAlchemy."""
    from sqlalchemy import text as _text

    r = await session.execute(
        _text("SELECT ST_Y(point) AS lat, ST_X(point) AS lon FROM accidents WHERE id = :id"),
        {"id": accident_id},
    )
    row = r.first()
    if row is None or row.lat is None or row.lon is None:
        return None
    return GeoPoint(lat=float(row.lat), lon=float(row.lon))


def _accident_to_summary(a: Accident, point: GeoPoint | None) -> AccidentSummary:
    """ORM Accident → AccidentSummary (point подменяем явно)."""
    return AccidentSummary(
        id=a.id,
        external_id=a.external_id,
        datetime=a.datetime,
        place=a.place,
        np=a.np,
        street=a.street,
        em_type=a.em_type,
        severity=a.severity,
        veh_amount=a.veh_amount,
        pers_amount=a.pers_amount,
        lost_amount=a.lost_amount,
        suffer_amount=a.suffer_amount,
        point=point,
        is_in_region=a.is_in_region,
    )


def _normalize_jsonb_array(value) -> list[str] | None:
    """JSONB-массив (clouds/defects/...) -> list[str] | None.

    Источник может содержать None / dict / list. Нормализуем в list[str].
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


# =====================================================================
# Эндпоинты
# =====================================================================


@router.get(
    "",
    response_model=AccidentsListResponse,
    summary="Пагинированный список ДТП с фильтрами",
)
async def list_accidents(
    session: Annotated[AsyncSession, Depends(get_db)],
    year: int | None = Query(None, ge=2010, le=2035, description="Год ДТП"),
    severity: SeverityClass | None = Query(None, description="Тяжесть"),
    em_type: str | None = Query(None, description="Тип ДТП (Столкновение, Наезд на пешехода, ...)"),
    place: str | None = Query(None, description="Городской округ / район (LIKE)"),
    in_region_only: bool = Query(False, description="Только ДТП в сухопутной границе Приморья"),
    limit: int = Query(50, ge=1, le=500, description="Максимум записей на странице"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
) -> AccidentsListResponse:
    """Возвращает список ДТП с фильтрами. Сортировка: по datetime DESC."""
    from sqlalchemy import text as _text

    # Базовый WHERE
    conditions = []
    params: dict = {}
    if year is not None:
        conditions.append("EXTRACT(YEAR FROM a.datetime)::int = :year")
        params["year"] = year
    if severity is not None:
        conditions.append("a.severity = :severity")
        params["severity"] = severity
    if em_type is not None:
        conditions.append("a.em_type = :em_type")
        params["em_type"] = em_type
    if place is not None:
        conditions.append("a.place ILIKE :place")
        params["place"] = f"%{place}%"
    if in_region_only:
        conditions.append("a.is_in_region IS TRUE")

    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # Total
    total_sql = _text(f"SELECT COUNT(*) FROM accidents a {where_sql}")
    total = int((await session.execute(total_sql, params)).scalar() or 0)

    # Page
    page_sql = _text(
        f"""
        SELECT
            a.id, a.external_id, a.datetime, a.place, a.np, a.street, a.em_type,
            a.severity, a.veh_amount, a.pers_amount, a.lost_amount, a.suffer_amount,
            a.is_in_region,
            ST_Y(a.point) AS lat, ST_X(a.point) AS lon
        FROM accidents a
        {where_sql}
        ORDER BY a.datetime DESC, a.id DESC
        LIMIT :limit OFFSET :offset
        """
    )
    rows = (await session.execute(page_sql, {**params, "limit": limit, "offset": offset})).all()

    items = [
        AccidentSummary(
            id=row.id,
            external_id=row.external_id,
            datetime=row.datetime,
            place=row.place,
            np=row.np,
            street=row.street,
            em_type=row.em_type,
            severity=row.severity,
            veh_amount=row.veh_amount,
            pers_amount=row.pers_amount,
            lost_amount=row.lost_amount,
            suffer_amount=row.suffer_amount,
            point=GeoPoint(lat=float(row.lat), lon=float(row.lon))
            if row.lat is not None and row.lon is not None
            else None,
            is_in_region=row.is_in_region,
        )
        for row in rows
    ]
    return AccidentsListResponse(total=total, limit=limit, offset=offset, items=items)


@router.get(
    "/stats",
    response_model=AccidentsStatsResponse,
    summary="Агрегаты для дашборда (severity / em_type / по годам)",
)
async def accidents_stats(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> AccidentsStatsResponse:
    """Сводные агрегаты — для главной страницы Streamlit."""
    from sqlalchemy import text as _text

    total = int((await session.execute(_text("SELECT COUNT(*) FROM accidents"))).scalar() or 0)
    in_region = int(
        (
            await session.execute(
                _text("SELECT COUNT(*) FROM accidents WHERE is_in_region IS TRUE")
            )
        ).scalar()
        or 0
    )
    total_dead = int(
        (
            await session.execute(_text("SELECT COALESCE(SUM(lost_amount),0) FROM accidents"))
        ).scalar()
        or 0
    )
    total_suffered = int(
        (
            await session.execute(_text("SELECT COALESCE(SUM(suffer_amount),0) FROM accidents"))
        ).scalar()
        or 0
    )

    by_severity = {
        row.severity or "null": int(row.cnt)
        for row in (
            await session.execute(
                _text(
                    "SELECT severity, COUNT(*) AS cnt FROM accidents GROUP BY severity ORDER BY cnt DESC"
                )
            )
        ).all()
    }
    by_em_type = {
        row.em_type or "null": int(row.cnt)
        for row in (
            await session.execute(
                _text(
                    "SELECT em_type, COUNT(*) AS cnt FROM accidents GROUP BY em_type "
                    "ORDER BY cnt DESC LIMIT 10"
                )
            )
        ).all()
    }
    by_year = {
        int(row.y): int(row.cnt)
        for row in (
            await session.execute(
                _text(
                    "SELECT EXTRACT(YEAR FROM datetime)::int AS y, COUNT(*) AS cnt "
                    "FROM accidents GROUP BY y ORDER BY y"
                )
            )
        ).all()
    }

    return AccidentsStatsResponse(
        total=total,
        in_region=in_region,
        by_severity=by_severity,
        by_em_type=by_em_type,
        by_year=by_year,
        total_dead=total_dead,
        total_suffered=total_suffered,
    )


@router.get(
    "/{accident_id}",
    response_model=AccidentDetail,
    summary="Детали одного ДТП с участниками, ТС и пешеходами",
    responses={404: {"description": "ДТП не найдено"}},
)
async def get_accident(
    accident_id: int,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> AccidentDetail:
    """Полная карточка ДТП: vehicles → participants + pedestrians,
    одним SQL-запросом через selectinload (защита от N+1)."""
    stmt = (
        select(Accident)
        .options(
            selectinload(Accident.vehicles).selectinload(Vehicle.participants),
            selectinload(Accident.pedestrians),
        )
        .where(Accident.id == accident_id)
    )
    result = await session.execute(stmt)
    accident: Accident | None = result.scalars().first()
    if accident is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Accident id={accident_id} not found")

    # Координаты — отдельный мини-запрос (Geometry не сериализуется автоматически)
    point = await _fetch_point(session, accident_id)

    vehicles_out = [
        VehicleOut(
            id=v.id,
            n=v.n,
            mark=v.mark,
            model=v.model,
            vehicle_year=v.vehicle_year,
            color=v.color,
            prod_type=v.prod_type,
            is_right_hand_drive=v.is_right_hand_drive,
            steering_confidence=v.steering_confidence,
            participants=[
                {
                    "id": p.id,
                    "n": p.n,
                    "part_type": p.part_type,
                    "sex": p.sex,
                    "driver_service_length": p.driver_service_length,
                    "hv_type": p.hv_type,
                    "safety_belt": p.safety_belt,
                    "med_result_permille": float(p.med_result_permille)
                    if p.med_result_permille is not None
                    else None,
                    "age_from_telegram": p.age_from_telegram,
                    "age_source": p.age_source,
                }
                for p in v.participants
            ],
        )
        for v in accident.vehicles
    ]

    pedestrians_out = [
        PedestrianOut(
            id=p.id,
            n=p.n,
            part_type=p.part_type,
            sex=p.sex,
            hv_type=p.hv_type,
            age_from_telegram=p.age_from_telegram,
            age_source=p.age_source,
        )
        for p in accident.pedestrians
    ]

    return AccidentDetail(
        id=accident.id,
        external_id=accident.external_id,
        datetime=accident.datetime,
        place=accident.place,
        np=accident.np,
        street=accident.street,
        em_type=accident.em_type,
        severity=accident.severity,
        veh_amount=accident.veh_amount,
        pers_amount=accident.pers_amount,
        lost_amount=accident.lost_amount,
        suffer_amount=accident.suffer_amount,
        point=point,
        is_in_region=accident.is_in_region,
        parent_region=accident.parent_region,
        house=accident.house,
        roads=accident.roads,
        road_km=accident.road_km,
        light_type=accident.light_type,
        traffic_area_state=accident.traffic_area_state,
        mt_rate=accident.mt_rate,
        clouds=_normalize_jsonb_array(accident.clouds),
        defects=_normalize_jsonb_array(accident.defects),
        motion_influences=_normalize_jsonb_array(accident.motion_influences),
        schema_id=accident.schema_id,
        children_attr=accident.children_attr,
        vehicles=vehicles_out,
        pedestrians=pedestrians_out,
    )
