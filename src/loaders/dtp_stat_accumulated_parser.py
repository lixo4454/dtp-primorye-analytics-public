"""
Парсер записи аккумулированного JSON dtp-stat.ru в ORM-объекты SQLAlchemy.

Превращает одну запись формата
{
    'EM_NUMBER': ...,
    'DATE_TIME': ...,
    'EM_TYPE': ...,
    'VEHICLES': [...],
    'PERSONS': [...] or None,
    ...
}
в Accident со связанными Vehicle, Participant, AccidentPedestrian.
"""

from datetime import datetime

from geoalchemy2.shape import from_shape
from loguru import logger
from shapely.geometry import Point

from src.database.models import (
    Accident,
    AccidentPedestrian,
    Participant,
    Vehicle,
)


def parse_datetime(value) -> datetime | None:
    """Преобразует ISO-строку '2024-01-01T05:25:00' в datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        logger.warning(f"Не удалось распарсить дату: {value!r}")
        return None


def parse_point(rec: dict):
    """
    Извлекает координаты из записи.

    В источнике координаты лежат либо в LAT/LNG, либо в POINT.coordinates.
    Возвращает PostGIS WKB или None.
    """
    lat = rec.get("LAT")
    lng = rec.get("LNG")

    # Запасной путь — POINT.coordinates
    if lat is None or lng is None:
        point_obj = rec.get("POINT") or {}
        coords = point_obj.get("coordinates")
        if coords and len(coords) == 2:
            lat, lng = coords[0], coords[1]

    if lat is None or lng is None:
        return None

    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except (ValueError, TypeError):
        return None

    # Защита от перепутанных координат и невалидных значений
    if not (-90 <= lat_f <= 90) or not (-180 <= lng_f <= 180):
        # Вероятно перепутали местами lat и lng (бывает в источнике)
        if -90 <= lng_f <= 90 and -180 <= lat_f <= 180:
            lat_f, lng_f = lng_f, lat_f
        else:
            logger.warning(f"Координаты невалидны: lat={lat}, lng={lng}")
            return None

    return from_shape(Point(lng_f, lat_f), srid=4326)


def compute_severity(lost: int, suffer: int) -> str:
    """Вычисляет тяжесть ДТП по официальной градации МВД РФ.

    Returns:
        - 'dead'             — есть погибшие
        - 'severe_multiple'  — массовые пострадавшие (4+)
        - 'severe'           — несколько пострадавших (2-3)
        - 'light'            — один пострадавший
        - 'no_injuries'      — без пострадавших
    """
    if lost > 0:
        return "dead"
    if suffer >= 4:
        return "severe_multiple"
    if suffer >= 2:
        return "severe"
    if suffer == 1:
        return "light"
    return "no_injuries"


def parse_participant_fields(p_data: dict) -> dict:
    """
    Извлекает поля участника. Возвращает dict.
    Используется и для Participant (внутри ТС), и для AccidentPedestrian.
    """
    permille = p_data.get("MED_RESULT_PERMILLE")
    # Numeric может быть строкой или числом
    if permille is not None:
        try:
            permille = float(permille)
        except (ValueError, TypeError):
            permille = None

    return {
        "n": p_data.get("N"),
        "part_type": p_data.get("PART_TYPE"),
        "sex": p_data.get("SEX"),
        "driver_service_length": p_data.get("DRIVER_SERVICE_LENGTH"),
        "hv_type": p_data.get("HV_TYPE"),
        "injured_card": p_data.get("INJURED_CARD"),
        "safety_belt": p_data.get("SAFETY_BELT"),
        "child_safety_type": p_data.get("CHILD_SAFETY_TYPE"),
        "med_result_permille": permille,
        "escape": p_data.get("ESCAPE"),
        "main_pdd_violation": p_data.get("MAIN_PDD_DERANGEMENT"),
        "attendant_pdd_violation": p_data.get("ATTENDANT_PDD_DERANGEMENT"),
    }


def parse_record(rec: dict) -> Accident | None:
    """
    Превращает одну запись JSON в объект Accident со связанными Vehicle/Participant/Pedestrian.

    Возвращает None, если запись битая (нет обязательных полей).
    """
    # Минимальная валидация
    external_id = rec.get("EM_NUMBER")
    if external_id is None:
        logger.warning("Запись без EM_NUMBER, пропускаю")
        return None

    dt = parse_datetime(rec.get("DATE_TIME"))
    if dt is None:
        logger.warning(f"ДТП {external_id}: не удалось распарсить DATE_TIME, пропускаю")
        return None

    # parent_region из массива REGIONS
    regions = rec.get("REGIONS") or []
    parent_region = None
    for r in regions:
        if r and "Российская Федерация" not in r:
            parent_region = r
            break

    # Числовые показатели с защитой от None
    lost = rec.get("LOST_AMOUNT") or 0
    suffer = rec.get("SUFFER_AMOUNT") or 0
    pers = rec.get("PERS_AMOUNT") or 0
    veh = rec.get("VEH_AMOUNT") or 0

    # Создаём Accident
    accident = Accident(
        external_id=external_id,
        gibdd_number=str(rec.get("EMTP_NUMBER")) if rec.get("EMTP_NUMBER") else None,
        datetime=dt,
        parent_region=parent_region,
        place=rec.get("PLACE"),
        np=rec.get("NP"),
        street=rec.get("STREET"),
        house=rec.get("HOUSE"),
        roads=rec.get("ROADS"),
        road_km=rec.get("ROAD_KM_M"),
        street_sign=rec.get("STREET_SIGN"),
        road_sign=rec.get("ROAD_SIGN"),
        road_type=rec.get("ROAD_TYPE"),
        point=parse_point(rec),
        em_type=rec.get("EM_TYPE"),
        schema_id=rec.get("SCHEMA_ID"),
        light_type=rec.get("LIGHT_TYPE"),
        clouds=rec.get("CLOUDS"),
        traffic_area_state=rec.get("TRAFFIC_AREA_STATE"),
        defects=rec.get("DEFECTS"),
        motion_influences=rec.get("MOTION_INFLUENCES"),
        mt_rate=rec.get("MT_RATE"),
        rd_constr_heres=rec.get("RD_CONSTR_HERES"),
        rd_constr_theres=rec.get("RD_CONSTR_THERES"),
        veh_amount=veh,
        pers_amount=pers,
        lost_amount=lost,
        suffer_amount=suffer,
        children_attr=rec.get("CHILDREN_ATTR"),
        additional_attr=rec.get("ADDITIONAL_ATTR"),
        severity=compute_severity(lost, suffer),
        raw_data=rec,
    )

    # Машины
    for veh_data in rec.get("VEHICLES") or []:
        vehicle = Vehicle(
            n=veh_data.get("N"),
            mark=veh_data.get("MARK"),
            model=veh_data.get("MODEL"),
            vehicle_year=veh_data.get("VEHICLE_YEAR"),
            color=veh_data.get("COLOR"),
            prod_type=veh_data.get("PROD_TYPE"),
            okfs=veh_data.get("OKFS"),
            okopf=veh_data.get("OKOPF"),
            rudder_type=veh_data.get("RUDDER_TYPE"),
            tech_failure_type=veh_data.get("TECH_FAILURE_TYPE"),
            damage_disposition=veh_data.get("DAMAGE_DISPOSITION"),
            escape=veh_data.get("ESCAPE"),
        )

        # Участники внутри ТС (водители/пассажиры)
        for p_data in veh_data.get("PERSONS") or []:
            participant = Participant(**parse_participant_fields(p_data))
            vehicle.participants.append(participant)

        accident.vehicles.append(vehicle)

    # Пешеходы (PERSONS на верхнем уровне ДТП)
    for p_data in rec.get("PERSONS") or []:
        pedestrian = AccidentPedestrian(**parse_participant_fields(p_data))
        accident.pedestrians.append(pedestrian)

    return accident
