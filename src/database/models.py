"""
ORM-модели SQLAlchemy для всех таблиц проекта.

Источник данных: dtp-stat.ru (аккумулированный JSON формат 2015-2026).
Извлекаем максимум полей из источника для последующего ML/NLP анализа.

Гибридный подход к строковым полям:
- TEXT — для полей с непредсказуемой длиной (адреса, описания, длинные категории)
- VARCHAR(N) — для категориальных с ограниченным набором значений
- JSONB — для массивов и структурированных данных
- raw_data JSONB — полная исходная запись на случай если что-то упустили
"""

from datetime import datetime
from typing import List, Optional

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

# =====================================================================
# ACCIDENT — главная таблица ДТП
# =====================================================================


class Accident(Base):
    """ДТП — основная сущность датасета.

    Одна запись = одно событие из аккумулированного JSON dtp-stat.ru.
    Источник: https://dtp-stat.ru/media/years/2015-2026.json.7z
    """

    __tablename__ = "accidents"

    # === Идентификаторы ===
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    external_id: Mapped[int] = mapped_column(
        BigInteger,
        unique=True,
        nullable=False,
        index=True,
        comment="EM_NUMBER из источника — уникальный ID ДТП",
    )
    gibdd_number: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="EMTP_NUMBER — номер ДТП в базе ГИБДД"
    )

    # === Время ===
    datetime: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True, comment="DATE_TIME — дата и время ДТП"
    )

    # === Место ===
    parent_region: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Регион верхнего уровня (для нашего датасета — 'Приморский край')",
    )
    place: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        index=True,
        comment="PLACE — городской округ или район (Владивостокский ГО, Уссурийский ГО, ...)",
    )
    np: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="NP — населённый пункт (г Владивосток, г Уссурийск, ...)",
    )
    street: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="STREET — улица")
    house: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="HOUSE — дом")
    roads: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="ROADS — название трассы (для трассовых ДТП)"
    )
    road_km: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="ROAD_KM_M — километр.метр трассы"
    )
    street_sign: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="STREET_SIGN — категория улицы (Магистральные/Местного значения)",
    )
    road_sign: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="ROAD_SIGN — категория дороги"
    )
    road_type: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="ROAD_TYPE — тип дороги"
    )

    # === Координаты — PostGIS Point (WGS84, EPSG:4326) ===
    point: Mapped[Optional[str]] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326),
        nullable=True,
        comment="Точка на карте",
    )

    # === Характеристики ДТП ===
    em_type: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        index=True,
        comment="EM_TYPE — тип ДТП (Столкновение, Опрокидывание, Наезд на пешехода...)",
    )
    schema_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="SCHEMA_ID — код схемы ДТП"
    )

    # === Условия ===
    light_type: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="LIGHT_TYPE — освещение/время суток"
    )
    clouds: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, comment="CLOUDS — погода (массив)"
    )
    traffic_area_state: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="TRAFFIC_AREA_STATE — состояние дорожного покрытия"
    )
    defects: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, comment="DEFECTS — дефекты дорожного покрытия (массив)"
    )
    motion_influences: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="MOTION_INFLUENCES — факторы, влияющие на движение",
    )
    mt_rate: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="MT_RATE — режим движения"
    )

    # === Объекты на месте и вблизи ===
    rd_constr_heres: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, comment="RD_CONSTR_HERES — объекты УДС на месте ДТП"
    )
    rd_constr_theres: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, comment="RD_CONSTR_THERES — объекты УДС вблизи места ДТП"
    )

    # === Числовые показатели ===
    veh_amount: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="VEH_AMOUNT — количество ТС в ДТП"
    )
    pers_amount: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="PERS_AMOUNT — количество участников",
    )
    lost_amount: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        index=True,
        comment="LOST_AMOUNT — количество погибших",
    )
    suffer_amount: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="SUFFER_AMOUNT — количество раненых"
    )
    children_attr: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="CHILDREN_ATTR — количество детей среди участников",
    )
    additional_attr: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="ADDITIONAL_ATTR — дополнительный атрибут (значение из источника)",
    )

    # === Вычисляемая тяжесть для удобства аналитики ===
    severity: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Вычисляемая тяжесть: dead/injured/minor",
    )

    # === Геопространственная валидация координат (вычисляется через ST_Within) ===
    is_in_region: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        comment="True если point внутри region_boundaries (суша Приморья); "
        "False если в воде/вне региона (дефекты #6 и #7); "
        "NULL если не вычислено или point IS NULL",
    )

    # === Snap-to-road ===
    point_snapped: Mapped[Optional[str]] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326),
        nullable=True,
        comment="Snapped-координата на ближайшее ребро OSM road graph (NULL если ещё не обработано)",
    )
    snap_distance_m: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Расстояние raw point → snapped (geography meters)",
    )
    snap_method: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        index=True,
        comment="osm_road / unchanged (>50м от дороги) / failed / NULL=не обработано",
    )
    snap_road_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
        comment="osm_roads.id выбранного сегмента (NULL если unchanged/failed)",
    )

    # === Сырые данные ===
    raw_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Полная запись из источника (для будущих расширений)",
    )

    # === Служебные временные метки ===
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # === Связи ===
    vehicles: Mapped[List["Vehicle"]] = relationship(
        back_populates="accident", cascade="all, delete-orphan"
    )
    pedestrians: Mapped[List["AccidentPedestrian"]] = relationship(
        back_populates="accident", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_accidents_datetime_severity", "datetime", "severity"),
        Index("ix_accidents_em_type_severity", "em_type", "severity"),
    )

    def __repr__(self) -> str:
        return (
            f"<Accident(id={self.id}, ext={self.external_id}, "
            f"dt={self.datetime}, severity={self.severity!r})>"
        )


# =====================================================================
# VEHICLE — транспортное средство, участвовавшее в ДТП
# =====================================================================


class Vehicle(Base):
    """ТС — машина, мотоцикл, автобус и т.п."""

    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    accident_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("accidents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # === Номер машины внутри ДТП (1, 2, 3...) ===
    n: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="N — номер ТС в этом ДТП"
    )

    # === Идентификация ТС ===
    mark: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True, comment="MARK — марка"
    )
    model: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True, comment="MODEL — модель"
    )
    vehicle_year: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="VEHICLE_YEAR — год выпуска"
    )
    color: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    prod_type: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="PROD_TYPE — категория ТС"
    )

    # === Собственность и регистрация ===
    okfs: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="OKFS — форма собственности"
    )
    okopf: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="OKOPF — организационно-правовая форма владельца"
    )

    # === Привод и техническое состояние ===
    rudder_type: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="RUDDER_TYPE — тип привода (передний/задний/полный)",
    )
    tech_failure_type: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, comment="TECH_FAILURE_TYPE — технические неисправности"
    )
    damage_disposition: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="DAMAGE_DISPOSITION — расположение повреждений"
    )

    # === Поведение водителя ===
    escape: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="ESCAPE — скрылся ли с места ДТП"
    )

    # === Поля для обогащения справочником (заполняем позже) ===
    is_right_hand_drive: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True, comment="Правый руль (вычисляется по справочнику марок)"
    )
    steering_confidence: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Уверенность определения: high/medium/unknown/not_applicable",
    )

    # === Связи ===
    accident: Mapped["Accident"] = relationship(back_populates="vehicles")
    participants: Mapped[List["Participant"]] = relationship(
        back_populates="vehicle", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Vehicle(id={self.id}, mark={self.mark!r}, model={self.model!r}, year={self.vehicle_year})>"


# =====================================================================
# Базовый класс для участников — общие поля Participant и AccidentPedestrian
# =====================================================================
# Делаем без Mixin для простоты — повторяем поля в каждом классе.
# Это даёт более понятный код и независимость классов.


# =====================================================================
# PARTICIPANT — водитель или пассажир внутри ТС
# =====================================================================


class Participant(Base):
    """Участник внутри ТС (водитель/пассажир)."""

    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    vehicle_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("vehicles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # === Идентификация участника ===
    n: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="N — номер участника внутри ТС"
    )
    part_type: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        index=True,
        comment="PART_TYPE — роль (Водитель/Пассажир/...)",
    )

    # === Демография ===
    sex: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    driver_service_length: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="DRIVER_SERVICE_LENGTH — стаж вождения (только водители)",
    )

    # === Здоровье ===
    hv_type: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="HV_TYPE — состояние здоровья после ДТП"
    )
    injured_card: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="INJURED_CARD — медицинская карточка пострадавшего"
    )

    # === Безопасность ===
    safety_belt: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, comment="SAFETY_BELT — был ли пристёгнут (Да/Нет)"
    )
    child_safety_type: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="CHILD_SAFETY_TYPE — тип детского удерживающего устройства",
    )

    # === Алкоголь ===
    med_result_permille: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 3),
        nullable=True,
        index=True,
        comment="MED_RESULT_PERMILLE — промилле алкоголя (только водители)",
    )

    # === Поведение ===
    escape: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="ESCAPE — скрылся ли с места"
    )

    # === Нарушения ПДД ===
    main_pdd_violation: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, comment="MAIN_PDD_DERANGEMENT — основное нарушение ПДД"
    )
    attendant_pdd_violation: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="ATTENDANT_PDD_DERANGEMENT — сопутствующее нарушение",
    )

    # === Возраст из Telegram NLP ===
    age_from_telegram: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Возраст из NER Telegram-поста (Дни 5+8)"
    )
    age_source: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Источник: telegram_gold / telegram_high_precision",
    )
    age_match_context: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="NER-контекст возраста (водитель/пенсионерка/...) — аудит",
    )
    age_match_post_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="tg_id поста — для трассировки"
    )

    # === Связи ===
    vehicle: Mapped["Vehicle"] = relationship(back_populates="participants")

    def __repr__(self) -> str:
        return f"<Participant(id={self.id}, part_type={self.part_type!r}, sex={self.sex!r})>"


# =====================================================================
# ACCIDENT_PEDESTRIAN — участник ДТП без ТС (обычно пешеход)
# =====================================================================


class AccidentPedestrian(Base):
    """Пешеход или другой участник ДТП без транспортного средства.

    В источнике лежит в массиве PERSONS на верхнем уровне ДТП
    (а не внутри VEHICLES[*].PERSONS).
    """

    __tablename__ = "accident_pedestrians"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    accident_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("accidents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    n: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    part_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True, index=True)

    sex: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    driver_service_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    hv_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    injured_card: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    safety_belt: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    child_safety_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    med_result_permille: Mapped[Optional[float]] = mapped_column(Numeric(5, 3), nullable=True)

    escape: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    main_pdd_violation: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    attendant_pdd_violation: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # === Возраст из Telegram NLP ===
    age_from_telegram: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Возраст из NER Telegram-поста (Дни 5+8)"
    )
    age_source: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Источник: telegram_gold / telegram_high_precision",
    )
    age_match_context: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="NER-контекст возраста (пешеход/мальчик/...) — аудит",
    )
    age_match_post_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="tg_id поста — для трассировки"
    )

    accident: Mapped["Accident"] = relationship(back_populates="pedestrians")

    def __repr__(self) -> str:
        return f"<AccidentPedestrian(id={self.id}, part_type={self.part_type!r})>"


# =====================================================================
# DATA_UPDATE_LOG — журнал загрузок
# =====================================================================


class DataUpdateLog(Base):
    """Журнал загрузок и обновлений данных."""

    __tablename__ = "data_updates_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    source: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Источник: dtp-stat-accumulated / mvd-25"
    )
    records_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="success / error / running"
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<DataUpdateLog(id={self.id}, source={self.source!r}, "
            f"status={self.status!r}, added={self.records_added})>"
        )


# =====================================================================
# REGION_BOUNDARY — географическая граница региона (для PostGIS-фильтра)
# =====================================================================


class RegionBoundary(Base):
    """Полигон границы региона (для фильтрации точек, кластеризации, аналитики).

    Источник для Приморского края: OSM admin_boundary (relation 151225) ∩
    OSM land_polygons (osmdata.openstreetmap.de). Это даёт чистую сухопутную
    границу с островами Русский, Попова, Путятина, Аскольд (без 12-мильной
    территориальной акватории, которая входит в admin_boundary).
    """

    __tablename__ = "region_boundaries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    region_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        comment="Уникальный код (primorye_krai_land и т.п.) — для идемпотентного UPSERT",
    )
    display_name: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Человекочитаемое название (Приморский край, суша)",
    )
    source: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Источник полигона с датой выгрузки — для аудита",
    )

    geom: Mapped[str] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326),
        nullable=False,
        comment="MultiPolygon WGS84 — материк + острова",
    )

    area_km2: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2),
        nullable=True,
        comment="Площадь км² (вычислена в UTM zone 52N для минимума искажений)",
    )

    raw_meta: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="osm_relation_id, дата OSM-выгрузки, метод обработки и т.п.",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<RegionBoundary(id={self.id}, region_name={self.region_name!r}, "
            f"area_km2={self.area_km2})>"
        )


# =====================================================================
# TASK_RUN — журнал запусков Celery-задач
# =====================================================================


class TaskRun(Base):
    """Аудит каждого запуска Celery-задачи.

    Пишется в начале (status='running') и обновляется в конце
    ('success'/'error'). Позволяет отвечать на вопросы:
    - когда последний раз отрабатывал weekly-парсер?
    - сколько в среднем длится prophet retrain?
    - какие запуски падали и почему?
    """

    __tablename__ = "task_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="running / success / error / skipped",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, index=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Параметры запуска + summary результата",
    )

    def __repr__(self) -> str:
        return (
            f"<TaskRun(id={self.id}, task={self.task_name!r}, "
            f"status={self.status!r}, dur_ms={self.duration_ms})>"
        )


# =====================================================================
# MODEL_VERSION — версионирование ML-моделей
# =====================================================================


class ModelVersion(Base):
    """Снапшот ML-модели после очередного retrain.

    Один model_name (например, 'prophet_dtp') может иметь много
    версий с разными trained_at. Ровно одна запись имеет
    is_current=TRUE — это enforced'ится partial unique index'ом
    `uq_model_versions_current` (см. миграцию 814aaf06d2cb).

    metadata_json содержит метрики (MAPE/F1/ECE/holdout_period) +
    hyperparams — это и audit trail, и основа для сравнения версий
    (например, новый retrain хуже старого по F1 → блокируем swap).
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    version_path: Mapped[str] = mapped_column(Text, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    train_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<ModelVersion(id={self.id}, name={self.model_name!r}, "
            f"trained_at={self.trained_at}, current={self.is_current})>"
        )


# =====================================================================
# OSM_ROAD — OSM road graph для snap-to-road
# =====================================================================


class OsmRoad(Base):
    """Ребро OSM road graph для Приморского края.

    Источник: Geofabrik PBF (russia-far-eastern-fed-district-latest.osm.pbf),
    извлечение через pyrosm с фильтром highway IN (motorway, trunk, primary,
    secondary, tertiary, unclassified, residential, service).

    Используется в snap-to-road: для каждой точки ДТП находим ближайшее
    ребро через ST_ClosestPoint + ST_Distance(geography), и если расстояние
    ≤ 50 м — переносим точку на это ребро.
    """

    __tablename__ = "osm_roads"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    osm_way_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    highway_class: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    oneway: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    maxspeed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    geom: Mapped[str] = mapped_column(
        Geometry(geometry_type="LINESTRING", srid=4326),
        nullable=False,
    )
    length_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tags: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<OsmRoad(id={self.id}, osm_way={self.osm_way_id}, "
            f"class={self.highway_class!r}, name={self.name!r})>"
        )
