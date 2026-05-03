"""
Pydantic v2 модели для request/response FastAPI.

Сгруппированы по тематикам эндпоинтов:
- Accidents: AccidentSummary, AccidentDetail, ParticipantOut, VehicleOut, PedestrianOut, AccidentsStats
- Clusters: HotspotOut, HotspotsResponse
- Predict: SeverityPredictRequest, SeverityPredictResponse
- Forecast: MonthlyForecastPoint, MonthlyForecastResponse
- NLP: TopicOut, TopicsResponse, PostTopicMatch

Pydantic 2 синтаксис:
- model_config = ConfigDict(...) вместо class Config
- Field(..., description=...) для OpenAPI auto-doc
- @field_validator для проверок (если нужно)

Все схемы — readonly response models, кроме SeverityPredictRequest.
ConfigDict(from_attributes=True) включает SQLAlchemy ORM-mode.
"""

from __future__ import annotations

# Импортируем datetime под алиасом DateTime, чтобы можно было использовать
# имя поля `datetime` в Pydantic-моделях (Pydantic 2 ругается на коллизию
# имени поля с именем типа в class namespace).
from datetime import datetime as DateTime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# =====================================================================
# Базовые типы
# =====================================================================

SeverityClass = Literal["light", "severe", "severe_multiple", "dead"]


class GeoPoint(BaseModel):
    """Координаты ДТП в WGS84 (EPSG:4326)."""

    model_config = ConfigDict(json_schema_extra={"example": {"lat": 43.1155, "lon": 131.8855}})

    lat: float = Field(..., description="Широта", ge=-90, le=90)
    lon: float = Field(..., description="Долгота", ge=-180, le=180)


# =====================================================================
# Accidents
# =====================================================================


class ParticipantOut(BaseModel):
    """Участник внутри ТС (водитель / пассажир)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    n: int | None = Field(None, description="Номер участника внутри ТС")
    part_type: str | None = Field(None, description="Роль: Водитель / Пассажир / ...")
    sex: str | None = Field(None, description="Пол: Мужской / Женский")
    driver_service_length: int | None = Field(None, description="Стаж вождения, лет")
    hv_type: str | None = Field(None, description="Состояние здоровья после ДТП")
    safety_belt: str | None = Field(None, description="Был ли пристёгнут (Да/Нет)")
    med_result_permille: float | None = Field(None, description="Промилле алкоголя")
    age_from_telegram: int | None = Field(None, description="Возраст из Telegram NER")
    age_source: str | None = Field(None, description="Источник возраста")


class VehicleOut(BaseModel):
    """ТС-участник ДТП с участниками."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    n: int | None
    mark: str | None = Field(None, description="Марка")
    model: str | None = Field(None, description="Модель")
    vehicle_year: int | None = Field(None, description="Год выпуска")
    color: str | None = None
    prod_type: str | None = Field(None, description="Категория ТС")
    is_right_hand_drive: bool | None = Field(None, description="Правый руль (RHD)")
    steering_confidence: str | None = Field(
        None, description="Уверенность RHD/LHD: high/medium/..."
    )
    participants: list[ParticipantOut] = Field(default_factory=list)


class PedestrianOut(BaseModel):
    """Пешеход / участник без ТС."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    n: int | None
    part_type: str | None
    sex: str | None
    hv_type: str | None
    age_from_telegram: int | None
    age_source: str | None


class AccidentSummary(BaseModel):
    """Краткая запись ДТП — для list-эндпоинта."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="ID в БД (PK)")
    external_id: int = Field(..., description="EM_NUMBER из источника")
    datetime: DateTime = Field(..., description="Дата и время ДТП")
    place: str | None = Field(None, description="Городской округ или район")
    np: str | None = Field(None, description="Населённый пункт")
    street: str | None = None
    em_type: str | None = Field(None, description="Тип ДТП (Столкновение, Наезд на пешехода, ...)")
    severity: SeverityClass | None = Field(None, description="Тяжесть")
    veh_amount: int = Field(..., description="Количество ТС")
    pers_amount: int = Field(..., description="Количество участников")
    lost_amount: int = Field(..., description="Погибло")
    suffer_amount: int = Field(..., description="Ранено")
    point: GeoPoint | None = Field(None, description="Координаты")
    is_in_region: bool | None = Field(None, description="Внутри сухопутной границы Приморья")


class AccidentDetail(AccidentSummary):
    """Полная запись ДТП — для эндпоинта /accidents/{id}."""

    parent_region: str | None = None
    house: str | None = None
    roads: str | None = Field(None, description="Название трассы (если трассовое)")
    road_km: float | None = None
    light_type: str | None = Field(None, description="Освещение/время суток")
    traffic_area_state: str | None = Field(None, description="Состояние покрытия")
    mt_rate: str | None = None
    clouds: list[str] | None = Field(None, description="Погода (массив)")
    defects: list[str] | None = Field(None, description="Дефекты покрытия")
    motion_influences: list[str] | None = None
    schema_id: int | None = None
    children_attr: int | None = None
    vehicles: list[VehicleOut] = Field(default_factory=list)
    pedestrians: list[PedestrianOut] = Field(default_factory=list)


class AccidentsListResponse(BaseModel):
    """Пагинированный список ДТП."""

    total: int = Field(..., description="Всего записей по фильтру")
    limit: int
    offset: int
    items: list[AccidentSummary]


class AccidentsStatsResponse(BaseModel):
    """Агрегаты для дашборда."""

    total: int = Field(..., description="Всего ДТП в БД")
    in_region: int = Field(..., description="ДТП с координатами в сухопутной границе Приморья")
    by_severity: dict[str, int] = Field(..., description="Количество ДТП по severity")
    by_em_type: dict[str, int] = Field(..., description="ТОП-10 типов ДТП по частоте")
    by_year: dict[int, int] = Field(..., description="Количество ДТП по годам")
    total_dead: int = Field(..., description="Сумма погибших")
    total_suffered: int = Field(..., description="Сумма раненых")


# =====================================================================
# Clusters / Hotspots
# =====================================================================


class HotspotOut(BaseModel):
    """Очаг аварийности (DBSCAN-кластер)."""

    rank: int = Field(..., description="Ранг очага по количеству ДТП (1 = крупнейший)")
    cluster_id: int = Field(..., description="ID кластера из DBSCAN")
    n_points: int = Field(..., description="Количество ДТП в очаге")
    centroid: GeoPoint
    radius_meters: float = Field(..., description="95-перцентиль расстояния от центроида")
    median_distance_meters: float = Field(..., description="Медианное расстояние от центроида")
    pct_dead: float = Field(..., description="Доля смертельных ДТП (0..1)")
    pct_severe_or_dead: float = Field(..., description="Доля severe + severe_multiple + dead")
    severity_distribution: dict[str, int]
    top_em_types: list[tuple[str, int]] = Field(..., description="ТОП-5 типов ДТП в очаге")
    top_np: list[tuple[str, int]] = Field(..., description="ТОП-5 населённых пунктов")


class HotspotsResponse(BaseModel):
    """Список очагов с метаданными DBSCAN-прогона."""

    params: dict[str, Any] = Field(..., description="Параметры DBSCAN (eps, min_samples, ...)")
    stats: dict[str, Any] = Field(..., description="Агрегатная статистика DBSCAN")
    items: list[HotspotOut]


# =====================================================================
# Predict severity
# =====================================================================


class SeverityPredictRequest(BaseModel):
    """Признаки одного ДТП для предсказания severity.

    Все 34 признака CatBoost v2. Ни один не обязателен по типу
    Optional, но НЕ передавать обязательные в production-сценарии —
    это усреднит точность.

    Категориальные (light_type, traffic_area_state, mt_rate, clouds_top,
    em_type, np_top, mark_top) — допускают строку 'unknown' если значение
    не известно (CatBoost обучен на 'unknown' как валидной категории).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hour": 22,
                "dow": 5,
                "month": 11,
                "year": 2026,
                "lat": 43.115,
                "lon": 131.886,
                "veh_amount": 1,
                "veh_count_actual": 1,
                "rhd_share": 1.0,
                "classified_veh": 1.0,
                "avg_vehicle_year": 2008.0,
                "part_count": 2,
                "drunk_share": 0.0,
                "med_known_count": 1.0,
                "unbelted_share": 0.5,
                "avg_age_from_tg": None,
                "ped_count": 1,
                "avg_ped_age_from_tg": None,
                "light_type": "В темное время суток, освещение не включено",
                "traffic_area_state": "Сухое",
                "mt_rate": "В населенном пункте",
                "clouds_top": "Ясно",
                "em_type": "Наезд на пешехода",
                "np_top": "г Владивосток",
                "mark_top": "TOYOTA",
                "is_weekend": True,
                "is_holiday": False,
                "is_highway": False,
                "is_in_region": True,
                "has_defect": False,
                "has_moto": False,
                "has_truck_or_bus": False,
                "has_known_age": False,
                "has_known_ped_age": False,
            }
        }
    )

    # Числовые
    hour: int = Field(..., ge=0, le=23, description="Час суток 0..23")
    dow: int = Field(..., ge=0, le=6, description="День недели 0=Вс..6=Сб (PostgreSQL DOW)")
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2010, le=2035)
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    veh_amount: int = Field(..., ge=0, description="Количество ТС в ДТП")
    veh_count_actual: int = Field(..., ge=0, description="Фактическое количество ТС в БД")
    rhd_share: float | None = Field(
        None, ge=0.0, le=1.0, description="Доля RHD среди классифицированных ТС"
    )
    classified_veh: float | None = Field(
        None, ge=0, description="Сколько ТС прошло steering-классификацию"
    )
    avg_vehicle_year: float | None = Field(None, description="Средний год выпуска ТС")
    part_count: int = Field(..., ge=0, description="Количество participants в БД")
    drunk_share: float | None = Field(None, ge=0.0, le=1.0)
    med_known_count: float | None = Field(None, ge=0)
    unbelted_share: float | None = Field(None, ge=0.0, le=1.0)
    avg_age_from_tg: float | None = Field(None, ge=0.0, le=120)
    ped_count: int = Field(..., ge=0, description="Количество пешеходов")
    avg_ped_age_from_tg: float | None = Field(None, ge=0.0, le=120)

    # Категориальные (string)
    light_type: str = Field("unknown", description="Освещение")
    traffic_area_state: str = Field("unknown", description="Состояние покрытия")
    mt_rate: str = Field("unknown")
    clouds_top: str = Field("unknown", description="Первое значение clouds JSONB")
    em_type: str = Field("unknown", description="Тип ДТП")
    np_top: str = Field("other", description="Населённый пункт (топ-30 + 'other')")
    mark_top: str = Field("other", description="Доминирующая марка ТС (топ-15 + 'other')")

    # Bool
    is_weekend: bool
    is_holiday: bool
    is_highway: bool
    is_in_region: bool
    has_defect: bool
    has_moto: bool
    has_truck_or_bus: bool
    has_known_age: bool
    has_known_ped_age: bool


class SeverityPredictResponse(BaseModel):
    """4 калиброванные вероятности + предсказанный класс."""

    # Pydantic v2 защищает namespace "model_*" — отключаем для поля model_version
    model_config = ConfigDict(protected_namespaces=())

    predicted_class: SeverityClass = Field(..., description="Класс с максимальной вероятностью")
    probabilities: dict[str, float] = Field(
        ..., description="Калиброванные вероятности всех 4 классов (сумма ≈ 1)"
    )
    calibration_method: str = Field(
        "isotonic_per_class", description="Метод калибровки (isotonic per class one-vs-rest)"
    )
    model_version: str = Field("catboost_severity_v2 + isotonic_v1", description="Версия модели")


# =====================================================================
# Forecast (Prophet)
# =====================================================================


class MonthlyForecastPoint(BaseModel):
    """Один месяц прогноза."""

    ds: DateTime = Field(..., description="Начало месяца (1-е число)")
    yhat: float = Field(..., description="Точечный прогноз количества ДТП за месяц")
    yhat_lower: float = Field(..., description="Нижняя граница 95% CI")
    yhat_upper: float = Field(..., description="Верхняя граница 95% CI")
    actual: int | None = Field(None, description="Факт из БД, если есть")


class MonthlyForecastResponse(BaseModel):
    """Прогноз количества ДТП по месяцам."""

    horizon_months: int = Field(..., description="Количество месяцев в прогнозе")
    items: list[MonthlyForecastPoint]
    metadata: dict[str, Any] = Field(..., description="Train range, MAPE на hold-out 2025 и пр.")


# =====================================================================
# NLP / BERTopic
# =====================================================================


class TopicEmTypeLink(BaseModel):
    """Связь BERTopic-темы с категориями em_type из БД (через 482 gold-пары)."""

    n_gold_posts: int = Field(
        ..., description="Количество постов в этой теме, имеющих gold-матч с ДТП"
    )
    shares_pct: dict[str, float] = Field(..., description="Доли em_type в %")
    dominant_em_type: str | None = Field(None, description="Самая частая em_type в теме")
    dominant_share_pct: float | None = None


class TopicOut(BaseModel):
    """Описание одной BERTopic-темы."""

    topic_id: int = Field(..., description="ID темы (-1 = шум)")
    size: int = Field(..., description="Количество постов в теме")
    share_pct: float = Field(..., description="Доля от корпуса в %")
    name: str = Field(..., description="Авто-название (топ-слова)")
    label: str | None = Field(None, description="Человекочитаемое описание (при наличии)")
    top_words: list[str] = Field(..., description="Топ-10 ключевых слов c-TF-IDF")
    examples: list[str] = Field(default_factory=list, description="5 примеров постов")
    em_type_link: TopicEmTypeLink | None = Field(
        None, description="Связь с em_type через gold-пары"
    )


class TopicsResponse(BaseModel):
    """Полный список тем + метаданные модели."""

    n_posts: int
    n_topics: int = Field(..., description="Реальные темы (без шума)")
    n_noise: int
    noise_share_pct: float
    embedding_model: str
    items: list[TopicOut]


class SemanticSearchHit(BaseModel):
    """Один результат semantic search'а постов."""

    tg_id: int
    similarity: float = Field(..., description="Cosine similarity ∈ [-1, 1]")
    topic_id: int | None = Field(
        None, description="Тема из BERTopic (если есть в topic_assignments)"
    )
    text_preview: str | None = Field(None, description="Первые 300 символов поста")


class SemanticSearchResponse(BaseModel):
    """Top-k наиболее похожих Telegram-постов на текст-запрос."""

    query: str
    top_k: int
    embedding_model: str = Field("paraphrase-multilingual-MiniLM-L12-v2")
    elapsed_ms: float
    items: list[SemanticSearchHit]


class PostTopicMatch(BaseModel):
    """Тема одного Telegram-поста + (опционально) связанная запись ДТП."""

    tg_id: int
    topic_id: int
    topic_name: str | None = None
    topic_label: str | None = None
    post_text_preview: str | None = Field(None, description="Первые 300 символов поста")
    matched_accident_id: int | None = Field(None, description="accident.id из gold-пар, если есть")
    matched_accident_external_id: int | None = None


# =====================================================================
# Health
# =====================================================================


class HealthResponse(BaseModel):
    """/health — sanity-check состояния сервиса."""

    status: Literal["ok", "degraded"]
    db_connected: bool
    accidents_count: int
    models_loaded: dict[str, bool]
    version: str


# =====================================================================
# Recommendations (СППР,6)
# =====================================================================


class CitationOut(BaseModel):
    """Ссылка на первичный источник CMF (Cochrane SR / FHWA PSC / etc.)."""

    label: str
    url: str
    quoted_effect: str


class EffectEstimateOut(BaseModel):
    """Количественная оценка эффекта меры с 95 % CI."""

    metric: str = Field(..., description="Что метрит: fatal_pedestrian_crashes, all_crashes, etc.")
    point_estimate: float = Field(..., description="-0.30 = снижение на 30 %")
    ci_low: float
    ci_high: float
    note: str = Field("", description="Уточнение применимости")


class RecommendationOut(BaseModel):
    """Готовая рекомендация для popup'а / panel'и."""

    rule_id: str = Field(..., description="R01_speed_limit_30 и т.д.")
    priority: int = Field(..., ge=1, le=3, description="1=критично, 2=важно, 3=превентивно")
    title: str
    icon: str = Field(..., description="Material symbol, например :material/speed:")
    trigger_human: str = Field(..., description="Человекочитаемое: «Сработало потому что ...»")
    expected_effect: EffectEstimateOut
    evidence_basis: list[CitationOut] = Field(..., description="≥1 источник")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Применимость к Приморью 0..1")
    implementation_cost: Literal["low", "medium", "high"]
    target_severity: list[str] = Field(default_factory=list)
    target_em_types: list[str] = Field(default_factory=list)
    score: float = Field(..., description="priority_weight × confidence × |effect|")


class HotspotProfileOut(BaseModel):
    """Сводка профиля очага, на которой запускался rule engine."""

    n_points: int
    radius_meters: float
    pct_dead: float
    pct_severe_or_dead: float
    top_em_type: str | None = None
    top_em_types: list[tuple[str, int]] = Field(default_factory=list)
    top_np: str | None = None
    dominant_light_type: str | None = None
    dominant_state: str | None = None
    is_highway: bool
    is_in_city: bool
    has_night_dominant: bool = False
    has_winter_spike: bool = False
    centroid_lat: float | None = None
    centroid_lon: float | None = None
    source: str = Field(..., description="dbscan | dynamic_radius")
    cluster_id: int | None = None
    radius_query_m: int | None = None


class RecommendationsResponse(BaseModel):
    """Топ-K рекомендаций + профиль, на котором они запускались."""

    profile: HotspotProfileOut
    items: list[RecommendationOut]
    note: str | None = Field(
        None, description="Замечание (например, 'sample size <10 — увеличь радиус')"
    )


# =====================================================================
# Counterfactual через CatBoost
# =====================================================================


class CounterfactualScenario(BaseModel):
    """Один сценарий: имя + override-ы признаков baseline'а.

    overrides — частичный словарь {feature_name: new_value}. Поля,
    которых нет в overrides, берутся из baseline'а.
    """

    name: str = Field(..., description="«Все пристёгнуты», «Трезвый водитель», ...")
    overrides: dict[str, Any] = Field(
        ...,
        description="Перезаписываемые поля baseline'а",
        json_schema_extra={"example": {"unbelted_share": 0.0}},
    )


class CounterfactualRequest(BaseModel):
    """Запрос: baseline 34-фичи + список сценариев."""

    baseline: SeverityPredictRequest
    scenarios: list[CounterfactualScenario] = Field(..., min_length=1)


class CounterfactualResult(BaseModel):
    """Один результат counterfactual'а — modified vs baseline."""

    name: str
    modified_proba: dict[str, float] = Field(
        ..., description="Калиброванные вероятности 4 классов после применения override'ов"
    )
    delta_proba: dict[str, float] = Field(
        ..., description="modified − baseline (отрицательное = снижение risk)"
    )
    delta_dead_pct_points: float = Field(
        ..., description="Удобно: (modified − baseline) для класса dead, в п.п."
    )


class CounterfactualResponse(BaseModel):
    """Полный ответ /predict/severity_counterfactual."""

    model_config = ConfigDict(protected_namespaces=())

    baseline_proba: dict[str, float] = Field(
        ..., description="Калиброванные вероятности 4 классов для baseline'а"
    )
    baseline_predicted_class: SeverityClass
    scenarios: list[CounterfactualResult]
    model_version: str = Field("catboost_severity_v2 + isotonic_v1")
