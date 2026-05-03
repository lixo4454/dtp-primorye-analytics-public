"""Rule engine для рекомендаций ГИБДД на основе 18 правил с evidence base.

Реализация раздела 3 ``docs/recommendations_methodology.md``. Каждое
правило R01..R18 — отдельный класс наследник :class:`BaseRule` с
двумя методами:

* :py:meth:`BaseRule.applies_to` — boolean-триггер по полям
  :class:`HotspotProfile`.
* :py:meth:`BaseRule.evaluate` — возвращает :class:`Recommendation`
  с количественным эффектом, источником и уровнем confidence.

Каждый CMF (crash modification factor) в ``expected_effect`` ОБЯЗАН
ссылаться на конкретный пункт в methodology (Cochrane SR / FHWA PSC /
CMF Clearinghouse / iRAP / TRB NCHRP). Не нашёл meta-analysis →
правило не реализуется или ставится ``confidence = 0.5`` с
qualitative-эффектом. Это страховка от «галлюцинаций».

Группы правил:

* **A. Скорость** (R01-R04): лимит 30, камеры, шумовые полосы,
  лежачие полицейские
* **B. Освещение** (R05-R07): пешеходный переход, трасса, делиниаторы
* **C. Геометрия** (R08-R11): roundabout, cable barrier, HFST,
  guardrail
* **D. Светофоры** (R12-R13): LPI, координация
* **E. Уязвимые** (R14-R15): велодорожка, RRFB
* **F. Поведение** (R16-R17): enforcement ремней, RBT
* **G. Зима** (R18): anti-icing brine

Использование::

    engine = RuleEngine()
    recs = engine.recommend(profile, top_k=5)
    for r in recs:
        print(r.rule_id, r.title, r.expected_effect.point_estimate)

Score'инг (раздел 2.2 methodology):

    score = priority_weight × confidence × abs(effect)
    priority_weight = {1: 3.0, 2: 1.5, 3: 1.0}

Это даёт сильное предпочтение критичным мерам с высокой уверенностью
и большим эффектом, не вытесняя превентивные с малым.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from src.analysis.spot_profile import (
    DARK_NO_LIGHT_TYPES,
    EM_TYPE_BIKE,
    EM_TYPE_COLLISION,
    EM_TYPE_PARKED,
    EM_TYPE_PED,
    EM_TYPE_ROADSIDE,
    EM_TYPE_ROLLOVER,
    NIGHT_LIGHT_TYPES,
    STATE_ICE,
    STATE_SNOW_PACK,
    STATE_SNOWY,
    STATE_WET,
    WINTER_SLIPPERY_STATES,
    HotspotProfile,
)

# =====================================================================
# Data-классы Recommendation, EffectEstimate, Citation
# =====================================================================


@dataclass(frozen=True)
class Citation:
    """Ссылка на первичный источник CMF."""

    label: str
    url: str
    quoted_effect: str


@dataclass(frozen=True)
class EffectEstimate:
    """Количественная оценка эффекта меры (CMF в обратной форме).

    ``point_estimate``: -0.30 = снижение на 30 %; +0.05 = рост на 5 %
    (для контр-показанных мер — но мы их в каталоге не держим).

    ``ci_low``, ``ci_high``: 95 % CI границы того же знака. Если у
    источника CI не публиковали — заполняем ``ci_low = ci_high =
    point_estimate`` и снижаем confidence.
    """

    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    note: str = ""


@dataclass(frozen=True)
class Recommendation:
    """Готовая рекомендация для popup'а / panel'и."""

    rule_id: str
    priority: int  # 1..3
    title: str
    icon: str
    trigger_human: str
    expected_effect: EffectEstimate
    evidence_basis: list[Citation]
    confidence: float
    implementation_cost: str  # "low" | "medium" | "high"
    target_severity: list[str] = field(default_factory=list)
    target_em_types: list[str] = field(default_factory=list)


PRIORITY_WEIGHTS = {1: 3.0, 2: 1.5, 3: 1.0}


def score_recommendation(rec: Recommendation) -> float:
    """``priority_weight × confidence × |effect|`` — раздел 2.2 methodology."""
    return (
        PRIORITY_WEIGHTS.get(rec.priority, 1.0)
        * rec.confidence
        * abs(rec.expected_effect.point_estimate)
    )


# =====================================================================
# Базовый класс правила
# =====================================================================


class BaseRule(ABC):
    """Абстрактный класс правила. У наследников должны быть:

    * ``rule_id``: ClassVar строка
    * ``applies_to``: возвращает True если правило срабатывает
      на этом профиле
    * ``evaluate``: возвращает :class:`Recommendation` (только если
      ``applies_to`` вернул True)
    """

    rule_id: ClassVar[str] = ""

    @abstractmethod
    def applies_to(self, p: HotspotProfile) -> bool: ...

    @abstractmethod
    def evaluate(self, p: HotspotProfile) -> Recommendation: ...


# =====================================================================
# Группа A — Управление скоростью (R01-R04)
# =====================================================================


class R01_SpeedLimit30(BaseRule):
    """Снижение лимита скорости до 30 км/ч в зонах ходьбы.

    Trigger: пешеходные ДТП в населённом пункте + pct_dead >= 3 %.
    Effect: WHO Streets for Life 2021 — kinetic energy ~ V², пешеход
    выживает на 30 км/ч в 87 % случаев против 60 % при 50 км/ч.
    """

    rule_id: ClassVar[str] = "R01_speed_limit_30"

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.top_em_type == EM_TYPE_PED and p.is_in_city and p.pct_dead >= 0.03

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=1,
            title="Снижение лимита скорости до 30 км/ч",
            icon=":material/speed:",
            trigger_human=(
                f"Доминирующий тип ДТП — наезд на пешехода ({p.top_em_type}), "
                f"в населённом пункте, pct_dead = {p.pct_dead:.1%} ≥ 3 %"
            ),
            expected_effect=EffectEstimate(
                metric="fatal_pedestrian_crashes",
                point_estimate=-0.40,
                ci_low=-0.24,
                ci_high=-0.63,
                note="WHO Streets for Life: 30 vs 50 км/ч — выживаемость 87 % vs 60 %",
            ),
            evidence_basis=[
                Citation(
                    label="WHO Streets for Life 2021",
                    url="https://www.who.int/news/item/17-05-2021-streets-for-life-campaign-calls-for-30-km-h-urban-streets-to-ensure-safe-healthy-green-and-liveable-cities",
                    quoted_effect="Implementation of 30 km/h: −24% to −63% fatal pedestrian crashes",
                ),
                Citation(
                    label="Rosén & Sander 2009",
                    url="https://doi.org/10.1016/j.aap.2009.01.011",
                    quoted_effect="Pedestrian fatality risk as function of impact speed",
                ),
            ],
            confidence=0.85,
            implementation_cost="low",
            target_severity=["dead", "severe"],
            target_em_types=[EM_TYPE_PED],
        )


class R02_SpeedCameras(BaseRule):
    """Камеры фотофиксации скорости.

    Trigger: pct_severe_or_dead ≥ 20 % И (трасса ИЛИ топ-3 крупных НП).
    Effect: Cochrane SR Wilson 2010 — −19 % всех, −44 % тяжёлых.
    """

    rule_id: ClassVar[str] = "R02_speed_cameras"
    LARGE_NPS = {"г Владивосток", "г Находка", "г Уссурийск"}

    def applies_to(self, p: HotspotProfile) -> bool:
        big_np = bool(p.top_np and any(np.lower() in p.top_np.lower() for np in self.LARGE_NPS))
        return p.pct_severe_or_dead >= 0.20 and (p.is_highway or big_np)

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        priority = 1 if p.pct_dead >= 0.05 else 2
        return Recommendation(
            rule_id=self.rule_id,
            priority=priority,
            title="Камеры фотофиксации скорости",
            icon=":material/photo_camera:",
            trigger_human=(
                f"pct_severe_or_dead = {p.pct_severe_or_dead:.1%} ≥ 20 %, "
                f"{'трасса' if p.is_highway else 'крупный НП ' + (p.top_np or '')}"
            ),
            expected_effect=EffectEstimate(
                metric="all_crashes",
                point_estimate=-0.19,
                ci_low=-0.08,
                ci_high=-0.44,
                note="Cochrane SR Wilson 2010, 35 RCT/before-after studies",
            ),
            evidence_basis=[
                Citation(
                    label="Cochrane SR CD004607 Wilson 2010",
                    url="https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.CD004607.pub4/abstract",
                    quoted_effect="−8% to −49% all crashes; −44% serious/fatal в vicinity камеры",
                )
            ],
            confidence=0.80,
            implementation_cost="medium",
            target_severity=["dead", "severe", "severe_multiple"],
            target_em_types=[],
        )


class R03_RumbleStrips(BaseRule):
    """Поперечные шумовые полосы на approach к перекрёстку/обгонной зоне.

    Trigger: столкновения / наезды на стоящее ТС на трассе.

    ВАЖНО: проверка `radius_meters` имеет смысл ТОЛЬКО для DBSCAN-
    кластеров, у которых radius — статистический spread (95-перцентиль
    расстояния от центроида), и тоже — слабый прокси road-geometry'и.
    Поэтому R03 НЕ триггерится на DBSCAN-источниках (там radius
    = cluster spread, не длина прямого участка дороги). Для
    dynamic_radius (произвольная точка + слайдер) тоже не имеет смысла
    — пользователь сам выбирает radius. Поэтому правило применимо
    ТОЛЬКО при наличии явного road-geometry-сигнала, которого у нас
    в6 нет → правило **временно дисейблено** до интеграции
    OSM road graph.

    Когда road-graph будет: применять при `dominant_road_curvature
    < threshold` AND `road_length >= 1000 m`.

    Effect: iRAP Toolkit + FHWA NCHRP 641 — −30 % всех, −50 % для
    усталых.
    """

    rule_id: ClassVar[str] = "R03_rumble_strips"

    def applies_to(self, p: HotspotProfile) -> bool:
        # Дисейблено до7 (snap-to-road OSM) — radius_meters
        # ни для DBSCAN, ни для dynamic_radius НЕ отражает road geometry.
        # См. docstring и `docs/recommendations_methodology.md` §6.1.
        return False

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=2,
            title="Поперечные шумовые полосы (rumble strips)",
            icon=":material/timeline:",
            trigger_human=(
                f"Тип ДТП «{p.top_em_type}» на трассе (radius ≈ "
                f"{p.radius_meters:.0f} м — длинный прямой участок)"
            ),
            expected_effect=EffectEstimate(
                metric="all_crashes_on_approach",
                point_estimate=-0.30,
                ci_low=-0.20,
                ci_high=-0.60,
                note="−30% всех ДТП на approach, −50% для уснувших водителей",
            ),
            evidence_basis=[
                Citation(
                    label="FHWA NCHRP 641",
                    url="https://www.trb.org/main/blurbs/162962.aspx",
                    quoted_effect="Audible/vibratory pavement markings reduce ROR by 50%",
                ),
                Citation(
                    label="iRAP Rumble Strips",
                    url="https://toolkit.irap.org/safer-roads-treatments/rumble-strips/",
                    quoted_effect="20-60% reduction depending on placement",
                ),
            ],
            confidence=0.75,
            implementation_cost="low",
            target_severity=["severe", "dead"],
            target_em_types=[EM_TYPE_COLLISION, EM_TYPE_PARKED],
        )


class R04_SpeedHumps(BaseRule):
    """Лежачие полицейские в населённом пункте.

    Trigger: пешеходные ДТП в городе, sample size ≥ 30.
    Effect: ETSC 2017 + WHO Save Lives 2017 — −50 % сильных ДТП в зоне.
    """

    rule_id: ClassVar[str] = "R04_speed_humps"

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.is_in_city and p.top_em_type == EM_TYPE_PED and p.n_points >= 30

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=2,
            title="Лежачие полицейские (искусственные неровности)",
            icon=":material/horizontal_rule:",
            trigger_human=(f"Пешеходные ДТП в населённом пункте, n = {p.n_points}"),
            expected_effect=EffectEstimate(
                metric="severe_crashes_in_zone",
                point_estimate=-0.50,
                ci_low=-0.30,
                ci_high=-0.70,
                note="Эффект ослабевает на трассах — водители объезжают",
            ),
            evidence_basis=[
                Citation(
                    label="ETSC 2017",
                    url="https://etsc.eu/wp-content/uploads/PIN-AR_2017_FINAL.pdf",
                    quoted_effect="Speed humps cut severe crashes by 30-70%",
                ),
                Citation(
                    label="WHO Save LIVES 2017",
                    url="https://www.who.int/publications/i/item/save-lives-a-road-safety-technical-package",
                    quoted_effect="Traffic calming reduces ped fatalities ~50%",
                ),
            ],
            confidence=0.70,
            implementation_cost="low",
            target_severity=["severe", "dead"],
            target_em_types=[EM_TYPE_PED],
        )


# =====================================================================
# Группа B — Освещение и видимость (R05-R07)
# =====================================================================


class R05_PedestrianLighting(BaseRule):
    """Освещение пешеходного перехода.

    Trigger: пешеходные ДТП ночью без освещения (либо сумерки),
    sample size ≥ 20.
    Effect: FHWA 2022 — −77 % ночных ДТП от категории «низкая→высокая»
    освещённость.
    """

    rule_id: ClassVar[str] = "R05_pedestrian_lighting"

    def applies_to(self, p: HotspotProfile) -> bool:
        if p.top_em_type != EM_TYPE_PED or p.n_points < 20:
            return False
        # либо доминирует тёмное время, либо известный dominant_light тёмный
        if p.has_night_dominant:
            return True
        return p.dominant_light_type in NIGHT_LIGHT_TYPES

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=1,
            title="Освещение пешеходного перехода",
            icon=":material/lightbulb:",
            trigger_human=(
                f"Пешеходные ДТП в тёмное время суток, n = {p.n_points}, "
                f"dominant_light = {p.dominant_light_type or 'ночь'}"
            ),
            expected_effect=EffectEstimate(
                metric="nighttime_pedestrian_crashes",
                point_estimate=-0.77,
                ci_low=-0.68,
                ci_high=-0.86,
                note="FHWA 2022: повышение категории освещённости (low→high)",
            ),
            evidence_basis=[
                Citation(
                    label="FHWA Street Lighting for Pedestrian Safety 2022",
                    url="https://highways.dot.gov/sites/fhwa.dot.gov/files/2022-09/StreetLightingPedestrianSafety.pdf",
                    quoted_effect="−77% nighttime pedestrian crashes",
                ),
                Citation(
                    label="Cochrane Beyer & Ker 2009",
                    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11743125/",
                    quoted_effect="Pooled rate ratio 0.68 для ночных injuries",
                ),
            ],
            confidence=0.85,
            implementation_cost="medium",
            target_severity=["dead", "severe"],
            target_em_types=[EM_TYPE_PED],
        )


class R06_HighwayLighting(BaseRule):
    """Освещение трассового участка с опрокидываниями/съездами.

    Trigger: опрокидывания/съезды на трассе ночью без освещения.
    Effect: Cochrane Beyer 2009 — pooled rate ratio 0.68 (CMF 0.68),
    то есть −32 % ночных ДТП.
    """

    rule_id: ClassVar[str] = "R06_highway_lighting"

    def applies_to(self, p: HotspotProfile) -> bool:
        if not p.is_highway:
            return False
        if p.top_em_type not in {EM_TYPE_ROLLOVER, EM_TYPE_ROADSIDE}:
            return False
        # Тёмное время доминирует ИЛИ dominant_light явно «без освещения»
        if p.has_night_dominant:
            return True
        return p.dominant_light_type in DARK_NO_LIGHT_TYPES

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=2,
            title="Освещение трассового участка",
            icon=":material/wb_incandescent:",
            trigger_human=(
                f"{p.top_em_type} на трассе в тёмное время суток " f"(n = {p.n_points})"
            ),
            expected_effect=EffectEstimate(
                metric="night_crashes_highway",
                point_estimate=-0.32,
                ci_low=-0.20,
                ci_high=-0.45,
                note="Cochrane CMF 0.68 (rate ratio); FHWA recommended estimate",
            ),
            evidence_basis=[
                Citation(
                    label="Cochrane Beyer & Ker 2009",
                    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11743125/",
                    quoted_effect="Pooled rate ratio 0.68 для ночных injury crashes",
                ),
                Citation(
                    label="FHWA-SA-22 Lighting Research",
                    url="https://highways.dot.gov/sites/fhwa.dot.gov/files/2022-09/StreetLightingPedestrianSafety.pdf",
                    quoted_effect="Recommended highway lighting CMF 0.68",
                ),
            ],
            confidence=0.75,
            implementation_cost="high",
            target_severity=["severe", "dead"],
            target_em_types=[EM_TYPE_ROLLOVER, EM_TYPE_ROADSIDE],
        )


class R07_Delineators(BaseRule):
    """Светоотражающие делиниаторы / разметка на разделительной.

    Trigger: ночные столкновения на трассе.
    Effect: FHWA delineator family — CMF 0.85 (−15 %).
    """

    rule_id: ClassVar[str] = "R07_delineators"

    def applies_to(self, p: HotspotProfile) -> bool:
        if not p.is_highway or p.top_em_type != EM_TYPE_COLLISION:
            return False
        if p.has_night_dominant:
            return True
        return p.dominant_light_type in NIGHT_LIGHT_TYPES

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=3,
            title="Светоотражающая разметка / делиниаторы",
            icon=":material/segment:",
            trigger_human=(
                f"Ночные столкновения на трассе "
                f"(n = {p.n_points}, light = {p.dominant_light_type or 'ночь'})"
            ),
            expected_effect=EffectEstimate(
                metric="night_collisions",
                point_estimate=-0.15,
                ci_low=-0.05,
                ci_high=-0.25,
                note="FHWA Roadway Departure Countermeasures, delineator family CMF 0.85",
            ),
            evidence_basis=[
                Citation(
                    label="FHWA Roadway Departure",
                    url="https://highways.dot.gov/safety/proven-safety-countermeasures",
                    quoted_effect="Delineators: CMF 0.85 для ночных ДТП",
                )
            ],
            confidence=0.65,
            implementation_cost="low",
            target_severity=["severe", "dead", "light"],
            target_em_types=[EM_TYPE_COLLISION],
        )


# =====================================================================
# Группа C — Геометрия и инфраструктура (R08-R11)
# =====================================================================


class R08_Roundabout(BaseRule):
    """Преобразование перекрёстка в круговой.

    Trigger: столкновения в населённом пункте, n_points ≥ 50.
    Effect: CMF Clearinghouse #226 — −78 % смертельных + тяжёлых
    на перекрёстке.
    """

    rule_id: ClassVar[str] = "R08_roundabout"

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.top_em_type == EM_TYPE_COLLISION and p.is_in_city and p.n_points >= 50

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=2,
            title="Круговое движение (roundabout) на перекрёстке",
            icon=":material/sync:",
            trigger_human=(f"Столкновения в населённом пункте, n = {p.n_points} (≥ 50)"),
            expected_effect=EffectEstimate(
                metric="fatal_severe_intersection",
                point_estimate=-0.78,
                ci_low=-0.65,
                ci_high=-0.90,
                note="CMF Clearinghouse #226: signalized → roundabout",
            ),
            evidence_basis=[
                Citation(
                    label="CMF Clearinghouse #226",
                    url="https://cmfclearinghouse.fhwa.dot.gov/detail.php?facid=10086",
                    quoted_effect="−78% fatal+severe; устранение T-bone collisions",
                ),
                Citation(
                    label="FHWA PSC Roundabouts",
                    url="https://highways.dot.gov/safety/proven-safety-countermeasures/roundabouts",
                    quoted_effect="Proven Safety Countermeasure",
                ),
            ],
            confidence=0.80,
            implementation_cost="high",
            target_severity=["dead", "severe", "severe_multiple"],
            target_em_types=[EM_TYPE_COLLISION],
        )


class R09_CableMedianBarrier(BaseRule):
    """Тросовое разделительное ограждение.

    Trigger: столкновения на трассе с высокой долей смертельных.

    Изначально требовался `radius_meters >= 2000` как proxy для
    «длинного прямого участка» — но (а) для DBSCAN это spread кластера,
    не road geometry, (б) для dynamic_radius — пользовательский ввод.
    Заменено на `pct_dead >= 0.03` (3 %): cross-median fatal crashes
    сами по себе сигнал того, что нужна median barrier, независимо
    от длины участка. Реальная road geometry будет в7 через
    OSM snap-to-road.

    Effect: FHWA-HRT-17-070 + MnDOT 2023 — −95 % смертельных
    cross-median.
    """

    rule_id: ClassVar[str] = "R09_cable_median_barrier"

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.top_em_type == EM_TYPE_COLLISION and p.is_highway and p.pct_dead >= 0.03

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        priority = 1 if p.pct_dead >= 0.05 else 2
        return Recommendation(
            rule_id=self.rule_id,
            priority=priority,
            title="Тросовое разделительное ограждение",
            icon=":material/fence:",
            trigger_human=(
                f"Столкновения на трассе, radius ≈ {p.radius_meters:.0f} м "
                f"(длинный прямой), pct_dead = {p.pct_dead:.1%}"
            ),
            expected_effect=EffectEstimate(
                metric="cross_median_fatal",
                point_estimate=-0.95,
                ci_low=-0.85,
                ci_high=-0.99,
                note="MnDOT 2023: 95% effective for cross-median fatal crashes",
            ),
            evidence_basis=[
                Citation(
                    label="FHWA-HRT-17-070",
                    url="https://www.fhwa.dot.gov/publications/research/safety/17070/001.cfm",
                    quoted_effect="Cable median barriers: −95% cross-median fatal",
                )
            ],
            confidence=0.85,
            implementation_cost="high",
            target_severity=["dead", "severe_multiple"],
            target_em_types=[EM_TYPE_COLLISION],
        )


class R10_HighFrictionSurface(BaseRule):
    """Высокофрикционное покрытие (HFST).

    Trigger: опрокидывания/съезды/столкновения на мокром или
    обледенелом покрытии.
    Effect: FHWA HFST EDC-2 — −83 % мокрых, −57 % всех ДТП.
    """

    rule_id: ClassVar[str] = "R10_high_friction_surface"
    SLIPPERY_STATES = {STATE_WET, STATE_ICE, STATE_SNOW_PACK, STATE_SNOWY}

    def applies_to(self, p: HotspotProfile) -> bool:
        if p.top_em_type not in {EM_TYPE_ROLLOVER, EM_TYPE_ROADSIDE, EM_TYPE_COLLISION}:
            return False
        return p.dominant_state in self.SLIPPERY_STATES

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        priority = 1 if p.top_em_type == EM_TYPE_ROLLOVER else 2
        return Recommendation(
            rule_id=self.rule_id,
            priority=priority,
            title="Высокофрикционное покрытие (HFST)",
            icon=":material/blur_on:",
            trigger_human=(f"{p.top_em_type} при покрытии «{p.dominant_state}»"),
            expected_effect=EffectEstimate(
                metric="wet_crashes",
                point_estimate=-0.83,
                ci_low=-0.60,
                ci_high=-0.93,
                note="FHWA HFST EDC-2: −83% мокрых ДТП, −57% всех в зоне",
            ),
            evidence_basis=[
                Citation(
                    label="FHWA HFST EDC-2",
                    url="https://www.fhwa.dot.gov/innovation/everydaycounts/edc-2/hfst.cfm",
                    quoted_effect="Every Day Counts initiative: −83% wet, −57% all",
                )
            ],
            confidence=0.80,
            implementation_cost="medium",
            target_severity=["severe", "dead"],
            target_em_types=[EM_TYPE_ROLLOVER, EM_TYPE_ROADSIDE, EM_TYPE_COLLISION],
        )


class R11_Guardrail(BaseRule):
    """Защитное барьерное ограждение обочины.

    Trigger: съезды на трассе, pct_dead ≥ 4 %.
    Effect: FHWA Roadside Design — −50 % смертельных съездов.
    """

    rule_id: ClassVar[str] = "R11_guardrail"

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.top_em_type == EM_TYPE_ROADSIDE and p.is_highway and p.pct_dead >= 0.04

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=1,
            title="Защитное барьерное ограждение обочины",
            icon=":material/safety_divider:",
            trigger_human=(f"Съезды на трассе, pct_dead = {p.pct_dead:.1%} ≥ 4 %"),
            expected_effect=EffectEstimate(
                metric="fatal_run_off_road",
                point_estimate=-0.50,
                ci_low=-0.30,
                ci_high=-0.65,
                note="FHWA Proven Safety Countermeasure: Roadside Design",
            ),
            evidence_basis=[
                Citation(
                    label="FHWA Roadside Design Guide",
                    url="https://highways.dot.gov/safety/proven-safety-countermeasures",
                    quoted_effect="Guardrail: −50% fatal run-off-road",
                ),
                Citation(
                    label="AASHTO RDG 2011",
                    url="https://store.transportation.org/Item/CollectionDetail?ID=105",
                    quoted_effect="AASHTO Roadside Design Guide",
                ),
            ],
            confidence=0.75,
            implementation_cost="medium",
            target_severity=["dead", "severe"],
            target_em_types=[EM_TYPE_ROADSIDE],
        )


# =====================================================================
# Группа D — Светофорное регулирование (R12-R13)
# =====================================================================


class R12_LeadingPedestrianInterval(BaseRule):
    """Leading Pedestrian Interval (LPI).

    Trigger: пешеходные ДТП в населённом пункте, n_points ≥ 100
    (proxy для оживлённого signalized intersection — данных о наличии
    светофора в БД нет). Порог 100 (вместо 30) убирает мелкие
    некрупные участки, на которых LPI не применим (нет светофора).

    Confidence снижена с 0.85 до 0.65 — proxy-trigger без прямого
    подтверждения signal-data из OSM. После7 (snap-to-road)
    можно вернуть 0.85, проверяя реальное наличие светофоров.

    Effect: FHWA PSC LPI — CMF 0.413 (−58.7 % пешеходных).
    """

    rule_id: ClassVar[str] = "R12_lpi"

    def applies_to(self, p: HotspotProfile) -> bool:
        # n_points ≥ 100 как proxy для оживлённого signalized perekrjostka
        return p.top_em_type == EM_TYPE_PED and p.is_in_city and p.n_points >= 100

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=1,
            title="Leading Pedestrian Interval (LPI) — приоритетный зелёный для пешехода",
            icon=":material/traffic:",
            trigger_human=(
                f"Пешеходные ДТП в населённом пункте, n = {p.n_points} "
                f"(proxy для signalized intersection — без OSM signal-data)"
            ),
            expected_effect=EffectEstimate(
                metric="pedestrian_crashes",
                point_estimate=-0.587,
                ci_low=-0.34,
                ci_high=-0.78,
                note=(
                    "FHWA PSC LPI: CMF 0.413; Seattle 2009-2018: −48%. "
                    "Применимо ТОЛЬКО при наличии светофорного регулирования "
                    "— подтверди на месте перед внедрением."
                ),
            ),
            evidence_basis=[
                Citation(
                    label="FHWA PSC LPI",
                    url="https://highways.dot.gov/safety/proven-safety-countermeasures/leading-pedestrian-interval",
                    quoted_effect="−58.7% pedestrian crashes; CMF 0.413",
                )
            ],
            # 0.65 (вместо 0.85) — снижено из-за proxy-trigger без OSM signal-data
            confidence=0.65,
            implementation_cost="low",
            target_severity=["dead", "severe"],
            target_em_types=[EM_TYPE_PED],
        )


class R13_SignalCoordination(BaseRule):
    """Координация светофоров (green wave) на коридоре.

    Trigger: столкновения в населённом пункте, n_points ≥ 80
    (большой коридор с светофорами).
    Effect: TRB NCHRP 500 — −25 % T-intersection collisions.
    """

    rule_id: ClassVar[str] = "R13_signal_coordination"

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.top_em_type == EM_TYPE_COLLISION and p.is_in_city and p.n_points >= 80

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=2,
            title="Координация светофоров (green wave)",
            icon=":material/route:",
            trigger_human=(f"Столкновения в крупном городском коридоре, n = {p.n_points} (≥ 80)"),
            expected_effect=EffectEstimate(
                metric="t_intersection_collisions",
                point_estimate=-0.25,
                ci_low=-0.10,
                ci_high=-0.40,
                note="TRB NCHRP 500 — signal coordination family",
            ),
            evidence_basis=[
                Citation(
                    label="TRB NCHRP 500",
                    url="https://www.trb.org/Publications/PubsNCHRPProjectReportsSafety.aspx",
                    quoted_effect="Signal coordination: −25% T-intersection collisions",
                )
            ],
            confidence=0.65,
            implementation_cost="medium",
            target_severity=["severe", "light"],
            target_em_types=[EM_TYPE_COLLISION],
        )


# =====================================================================
# Группа E — Уязвимые участники (R14-R15)
# =====================================================================


class R14_ProtectedBikeLane(BaseRule):
    """Защищённая велодорожка (PBL).

    Trigger: ДТП с велосипедистами в населённом пункте.
    Effect: Marshall & Ferenchak 2019 (12-city US) — −50 % cyclist
    injuries.
    """

    rule_id: ClassVar[str] = "R14_protected_bike_lane"

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.top_em_type == EM_TYPE_BIKE and p.is_in_city

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=1,
            title="Защищённая велодорожка (PBL)",
            icon=":material/directions_bike:",
            trigger_human=(
                f"Доминирующий тип ДТП — наезд на велосипедиста, "
                f"в населённом пункте, n = {p.n_points}"
            ),
            expected_effect=EffectEstimate(
                metric="cyclist_injuries",
                point_estimate=-0.50,
                ci_low=-0.30,
                ci_high=-0.70,
                note="Marshall & Ferenchak 2019: 12-city US study",
            ),
            evidence_basis=[
                Citation(
                    label="Marshall & Ferenchak 2019",
                    url="https://doi.org/10.1016/j.jth.2019.100539",
                    quoted_effect="−50% cyclist injuries в городах с PBL",
                )
            ],
            confidence=0.75,
            implementation_cost="high",
            target_severity=["dead", "severe"],
            target_em_types=[EM_TYPE_BIKE],
        )


class R15_RRFB(BaseRule):
    """RRFB — мигающий пешеходный знак.

    Trigger: пешеходные ДТП в городе с pct_dead ≥ 2 % (несигнальный
    переход — proxy: n_points < 30, иначе LPI лучше).
    Effect: FHWA PSC RRFB — −47 % пешеходных, yielding 0 % → 90 %.
    """

    rule_id: ClassVar[str] = "R15_rrfb"

    def applies_to(self, p: HotspotProfile) -> bool:
        if p.top_em_type != EM_TYPE_PED or not p.is_in_city:
            return False
        if p.pct_dead < 0.02:
            return False
        # Несигнальный переход: малый/средний поток
        return p.n_points < 50

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=1,
            title="RRFB — мигающий пешеходный знак",
            icon=":material/flashlight_on:",
            trigger_human=(
                f"Пешеходные ДТП в населённом пункте на нерегулируемом "
                f"переходе, pct_dead = {p.pct_dead:.1%}"
            ),
            expected_effect=EffectEstimate(
                metric="pedestrian_crashes",
                point_estimate=-0.47,
                ci_low=-0.28,
                ci_high=-0.65,
                note="FHWA PSC RRFB: yielding 0% → 90%, −47% pedestrian",
            ),
            evidence_basis=[
                Citation(
                    label="FHWA PSC RRFB",
                    url="https://highways.dot.gov/safety/proven-safety-countermeasures/rectangular-rapid-flashing-beacons-rrfb",
                    quoted_effect="−47% ped crashes, yielding 0%→90%",
                )
            ],
            confidence=0.80,
            implementation_cost="low",
            target_severity=["dead", "severe"],
            target_em_types=[EM_TYPE_PED],
        )


# =====================================================================
# Группа F — Поведение и enforcement (R16-R17)
# =====================================================================


class R16_SeatbeltEnforcement(BaseRule):
    """Усиленный контроль ремней безопасности.

    Trigger: pct_dead ≥ 4 % И тип ДТП связан с ТС (не пешеход).
    Effect: NHTSA + BMC Public Health 2018 — +14 п.п. использование,
    −8 п.п. occupant fatality.
    """

    rule_id: ClassVar[str] = "R16_seatbelt_enforcement"
    NON_VEHICLE_TYPES = {EM_TYPE_PED, EM_TYPE_BIKE}

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.pct_dead >= 0.04 and (p.top_em_type not in self.NON_VEHICLE_TYPES)

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=2,
            title="Усиленный контроль ремней (high-visibility enforcement)",
            icon=":material/airline_seat_recline_normal:",
            trigger_human=(f"pct_dead = {p.pct_dead:.1%} ≥ 4 %, тип ДТП связан с ТС"),
            expected_effect=EffectEstimate(
                metric="occupant_fatalities",
                point_estimate=-0.08,
                ci_low=-0.04,
                ci_high=-0.14,
                note="Переход secondary → primary enforcement: +14 п.п. usage",
            ),
            evidence_basis=[
                Citation(
                    label="NHTSA Click-It-or-Ticket",
                    url="https://www.nhtsa.gov/risky-driving/seat-belts",
                    quoted_effect="+14% usage, −8% occupant fatality rate",
                ),
                Citation(
                    label="BMC Public Health 2018",
                    url="https://doi.org/10.1186/s12889-018-5466-x",
                    quoted_effect="Systematic review of seatbelt enforcement",
                ),
            ],
            confidence=0.80,
            implementation_cost="low",
            target_severity=["dead", "severe"],
            target_em_types=[],
        )


class R17_RandomBreathTesting(BaseRule):
    """Random breath testing (RBT) на стационарных постах.

    Trigger: pct_dead ≥ 5 %.
    Effect: Community Guide RBT review + Australian RBT 30+ years —
    −10..−20 % alcohol-related crashes.
    """

    rule_id: ClassVar[str] = "R17_rbt"

    def applies_to(self, p: HotspotProfile) -> bool:
        return p.pct_dead >= 0.05

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        return Recommendation(
            rule_id=self.rule_id,
            priority=2,
            title="Стационарные посты random breath testing (пятница-суббота 19-03)",
            icon=":material/local_police:",
            trigger_human=(f"Высокая летальность: pct_dead = {p.pct_dead:.1%} ≥ 5 %"),
            expected_effect=EffectEstimate(
                metric="alcohol_related_crashes",
                point_estimate=-0.15,
                ci_low=-0.10,
                ci_high=-0.20,
                note="Community Guide + Australian RBT 30+ years experience",
            ),
            evidence_basis=[
                Citation(
                    label="Community Preventive Services Task Force",
                    url="https://www.thecommunityguide.org/findings/motor-vehicle-injury-alcohol-impaired-driving-publicized-sobriety-checkpoint-programs",
                    quoted_effect="Sobriety checkpoints: −20% alcohol-related crashes",
                )
            ],
            confidence=0.75,
            implementation_cost="medium",
            target_severity=["dead", "severe"],
            target_em_types=[],
        )


# =====================================================================
# Группа G — Зимнее содержание (R18)
# =====================================================================


class R18_AntiIcingBrine(BaseRule):
    """Превентивное anti-icing brine на склонах и кривых.

    Trigger: dominant_state — гололёд/наст/снег + has_winter_spike.
    Effect: Marquette study + American Highway Users Alliance —
    −78 .. −87 % зимних ДТП.
    """

    rule_id: ClassVar[str] = "R18_anti_icing_brine"

    def applies_to(self, p: HotspotProfile) -> bool:
        if p.dominant_state not in WINTER_SLIPPERY_STATES:
            return False
        return p.has_winter_spike or p.top_em_type in {EM_TYPE_ROLLOVER, EM_TYPE_ROADSIDE}

    def evaluate(self, p: HotspotProfile) -> Recommendation:
        # Зимой priority=1, иначе 2
        priority = 1 if p.has_winter_spike else 2
        return Recommendation(
            rule_id=self.rule_id,
            priority=priority,
            title="Превентивное anti-icing brine (зимнее содержание)",
            icon=":material/ac_unit:",
            trigger_human=(f"Покрытие «{p.dominant_state}», winter_spike = {p.has_winter_spike}"),
            expected_effect=EffectEstimate(
                metric="winter_crashes",
                point_estimate=-0.78,
                ci_low=-0.65,
                ci_high=-0.87,
                note="Marquette study; American Highway Users Alliance",
            ),
            evidence_basis=[
                Citation(
                    label="Marquette / Hwy Users Alliance",
                    url="https://www.highways.org/2009/09/02/road-salt-cuts-crashes-by-up-to-87/",
                    quoted_effect="−78% to −87% winter crashes на обработанных участках",
                )
            ],
            confidence=0.70,
            implementation_cost="medium",
            target_severity=["severe", "dead"],
            target_em_types=[EM_TYPE_ROLLOVER, EM_TYPE_ROADSIDE],
        )


# =====================================================================
# RuleEngine
# =====================================================================


class RuleEngine:
    """Контейнер всех 18 правил + sorting/scoring.

    Пример::

        engine = RuleEngine()
        recs = engine.recommend(profile, top_k=3)
        for r in recs:
            ...
    """

    def __init__(self) -> None:
        self.rules: list[BaseRule] = [
            R01_SpeedLimit30(),
            R02_SpeedCameras(),
            R03_RumbleStrips(),
            R04_SpeedHumps(),
            R05_PedestrianLighting(),
            R06_HighwayLighting(),
            R07_Delineators(),
            R08_Roundabout(),
            R09_CableMedianBarrier(),
            R10_HighFrictionSurface(),
            R11_Guardrail(),
            R12_LeadingPedestrianInterval(),
            R13_SignalCoordination(),
            R14_ProtectedBikeLane(),
            R15_RRFB(),
            R16_SeatbeltEnforcement(),
            R17_RandomBreathTesting(),
            R18_AntiIcingBrine(),
        ]

    def recommend(self, profile: HotspotProfile, top_k: int = 5) -> list[Recommendation]:
        """Возвращает топ-K рекомендаций для профиля, отсортированных по score."""
        candidates: list[Recommendation] = []
        for rule in self.rules:
            try:
                if rule.applies_to(profile):
                    rec = rule.evaluate(profile)
                    if rec is not None:
                        candidates.append(rec)
            except Exception as e:  # noqa: BLE001
                # Защита от ошибки в одном правиле — не падаем на остальных.
                # Логируем, но не пробрасываем — UI важнее, чем 100 % покрытие.
                import logging

                logging.getLogger(__name__).warning("Rule %s raised: %s", rule.rule_id, e)

        candidates.sort(key=lambda r: -score_recommendation(r))
        return candidates[:top_k]


__all__ = [
    "BaseRule",
    "Citation",
    "EffectEstimate",
    "PRIORITY_WEIGHTS",
    "Recommendation",
    "RuleEngine",
    "R01_SpeedLimit30",
    "R02_SpeedCameras",
    "R03_RumbleStrips",
    "R04_SpeedHumps",
    "R05_PedestrianLighting",
    "R06_HighwayLighting",
    "R07_Delineators",
    "R08_Roundabout",
    "R09_CableMedianBarrier",
    "R10_HighFrictionSurface",
    "R11_Guardrail",
    "R12_LeadingPedestrianInterval",
    "R13_SignalCoordination",
    "R14_ProtectedBikeLane",
    "R15_RRFB",
    "R16_SeatbeltEnforcement",
    "R17_RandomBreathTesting",
    "R18_AntiIcingBrine",
    "score_recommendation",
]
