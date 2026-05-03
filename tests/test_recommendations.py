"""Pytest-проверки rule engine из ``src.analysis.recommendations``.

По 2-3 case'а на каждое из 18 правил:
* positive trigger — профиль удовлетворяет триггеру → правило срабатывает
* negative trigger — профиль НЕ удовлетворяет → правило молчит
* sanity по полю expected_effect (CMF в правильном диапазоне)

Защита от тихого регресса при правке порогов в methodology.

Не запускают БД и FastAPI — только логика на dataclass'е.
"""

from __future__ import annotations

import pytest

from src.analysis.recommendations import (
    R15_RRFB,
    R01_SpeedLimit30,
    R02_SpeedCameras,
    R03_RumbleStrips,
    R04_SpeedHumps,
    R05_PedestrianLighting,
    R06_HighwayLighting,
    R07_Delineators,
    R08_Roundabout,
    R09_CableMedianBarrier,
    R10_HighFrictionSurface,
    R11_Guardrail,
    R12_LeadingPedestrianInterval,
    R13_SignalCoordination,
    R14_ProtectedBikeLane,
    R16_SeatbeltEnforcement,
    R17_RandomBreathTesting,
    R18_AntiIcingBrine,
    RuleEngine,
    score_recommendation,
)
from src.analysis.spot_profile import (
    EM_TYPE_BIKE,
    EM_TYPE_COLLISION,
    EM_TYPE_PED,
    EM_TYPE_ROADSIDE,
    EM_TYPE_ROLLOVER,
    LIGHT_DAY,
    LIGHT_NIGHT_NO_LIGHT,
    LIGHT_NIGHT_NOT_ON,
    STATE_DRY,
    STATE_ICE,
    STATE_WET,
    HotspotProfile,
)


def make_profile(**kwargs) -> HotspotProfile:
    """Удобный конструктор: дефолт — нейтральный профиль (60 ДТП, 1 % смертельных)."""
    base = dict(
        n_points=60,
        radius_meters=300.0,
        pct_dead=0.01,
        pct_severe_or_dead=0.10,
        top_em_type=EM_TYPE_COLLISION,
        top_em_types=[(EM_TYPE_COLLISION, 30)],
        top_np="г Владивосток",
        dominant_light_type=LIGHT_DAY,
        dominant_state=STATE_DRY,
        is_highway=False,
        is_in_city=True,
        has_night_dominant=False,
        has_winter_spike=False,
        source="test",
    )
    base.update(kwargs)
    return HotspotProfile(**base)


# =====================================================================
# R01 Speed limit 30
# =====================================================================


class TestR01:
    rule = R01_SpeedLimit30()

    def test_positive(self):
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, pct_dead=0.05)
        assert self.rule.applies_to(p)
        rec = self.rule.evaluate(p)
        assert rec.priority == 1
        assert rec.expected_effect.point_estimate == pytest.approx(-0.40)

    def test_negative_collision(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, is_in_city=True, pct_dead=0.10)
        assert not self.rule.applies_to(p)

    def test_negative_low_pct_dead(self):
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, pct_dead=0.02)
        assert not self.rule.applies_to(p)


# =====================================================================
# R02 Speed cameras
# =====================================================================


class TestR02:
    rule = R02_SpeedCameras()

    def test_positive_highway(self):
        p = make_profile(
            is_highway=True, is_in_city=False, pct_severe_or_dead=0.30, top_np="-я трасса"
        )
        assert self.rule.applies_to(p)

    def test_positive_big_city(self):
        p = make_profile(top_np="г Владивосток", pct_severe_or_dead=0.25)
        assert self.rule.applies_to(p)

    def test_priority_high_when_pct_dead_5pct(self):
        p = make_profile(top_np="г Владивосток", pct_severe_or_dead=0.30, pct_dead=0.06)
        rec = self.rule.evaluate(p)
        assert rec.priority == 1

    def test_negative_low_severity(self):
        p = make_profile(top_np="г Владивосток", pct_severe_or_dead=0.10)
        assert not self.rule.applies_to(p)

    def test_negative_small_town(self):
        p = make_profile(top_np="с Кневичи", pct_severe_or_dead=0.30, is_highway=False)
        assert not self.rule.applies_to(p)


# =====================================================================
# R03 Rumble strips
# =====================================================================


class TestR03:
    """R03 временно дисейблено до интеграции snap-to-road OSM.

    radius_meters для DBSCAN-кластеров — это spread, не road geometry.
    Для dynamic_radius — пользовательский ввод. Без OSM road-graph
    triggers'у нет надёжного сигнала «длинный прямой участок».
    """

    rule = R03_RumbleStrips()

    def test_disabled_for_highway_collision(self):
        # Раньше срабатывало — теперь нет до интеграции OSM
        p = make_profile(
            top_em_type=EM_TYPE_COLLISION,
            is_highway=True,
            is_in_city=False,
            radius_meters=1500,
        )
        assert not self.rule.applies_to(p)

    def test_disabled_in_city(self):
        p = make_profile(
            top_em_type=EM_TYPE_COLLISION,
            is_highway=False,
            is_in_city=True,
        )
        assert not self.rule.applies_to(p)


# =====================================================================
# R04 Speed humps
# =====================================================================


class TestR04:
    rule = R04_SpeedHumps()

    def test_positive(self):
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, n_points=40)
        assert self.rule.applies_to(p)

    def test_negative_low_n(self):
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, n_points=20)
        assert not self.rule.applies_to(p)


# =====================================================================
# R05 Pedestrian lighting
# =====================================================================


class TestR05:
    rule = R05_PedestrianLighting()

    def test_positive(self):
        p = make_profile(
            top_em_type=EM_TYPE_PED,
            n_points=30,
            has_night_dominant=True,
            dominant_light_type=LIGHT_NIGHT_NO_LIGHT,
        )
        assert self.rule.applies_to(p)

    def test_positive_dominant_light_dark(self):
        p = make_profile(
            top_em_type=EM_TYPE_PED,
            n_points=30,
            has_night_dominant=False,
            dominant_light_type=LIGHT_NIGHT_NOT_ON,
        )
        assert self.rule.applies_to(p)

    def test_negative_daytime(self):
        p = make_profile(top_em_type=EM_TYPE_PED, n_points=30, dominant_light_type=LIGHT_DAY)
        assert not self.rule.applies_to(p)


# =====================================================================
# R06 Highway lighting
# =====================================================================


class TestR06:
    rule = R06_HighwayLighting()

    def test_positive(self):
        p = make_profile(
            top_em_type=EM_TYPE_ROLLOVER,
            is_highway=True,
            is_in_city=False,
            dominant_light_type=LIGHT_NIGHT_NO_LIGHT,
        )
        assert self.rule.applies_to(p)

    def test_negative_in_city(self):
        p = make_profile(
            top_em_type=EM_TYPE_ROLLOVER,
            is_highway=False,
            is_in_city=True,
            dominant_light_type=LIGHT_NIGHT_NO_LIGHT,
        )
        assert not self.rule.applies_to(p)


# =====================================================================
# R07 Delineators
# =====================================================================


class TestR07:
    rule = R07_Delineators()

    def test_positive(self):
        p = make_profile(
            top_em_type=EM_TYPE_COLLISION,
            is_highway=True,
            is_in_city=False,
            has_night_dominant=True,
        )
        assert self.rule.applies_to(p)

    def test_negative_in_city(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, is_highway=False)
        assert not self.rule.applies_to(p)


# =====================================================================
# R08 Roundabout
# =====================================================================


class TestR08:
    rule = R08_Roundabout()

    def test_positive(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, is_in_city=True, n_points=80)
        assert self.rule.applies_to(p)

    def test_negative_low_n(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, is_in_city=True, n_points=30)
        assert not self.rule.applies_to(p)


# =====================================================================
# R09 Cable median barrier
# =====================================================================


class TestR09:
    """R09: переписан с radius_meters на pct_dead >= 0.03 (fix)."""

    rule = R09_CableMedianBarrier()

    def test_positive_with_pct_dead(self):
        p = make_profile(
            top_em_type=EM_TYPE_COLLISION,
            is_highway=True,
            is_in_city=False,
            pct_dead=0.05,
        )
        assert self.rule.applies_to(p)

    def test_priority_one_high_pct_dead(self):
        p = make_profile(
            top_em_type=EM_TYPE_COLLISION,
            is_highway=True,
            is_in_city=False,
            pct_dead=0.08,
        )
        rec = self.rule.evaluate(p)
        assert rec.priority == 1

    def test_negative_low_pct_dead(self):
        p = make_profile(
            top_em_type=EM_TYPE_COLLISION,
            is_highway=True,
            is_in_city=False,
            pct_dead=0.01,
        )
        assert not self.rule.applies_to(p)

    def test_negative_in_city(self):
        p = make_profile(
            top_em_type=EM_TYPE_COLLISION,
            is_highway=False,
            is_in_city=True,
            pct_dead=0.05,
        )
        assert not self.rule.applies_to(p)


# =====================================================================
# R10 HFST
# =====================================================================


class TestR10:
    rule = R10_HighFrictionSurface()

    def test_positive_wet_rollover(self):
        p = make_profile(top_em_type=EM_TYPE_ROLLOVER, dominant_state=STATE_WET)
        assert self.rule.applies_to(p)
        rec = self.rule.evaluate(p)
        assert rec.priority == 1  # rollover повышает priority

    def test_positive_ice(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, dominant_state=STATE_ICE)
        assert self.rule.applies_to(p)

    def test_negative_dry(self):
        p = make_profile(top_em_type=EM_TYPE_ROLLOVER, dominant_state=STATE_DRY)
        assert not self.rule.applies_to(p)


# =====================================================================
# R11 Guardrail
# =====================================================================


class TestR11:
    rule = R11_Guardrail()

    def test_positive(self):
        p = make_profile(
            top_em_type=EM_TYPE_ROADSIDE,
            is_highway=True,
            is_in_city=False,
            pct_dead=0.06,
        )
        assert self.rule.applies_to(p)

    def test_negative_low_pct_dead(self):
        p = make_profile(top_em_type=EM_TYPE_ROADSIDE, is_highway=True, pct_dead=0.02)
        assert not self.rule.applies_to(p)

    def test_negative_in_city(self):
        p = make_profile(top_em_type=EM_TYPE_ROADSIDE, is_highway=False, pct_dead=0.10)
        assert not self.rule.applies_to(p)


# =====================================================================
# R12 LPI
# =====================================================================


class TestR12:
    rule = R12_LeadingPedestrianInterval()

    def test_positive(self):
        # Порог поднят до n_points >= 100 — proxy для оживлённого signal.intersection
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, n_points=120)
        assert self.rule.applies_to(p)

    def test_negative_below_100(self):
        # 60 ДТП — раньше срабатывало, теперь нет (слабый сигнал signal-intersection)
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, n_points=60)
        assert not self.rule.applies_to(p)

    def test_confidence_lowered(self):
        # 0.65 (вместо 0.85) — proxy-trigger без OSM signal-data
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, n_points=120)
        rec = self.rule.evaluate(p)
        assert rec.confidence == 0.65


# =====================================================================
# R13 Signal coordination
# =====================================================================


class TestR13:
    rule = R13_SignalCoordination()

    def test_positive(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, is_in_city=True, n_points=120)
        assert self.rule.applies_to(p)

    def test_negative_low_n(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, is_in_city=True, n_points=50)
        assert not self.rule.applies_to(p)


# =====================================================================
# R14 PBL
# =====================================================================


class TestR14:
    rule = R14_ProtectedBikeLane()

    def test_positive(self):
        p = make_profile(top_em_type=EM_TYPE_BIKE, is_in_city=True)
        assert self.rule.applies_to(p)

    def test_negative_other_em(self):
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True)
        assert not self.rule.applies_to(p)


# =====================================================================
# R15 RRFB
# =====================================================================


class TestR15:
    rule = R15_RRFB()

    def test_positive(self):
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, pct_dead=0.04, n_points=20)
        assert self.rule.applies_to(p)

    def test_negative_dense_signalized_better(self):
        # n_points >= 50 — лучше LPI (R12), не RRFB
        p = make_profile(top_em_type=EM_TYPE_PED, is_in_city=True, pct_dead=0.04, n_points=80)
        assert not self.rule.applies_to(p)


# =====================================================================
# R16 Seatbelt enforcement
# =====================================================================


class TestR16:
    rule = R16_SeatbeltEnforcement()

    def test_positive(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, pct_dead=0.05)
        assert self.rule.applies_to(p)

    def test_negative_pedestrian(self):
        p = make_profile(top_em_type=EM_TYPE_PED, pct_dead=0.10)
        assert not self.rule.applies_to(p)

    def test_negative_low_pct_dead(self):
        p = make_profile(top_em_type=EM_TYPE_COLLISION, pct_dead=0.02)
        assert not self.rule.applies_to(p)


# =====================================================================
# R17 RBT
# =====================================================================


class TestR17:
    rule = R17_RandomBreathTesting()

    def test_positive(self):
        p = make_profile(pct_dead=0.06)
        assert self.rule.applies_to(p)

    def test_negative(self):
        p = make_profile(pct_dead=0.03)
        assert not self.rule.applies_to(p)


# =====================================================================
# R18 Anti-icing brine
# =====================================================================


class TestR18:
    rule = R18_AntiIcingBrine()

    def test_positive_winter_spike(self):
        p = make_profile(dominant_state=STATE_ICE, has_winter_spike=True)
        assert self.rule.applies_to(p)
        rec = self.rule.evaluate(p)
        assert rec.priority == 1

    def test_positive_rollover(self):
        p = make_profile(
            dominant_state=STATE_ICE,
            has_winter_spike=False,
            top_em_type=EM_TYPE_ROLLOVER,
        )
        assert self.rule.applies_to(p)
        rec = self.rule.evaluate(p)
        assert rec.priority == 2

    def test_negative_dry(self):
        p = make_profile(dominant_state=STATE_DRY, has_winter_spike=True)
        assert not self.rule.applies_to(p)


# =====================================================================
# RuleEngine integration
# =====================================================================


class TestRuleEngine:
    def test_engine_returns_top_k(self):
        engine = RuleEngine()
        p = make_profile(
            top_em_type=EM_TYPE_PED,
            is_in_city=True,
            n_points=120,
            pct_dead=0.06,
            pct_severe_or_dead=0.30,
            top_np="г Владивосток",
            has_night_dominant=True,
            dominant_light_type=LIGHT_NIGHT_NO_LIGHT,
        )
        recs = engine.recommend(p, top_k=3)
        assert len(recs) <= 3
        # Должны попасть R01 (speed limit), R05 (lighting), R12 (LPI), R02 (cameras)
        ids = {r.rule_id for r in recs}
        # Хотя бы одна из топовых пеш-рекомендаций
        assert ids & {"R01_speed_limit_30", "R05_pedestrian_lighting", "R12_lpi"}

    def test_engine_sorted_by_score(self):
        engine = RuleEngine()
        p = make_profile(
            top_em_type=EM_TYPE_PED,
            is_in_city=True,
            n_points=200,
            pct_dead=0.07,
            pct_severe_or_dead=0.35,
            top_np="г Владивосток",
            has_night_dominant=True,
            dominant_light_type=LIGHT_NIGHT_NO_LIGHT,
        )
        recs = engine.recommend(p, top_k=10)
        scores = [score_recommendation(r) for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_engine_no_recommendations_for_clean_profile(self):
        engine = RuleEngine()
        # Низкая летальность + дневной свет + сухое + малая выборка
        p = make_profile(
            top_em_type=EM_TYPE_COLLISION,
            is_in_city=True,
            n_points=10,
            pct_dead=0.001,
            pct_severe_or_dead=0.05,
            top_np="с Малое",
            dominant_light_type=LIGHT_DAY,
            dominant_state=STATE_DRY,
        )
        recs = engine.recommend(p)
        assert len(recs) == 0

    def test_all_18_rules_loaded(self):
        engine = RuleEngine()
        assert len(engine.rules) == 18
        ids = {r.rule_id for r in engine.rules}
        expected = {f"R{n:02d}_" for n in range(1, 19)}
        # У всех правил есть rule_id с префиксом RNN_
        for prefix in expected:
            assert any(rid.startswith(prefix) for rid in ids), prefix


# =====================================================================
# Sanity по Citation/EffectEstimate структурам
# =====================================================================


class TestRecommendationStructure:
    """Каждая Recommendation должна иметь ≥ 1 Citation, корректные CI и confidence."""

    # R03 исключено — временно дисейблено до интеграции snap-to-road OSM.
    # См. TestR03 + docstring R03_RumbleStrips.
    @pytest.mark.parametrize(
        "rule_class",
        [
            R01_SpeedLimit30,
            R02_SpeedCameras,
            R04_SpeedHumps,
            R05_PedestrianLighting,
            R06_HighwayLighting,
            R07_Delineators,
            R08_Roundabout,
            R09_CableMedianBarrier,
            R10_HighFrictionSurface,
            R11_Guardrail,
            R12_LeadingPedestrianInterval,
            R13_SignalCoordination,
            R14_ProtectedBikeLane,
            R15_RRFB,
            R16_SeatbeltEnforcement,
            R17_RandomBreathTesting,
            R18_AntiIcingBrine,
        ],
    )
    def test_rule_structure(self, rule_class):
        """Generic-проверка структуры рекомендации каждого правила."""
        rule = rule_class()
        # Profile, который точно вызывает trigger для конкретного правила
        triggers = {
            "R01_speed_limit_30": dict(top_em_type=EM_TYPE_PED, is_in_city=True, pct_dead=0.05),
            "R02_speed_cameras": dict(top_np="г Владивосток", pct_severe_or_dead=0.30),
            "R04_speed_humps": dict(top_em_type=EM_TYPE_PED, is_in_city=True, n_points=40),
            "R05_pedestrian_lighting": dict(
                top_em_type=EM_TYPE_PED,
                n_points=30,
                has_night_dominant=True,
                dominant_light_type=LIGHT_NIGHT_NO_LIGHT,
            ),
            "R06_highway_lighting": dict(
                top_em_type=EM_TYPE_ROLLOVER,
                is_highway=True,
                is_in_city=False,
                dominant_light_type=LIGHT_NIGHT_NO_LIGHT,
            ),
            "R07_delineators": dict(
                top_em_type=EM_TYPE_COLLISION,
                is_highway=True,
                is_in_city=False,
                has_night_dominant=True,
            ),
            "R08_roundabout": dict(top_em_type=EM_TYPE_COLLISION, is_in_city=True, n_points=80),
            "R09_cable_median_barrier": dict(
                top_em_type=EM_TYPE_COLLISION, is_highway=True, is_in_city=False, pct_dead=0.05
            ),
            "R10_high_friction_surface": dict(
                top_em_type=EM_TYPE_ROLLOVER, dominant_state=STATE_WET
            ),
            "R11_guardrail": dict(
                top_em_type=EM_TYPE_ROADSIDE, is_highway=True, is_in_city=False, pct_dead=0.06
            ),
            "R12_lpi": dict(top_em_type=EM_TYPE_PED, is_in_city=True, n_points=120),
            "R13_signal_coordination": dict(
                top_em_type=EM_TYPE_COLLISION, is_in_city=True, n_points=120
            ),
            "R14_protected_bike_lane": dict(top_em_type=EM_TYPE_BIKE, is_in_city=True),
            "R15_rrfb": dict(top_em_type=EM_TYPE_PED, is_in_city=True, pct_dead=0.04, n_points=20),
            "R16_seatbelt_enforcement": dict(top_em_type=EM_TYPE_COLLISION, pct_dead=0.05),
            "R17_rbt": dict(pct_dead=0.06),
            "R18_anti_icing_brine": dict(dominant_state=STATE_ICE, has_winter_spike=True),
        }
        kwargs = triggers[rule.rule_id]
        p = make_profile(**kwargs)
        assert rule.applies_to(p)
        rec = rule.evaluate(p)

        # Структурные проверки
        assert rec.rule_id == rule.rule_id
        assert rec.priority in (1, 2, 3)
        assert rec.title and len(rec.title) > 5
        assert rec.icon.startswith(":material/")
        assert rec.trigger_human and len(rec.trigger_human) > 5
        assert rec.implementation_cost in ("low", "medium", "high")

        # Эффект должен быть негативным (мера снижает риск)
        assert rec.expected_effect.point_estimate < 0
        # CI должны быть совместимы (low/high того же знака)
        assert rec.expected_effect.ci_low <= 0
        assert rec.expected_effect.ci_high <= 0

        # confidence в диапазоне
        assert 0.0 <= rec.confidence <= 1.0

        # Хотя бы 1 Citation с непустым URL
        assert len(rec.evidence_basis) >= 1
        for cit in rec.evidence_basis:
            assert cit.url.startswith("http")
            assert cit.label and len(cit.label) > 3
