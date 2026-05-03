"""Unit-тесты scoring-matcher'а из ``src.analysis.telegram_db_matcher``.

Покрывает все pure-функции:
* ``normalize_natasha_location`` — справочник падежных форм
* ``extract_normalized_locations`` — батч-обёртка
* ``extract_streets_from_text`` — regex по типам улиц
* ``detect_expected_em_type`` — keyword → em_type БД
* ``detect_severity_from_text`` — death/severe/light маркеры
* ``count_brand_overlap`` — пересечение марок поста и БД
* ``match_post`` — конечный scorer

Зависит только от ``re`` + ``datetime`` (БД не нужна).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.analysis.telegram_db_matcher import (
    count_brand_overlap,
    detect_expected_em_type,
    detect_severity_from_text,
    extract_normalized_locations,
    extract_streets_from_text,
    match_post,
    normalize_natasha_location,
)

# =====================================================================
# normalize_natasha_location
# =====================================================================


@pytest.mark.parametrize(
    "natasha,expected",
    [
        ("Владивосток", "г Владивосток"),
        ("владивосток", "г Владивосток"),
        ("ВЛАДИВОСТОКЕ", "г Владивосток"),
        ("Уссурийска", "г Уссурийск"),
        ("Артёме", "г Артем"),
        ("Большой Камень", "г Большой Камень"),
        ("", None),
        ("Москва", None),  # не из Приморья — нет в справочнике
        ("какой-то мусор", None),
    ],
)
def test_normalize_natasha_location(natasha: str, expected):
    assert normalize_natasha_location(natasha) == expected


def test_extract_normalized_locations_dedupe_sort():
    """Дубликаты схлопываются, результат отсортирован."""
    posts = [
        {"text": "Владивостоке"},
        {"text": "Владивосток"},  # дубль
        {"text": "Уссурийск"},
        {"text": "Москва"},  # отбрасывается
    ]
    result = extract_normalized_locations(posts)
    assert result == ["г Владивосток", "г Уссурийск"]


def test_extract_normalized_locations_empty():
    assert extract_normalized_locations([]) == []


# =====================================================================
# extract_streets_from_text
# =====================================================================


def test_extract_streets_basic_ulitsa():
    text = "ДТП произошло на улице Светланская во Владивостоке."
    streets = extract_streets_from_text(text)
    assert any("светланская" in s.lower() for s in streets)
    assert all(s.startswith("ул ") for s in streets)


def test_extract_streets_prospect():
    """«проспект Океанский» в именительном падеже даёт префикс «пр-кт»."""
    text = "На проспект Океанский выехал автомобиль."
    streets = extract_streets_from_text(text)
    assert any("океанский" in s.lower() for s in streets)
    assert any(s.startswith("пр-кт ") for s in streets)


def test_extract_streets_prospect_locative_fallback():
    """В падежной форме «проспекте» префикс не нормализуется (fallback на «ул»).

    Это известное ограничение matcher'а — лемматизация была бы дорогой.
    Тест фиксирует текущее поведение, чтобы не было тихого регресса.
    """
    text = "На проспекте Океанский водитель не справился с управлением."
    streets = extract_streets_from_text(text)
    assert any("океанский" in s.lower() for s in streets)


def test_extract_streets_abbreviated():
    text = "ДТП на ул. Шилкинская и пер. Тихий."
    streets = extract_streets_from_text(text)
    # Должны быть оба
    joined = " ".join(streets).lower()
    assert "шилкинская" in joined
    assert "тихий" in joined


def test_extract_streets_empty_input():
    assert extract_streets_from_text("") == []


def test_extract_streets_no_match():
    assert extract_streets_from_text("Просто текст без улиц.") == []


# =====================================================================
# detect_expected_em_type
# =====================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Произошло столкновение двух автомобилей.", "Столкновение"),
        ("Водитель совершил столкновение с грузовиком.", "Столкновение"),
        ("Произошёл наезд на пешехода в зоне перехода.", "Наезд на пешехода"),
        ("Водитель сбил пешехода на переходе.", "Наезд на пешехода"),
        ("Автомобиль опрокинулся в кювет.", "Опрокидывание"),
        ("Водитель съехал с дороги.", "Съезд с дороги"),
        ("Произошёл наезд на препятствие.", "Наезд на препятствие"),
        ("Велосипедист получил травмы.", "Наезд на велосипедиста"),
        ("", None),
        ("Просто текст без классификатора.", None),
    ],
)
def test_detect_expected_em_type(text: str, expected):
    assert detect_expected_em_type(text) == expected


def test_detect_expected_em_type_motorcyclist_returns_none():
    """Мотоциклист — намеренно None (нет однозначного em_type)."""
    text = "Мотоциклист пострадал в ДТП."
    assert detect_expected_em_type(text) is None


# =====================================================================
# detect_severity_from_text
# =====================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Водитель погиб на месте.", "dead"),
        ("Пешеход скончался в больнице.", "dead"),
        ("ДТП со смертельным исходом.", "dead"),
        ("Водитель госпитализирован в тяжёлом состоянии.", "severe"),
        ("Получил телесные повреждения.", "severe"),
        ("Доставлен в реанимацию.", "severe"),
        ("Получил травмы лёгкой степени.", "light"),
        ("Назначено амбулаторное лечение.", "light"),
        ("", None),
        ("Просто текст без severity-маркеров.", None),
    ],
)
def test_detect_severity_from_text(text: str, expected):
    assert detect_severity_from_text(text) == expected


def test_detect_severity_dead_overrides_severe():
    """При наличии маркеров и смерти, и тяжести — побеждает 'dead'."""
    text = "Пострадавший был госпитализирован, но скончался в больнице."
    assert detect_severity_from_text(text) == "dead"


# =====================================================================
# count_brand_overlap
# =====================================================================


@pytest.mark.parametrize(
    "post,db,expected",
    [
        (["TOYOTA", "NISSAN"], ["TOYOTA", "MAZDA"], 1),
        (["TOYOTA"], ["toyota"], 1),  # case-insensitive
        (["TOYOTA", "NISSAN", "HONDA"], ["TOYOTA", "NISSAN", "HONDA"], 3),
        (["TOYOTA"], ["BMW"], 0),
        ([], ["TOYOTA"], 0),
        (["TOYOTA"], [], 0),
        ([], [], 0),
    ],
)
def test_count_brand_overlap(post: list[str], db: list[str], expected: int):
    assert count_brand_overlap(post, db) == expected


# =====================================================================
# match_post — high-precision scoring
# =====================================================================


def _make_candidate(**overrides) -> dict:
    """Дефолтный кандидат — для удобства тестов."""
    base = {
        "id": 1,
        "datetime": datetime(2024, 6, 1, 12, 0),
        "np": "г Владивосток",
        "road_km": None,
        "street": "ул Светланская",
        "severity": "light",
        "em_type": "Столкновение",
        "vehicle_marks": ["TOYOTA"],
    }
    base.update(overrides)
    return base


def test_match_post_perfect_match_max_score():
    """Все факторы совпадают → score близок к максимуму."""
    cand = _make_candidate()
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="Водитель Toyota совершил столкновение на ул. Светланская.",
        post_locations=["г Владивосток"],
        post_kilometers=[],
        post_streets=["ул Светланская"],
        post_brands=["TOYOTA"],
        candidates=[cand],
    )
    assert len(matches) == 1
    # +30 np, +30 street, +20 em_type, +25 vehicle_mark = 105
    assert matches[0]["score"] >= 100
    bd = matches[0]["score_breakdown"]
    assert bd["np"] == 30
    assert bd["street"] == 30
    assert bd["em_type"] == 20
    assert bd["vehicle_marks"] == 25


def test_match_post_only_np_below_threshold():
    """Только np-совпадение (+30) → НЕ проходит дефолтный threshold=30 НЕ проходит, но впритык."""
    cand = _make_candidate(street=None, em_type=None, vehicle_marks=[])
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="Какое-то происшествие.",
        post_locations=["г Владивосток"],
        post_kilometers=[],
        post_streets=[],
        post_brands=[],
        candidates=[cand],
    )
    # threshold=30 (>= по умолчанию), score=30 → попадает
    assert len(matches) == 1
    assert matches[0]["score"] == 30


def test_match_post_filters_out_of_date_window():
    """Кандидат за пределами days_before — не в результате."""
    cand = _make_candidate(datetime=datetime(2024, 5, 1, 12, 0))  # за 31 день
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="Столкновение во Владивостоке.",
        post_locations=["г Владивосток"],
        post_kilometers=[],
        post_streets=[],
        post_brands=[],
        candidates=[cand],
        days_before=3,
    )
    assert matches == []


def test_match_post_severity_dead_match():
    """Маркер «погиб на месте» → severity=dead → +15."""
    cand = _make_candidate(severity="dead")
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="Водитель погиб на месте после столкновения во Владивостоке.",
        post_locations=["г Владивосток"],
        post_kilometers=[],
        post_streets=[],
        post_brands=[],
        candidates=[cand],
    )
    assert len(matches) == 1
    assert matches[0]["score_breakdown"].get("severity") == 15


def test_match_post_road_km_nearby():
    """km-совпадение в пределах ±2 → +10."""
    cand = _make_candidate(road_km=600.0, street=None, em_type=None, vehicle_marks=[])
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="ДТП на 599 км трассы.",
        post_locations=["г Владивосток"],
        post_kilometers=[599],
        post_streets=[],
        post_brands=[],
        candidates=[cand],
    )
    assert len(matches) == 1
    assert matches[0]["score_breakdown"].get("road_km") == 10


def test_match_post_road_km_too_far():
    """km > 2 от road_km кандидата → не плюсуется."""
    cand = _make_candidate(road_km=600.0, street=None, em_type=None, vehicle_marks=[])
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="ДТП на 100 км трассы.",
        post_locations=["г Владивосток"],
        post_kilometers=[100],
        post_streets=[],
        post_brands=[],
        candidates=[cand],
    )
    # +30 np, без km → 30, всё ещё проходит threshold=30
    assert len(matches) == 1
    assert "road_km" not in matches[0]["score_breakdown"]


def test_match_post_vehicle_marks_capped_at_50():
    """3 совпавших марки → 25*3=75, обрезается до 50."""
    cand = _make_candidate(
        street=None,
        em_type=None,
        vehicle_marks=["TOYOTA", "NISSAN", "HONDA"],
    )
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="Столкновение нескольких машин.",
        post_locations=["г Владивосток"],
        post_kilometers=[],
        post_streets=[],
        post_brands=["TOYOTA", "NISSAN", "HONDA"],
        candidates=[cand],
    )
    assert len(matches) == 1
    assert matches[0]["score_breakdown"]["vehicle_marks"] == 50


def test_match_post_threshold_filters_low_score():
    """Кастомный высокий threshold отбрасывает слабые матчи."""
    cand = _make_candidate(street=None, em_type=None, vehicle_marks=[])
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="Просто что-то во Владивостоке.",
        post_locations=["г Владивосток"],
        post_kilometers=[],
        post_streets=[],
        post_brands=[],
        candidates=[cand],
        score_threshold=60,
    )
    # Только +30 за np — ниже 60
    assert matches == []


def test_match_post_empty_candidates():
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="Текст",
        post_locations=[],
        post_kilometers=[],
        post_streets=[],
        post_brands=[],
        candidates=[],
    )
    assert matches == []


def test_match_post_sorted_by_score_desc():
    """Результат отсортирован по убыванию score."""
    cand_low = _make_candidate(id=1, street=None, em_type=None, vehicle_marks=[])  # +30 np
    cand_high = _make_candidate(id=2)  # +30+30+20+25 = 105
    matches = match_post(
        post_date=datetime(2024, 6, 1).date(),
        post_text="Столкновение Toyota на ул. Светланская.",
        post_locations=["г Владивосток"],
        post_kilometers=[],
        post_streets=["ул Светланская"],
        post_brands=["TOYOTA"],
        candidates=[cand_low, cand_high],
    )
    assert len(matches) == 2
    assert matches[0]["score"] > matches[1]["score"]
    assert matches[0]["id"] == 2  # high первым
