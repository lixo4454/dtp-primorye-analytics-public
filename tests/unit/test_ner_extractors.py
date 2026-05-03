"""Unit-тесты regex-экстракторов из ``src.nlp.dtp_ner``.

Покрывает только regex-функции, которые НЕ требуют Natasha (она heavy
и грузится в integration-тестах через ``extract_all`` отдельно):
* ``extract_ages`` — три формата возраста
* ``extract_times`` — двоеточие vs точка + защита от КоАП
* ``extract_driving_experience``
* ``extract_kilometers``

Зависят только от ``re`` + stdlib.
"""

from __future__ import annotations

import pytest

from src.nlp.dtp_ner import (
    extract_ages,
    extract_driving_experience,
    extract_kilometers,
    extract_times,
    extract_vehicles,
)

# =====================================================================
# extract_ages — формат "63-летний водитель"
# =====================================================================


def test_extract_ages_hyphen_format():
    text = "63-летний водитель не справился с управлением."
    ages = extract_ages(text)
    assert len(ages) == 1
    assert ages[0]["age"] == 63
    assert ages[0]["context"] == "водитель"


def test_extract_ages_hyphen_female():
    text = "23-летняя женщина переходила дорогу в неустановленном месте."
    ages = extract_ages(text)
    assert len(ages) == 1
    assert ages[0]["age"] == 23
    assert ages[0]["context"] == "женщина"


# =====================================================================
# extract_ages — формат "47 лет водитель"
# =====================================================================


def test_extract_ages_num_noun_format():
    text = "47 лет водитель ехал из Артёма во Владивосток."
    ages = extract_ages(text)
    assert len(ages) >= 1
    assert any(a["age"] == 47 for a in ages)


def test_extract_ages_num_noun_pedestrian():
    text = "44 года женщина переходила дорогу."
    ages = extract_ages(text)
    assert any(a["age"] == 44 for a in ages)


# =====================================================================
# extract_ages — формат "в возрасте N лет"
# =====================================================================


def test_extract_ages_vozraste_format():
    text = "Пострадавший в возрасте 47 лет был доставлен в больницу."
    ages = extract_ages(text)
    assert any(a["age"] == 47 for a in ages)


# =====================================================================
# extract_ages — фильтр абсурдных значений и пустого ввода
# =====================================================================


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Просто текст без возрастов.",
        "150-летний дед",  # > 110 — фильтруется
        "0-летний малыш",  # < 1 — фильтруется
    ],
)
def test_extract_ages_no_match(text: str):
    """Возрасты вне [1, 110] и пустые тексты должны давать []."""
    if text == "":
        assert extract_ages(text) == []
    else:
        ages = extract_ages(text)
        # либо пусто, либо не содержит запрещённых значений
        for a in ages:
            assert 1 <= a["age"] <= 110


# =====================================================================
# extract_times — двоеточие
# =====================================================================


def test_extract_times_colon_format():
    text = "ДТП произошло в 23:20 на перекрёстке."
    times = extract_times(text)
    assert len(times) == 1
    assert times[0]["hour"] == 23
    assert times[0]["minute"] == 20


def test_extract_times_multiple_colon():
    text = "В 07:00 водитель выехал, а в 23:45 вернулся."
    times = extract_times(text)
    assert len(times) == 2
    assert {(t["hour"], t["minute"]) for t in times} == {(7, 0), (23, 45)}


# =====================================================================
# extract_times — точка с временным маркером
# =====================================================================


def test_extract_times_dot_with_marker():
    """«в 12.30» — точка распознаётся при наличии маркера «в»."""
    text = "ДТП произошло в 12.30 во Владивостоке."
    times = extract_times(text)
    assert any((t["hour"], t["minute"]) == (12, 30) for t in times)


# =====================================================================
# extract_times — защита от ложных срабатываний на КоАП
# =====================================================================


def test_extract_times_dot_blocks_koap():
    """«ст. 12.5 КоАП» НЕ должно интерпретироваться как 12:05."""
    text = "Привлечён к ответственности по ст. 12.5 КоАП РФ."
    times = extract_times(text)
    # 12.5 не должно быть распознано как время
    assert not any((t["hour"], t["minute"]) == (12, 5) for t in times)


# =====================================================================
# extract_times — невалидные часы/минуты
# =====================================================================


def test_extract_times_invalid_hour_filtered():
    """Час > 24 или минута > 59 фильтруются."""
    text = "Время 25:00 и 12:75 не валидны, а 14:30 — да."
    times = extract_times(text)
    valid = {(t["hour"], t["minute"]) for t in times}
    assert (25, 0) not in valid
    assert (12, 75) not in valid
    assert (14, 30) in valid


def test_extract_times_empty_returns_empty():
    assert extract_times("") == []


# =====================================================================
# extract_driving_experience
# =====================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Стаж вождения 5 лет.", [5]),
        ("Водительский стаж составляет 28 лет.", [28]),
        ("Стаж управления — 12 лет.", [12]),
        ("Стаж 30 лет.", [30]),
        ("", []),
        ("Никакого стажа в тексте нет.", []),
    ],
)
def test_extract_driving_experience(text: str, expected: list[int]):
    assert extract_driving_experience(text) == expected


def test_extract_driving_experience_filters_huge_values():
    """Стаж > 80 лет — нереалистичен, фильтруется."""
    text = "Стаж 99 лет."
    assert extract_driving_experience(text) == []


# =====================================================================
# extract_kilometers
# =====================================================================


def test_extract_kilometers_basic():
    text = "ДТП произошло на 599 км трассы Уссури."
    assert 599 in extract_kilometers(text)


def test_extract_kilometers_multiple():
    text = "От 82 км до 120 км идёт ремонт."
    kms = extract_kilometers(text)
    assert 82 in kms
    assert 120 in kms


def test_extract_kilometers_empty():
    assert extract_kilometers("") == []


def test_extract_kilometers_no_match():
    assert extract_kilometers("Просто текст без километровки.") == []


# =====================================================================
# extract_vehicles — латинские бренды из справочника + русские алиасы
# =====================================================================


def test_extract_vehicles_latin_brand():
    text = "Водитель Toyota Camry допустил столкновение."
    vehicles = extract_vehicles(text)
    assert len(vehicles) >= 1
    brands = {v["brand"] for v in vehicles}
    assert "TOYOTA" in brands


def test_extract_vehicles_russian_alias():
    """Русский алиас «тойота» должен дать TOYOTA."""
    text = "Тойота столкнулась с грузовиком на трассе."
    vehicles = extract_vehicles(text)
    assert any(v["brand"] == "TOYOTA" for v in vehicles)


def test_extract_vehicles_multiple_brands():
    text = "Столкновение Toyota Camry и Nissan X-Trail."
    vehicles = extract_vehicles(text)
    brands = {v["brand"] for v in vehicles}
    assert "TOYOTA" in brands
    assert "NISSAN" in brands


def test_extract_vehicles_typo_normalized():
    """«Нисан» (опечатка) должна нормализоваться в NISSAN."""
    text = "Нисан столкнулся с автобусом."
    vehicles = extract_vehicles(text)
    # Все NISSAN_TYPO должны быть переименованы в NISSAN
    assert all(v["brand"] != "NISSAN_TYPO" for v in vehicles)
    assert any(v["brand"] == "NISSAN" for v in vehicles)


def test_extract_vehicles_empty_input():
    assert extract_vehicles("") == []


def test_extract_vehicles_no_brand():
    assert extract_vehicles("Просто текст без марок.") == []


def test_extract_vehicles_position_sorted():
    """Найденные марки отсортированы по позиции в тексте."""
    text = "Сначала Honda Accord, потом Toyota Camry."
    vehicles = extract_vehicles(text)
    if len(vehicles) >= 2:
        positions = [v["position"] for v in vehicles]
        assert positions == sorted(positions)
