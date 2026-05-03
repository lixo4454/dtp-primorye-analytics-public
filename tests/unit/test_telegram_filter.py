"""Unit-тесты ДТП-фильтра постов из Telegram-канала @prim_police.

Покрывает чистые функции из ``src.loaders.telegram_export_loader``:
* ``detect_dtp`` — трёхуровневый STRONG/WEAK/NEGATIVE-фильтр
* ``extract_text`` — слияние формата ``"text"`` (строка vs. сегменты)

БД и ML-зависимостей не требует.
"""

from __future__ import annotations

import pytest

from src.loaders.telegram_export_loader import detect_dtp, extract_text

# =====================================================================
# detect_dtp — позитивные кейсы (STRONG-маркер → ДТП)
# =====================================================================


@pytest.mark.parametrize(
    "text",
    [
        "Сегодня ночью на трассе А-188 произошло столкновение двух автомобилей.",
        "63-летний водитель Toyota Camry допустил столкновение с грузовиком.",
        "Пешеход получил травмы при наезде автомобиля.",
        "Мотоциклист скончался на месте происшествия.",
        "Велосипедист пострадал в ДТП на улице Светланской.",
        "Автомобиль опрокинулся в кювет, водитель госпитализирован.",
        "Водитель съехал с дороги и врезался в дерево.",
    ],
)
def test_detect_dtp_strong_keywords(text: str):
    is_dtp, kws = detect_dtp(text)
    assert is_dtp is True
    assert len(kws) >= 1


# =====================================================================
# detect_dtp — негативные кейсы (НЕ ДТП)
# =====================================================================


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Сегодня хорошая погода в Приморье.",
        "Сотрудники полиции провели плановое мероприятие.",
        "Граждане могут обращаться по телефону доверия.",
    ],
)
def test_detect_dtp_no_match(text: str):
    is_dtp, kws = detect_dtp(text)
    assert is_dtp is False
    assert kws == []


# =====================================================================
# detect_dtp — NEGATIVE-фразы как «вето» (даже при STRONG-маркере)
# =====================================================================


def test_detect_dtp_negative_overrides_strong_fraud():
    """«Столкнулась с мошенниками» → не ДТП, хоть и есть слово столкновени*."""
    text = (
        "63-летняя жительница Владивостока столкнулась с мошенниками: "
        "ей позвонил неизвестный, представившись сотрудником банка."
    )
    is_dtp, kws = detect_dtp(text)
    assert is_dtp is False
    assert kws == []


def test_detect_dtp_negative_overrides_strong_obituary():
    """День памяти погибших сотрудников — НЕ ДТП, хотя есть «погиб»."""
    text = (
        "Сегодня в УМВД прошёл день памяти погибших сотрудников при "
        "исполнении служебных обязанностей. Светлая память коллегам."
    )
    is_dtp, kws = detect_dtp(text)
    assert is_dtp is False


def test_detect_dtp_negative_overrides_congratulations():
    text = "Поздравляю вас с Днём сотрудника органов внутренних дел!"
    is_dtp, kws = detect_dtp(text)
    assert is_dtp is False


# =====================================================================
# detect_dtp — возвращаемые keywords
# =====================================================================


def test_detect_dtp_returns_matched_keywords():
    """Срабатывание возвращает реально найденные ключевые слова."""
    text = "Водитель получил травмы и был госпитализирован после столкновения."
    is_dtp, kws = detect_dtp(text)
    assert is_dtp is True
    # должны быть найдены хотя бы 'получил травмы' и 'госпитализирован'
    assert "получил травмы" in kws
    assert "госпитализирован" in kws


def test_detect_dtp_case_insensitive():
    """Фильтр работает регистронезависимо."""
    text = "ВОДИТЕЛЬ СОВЕРШИЛ СТОЛКНОВЕНИЕ С ГРУЗОВИКОМ"
    is_dtp, _ = detect_dtp(text)
    assert is_dtp is True


# =====================================================================
# extract_text — формат «строка» vs «массив сегментов»
# =====================================================================


def test_extract_text_plain_string():
    msg = {"id": 1, "text": "Простой текст поста"}
    assert extract_text(msg) == "Простой текст поста"


def test_extract_text_strips_whitespace():
    msg = {"text": "  trim me  "}
    assert extract_text(msg) == "trim me"


def test_extract_text_segments_list_dicts():
    """Telegram форматирование — массив сегментов (bold/link/plain)."""
    msg = {
        "text": [
            {"type": "plain", "text": "Жители "},
            {"type": "bold", "text": "Владивостока"},
            {"type": "plain", "text": " были информированы."},
        ]
    }
    assert extract_text(msg) == "Жители Владивостока были информированы."


def test_extract_text_segments_list_mixed():
    """Сегменты могут включать строки и dict-сегменты вперемешку."""
    msg = {
        "text": [
            "Открытая часть. ",
            {"type": "italic", "text": "Курсив. "},
            "Снова обычная.",
        ]
    }
    assert extract_text(msg) == "Открытая часть. Курсив. Снова обычная."


def test_extract_text_missing_field():
    assert extract_text({}) == ""


def test_extract_text_unknown_type():
    """Если text — не строка и не list, возвращаем пустую строку."""
    assert extract_text({"text": 123}) == ""


def test_extract_text_empty_segments():
    """Пустой массив сегментов → пустая строка."""
    assert extract_text({"text": []}) == ""
