# -*- coding: utf-8 -*-
"""
Production matching: связываем Telegram-NLP посты со структурной БД ДТП.

Использует scoring system:
- np (город):          +30
- street (улица):      +30
- em_type ↔ keywords:  +20
- severity (dead):     +15
- vehicle.mark:        +25 (умноженное на число совпавших марок)
- road_km ±2:          +10

Кандидаты с score >= 60 считаются high-precision matches.

Главные функции:
    extract_normalized_locations(post_locations) -> list[str]
    extract_streets_from_text(text) -> list[str]
    match_post(...) -> list[matched_db_records_with_score]
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

# ────────────────────────────────────────────────────────────────────
# Нормализация локаций Natasha → формат БД
# ────────────────────────────────────────────────────────────────────

LOCATION_NORMALIZATION: dict[str, str] = {
    # Городские округа
    "владивосток": "г Владивосток",
    "уссурийск": "г Уссурийск",
    "находка": "г Находка",
    "артем": "г Артем",
    "артём": "г Артем",
    "партизанск": "г Партизанск",
    "лесозаводск": "г Лесозаводск",
    "арсеньев": "г Арсеньев",
    "дальнегорск": "г Дальнегорск",
    "спасск-дальний": "г Спасск-Дальний",
    "большой камень": "г Большой Камень",
    "фокино": "г Фокино",
    "дальнереченск": "г Дальнереченск",
    "трудовое": "п Трудовое",
    "чугуевка": "с Чугуевка",
    "вольно-надеждинское": "с Вольно-Надеждинское",
    "раздольное": "п Раздольное",
    "кавалерово": "пгт Кавалерово",
    "тавричанка": "п Тавричанка",
    "пограничный": "пгт Пограничный",
    "славянка": "пгт Славянка",
    # Падежные формы
    "уссурийска": "г Уссурийск",
    "уссурийске": "г Уссурийск",
    "уссурийску": "г Уссурийск",
    "владивостока": "г Владивосток",
    "владивостоке": "г Владивосток",
    "владивостоку": "г Владивосток",
    "находки": "г Находка",
    "находке": "г Находка",
    "находку": "г Находка",
    "артеме": "г Артем",
    "артёме": "г Артем",
    "арсеньева": "г Арсеньев",
    "арсеньеве": "г Арсеньев",
    "партизанска": "г Партизанск",
    "партизанске": "г Партизанск",
    "лесозаводска": "г Лесозаводск",
    "лесозаводске": "г Лесозаводск",
    "дальнегорска": "г Дальнегорск",
    "дальнегорске": "г Дальнегорск",
    "спасска": "г Спасск-Дальний",
    "спасске": "г Спасск-Дальний",
    "большом камне": "г Большой Камень",
    "дальнереченска": "г Дальнереченск",
    "дальнереченске": "г Дальнереченск",
}


def normalize_natasha_location(natasha_text: str) -> Optional[str]:
    if not natasha_text:
        return None
    return LOCATION_NORMALIZATION.get(natasha_text.lower().strip())


def extract_normalized_locations(post_locations: list[dict]) -> list[str]:
    result: set[str] = set()
    for loc in post_locations:
        text_value = loc.get("text", "")
        normalized = normalize_natasha_location(text_value)
        if normalized:
            result.add(normalized)
    return sorted(result)


# ────────────────────────────────────────────────────────────────────
# Извлечение улиц из текста поста
# ────────────────────────────────────────────────────────────────────

# "улица Шилкинская", "ул. Светланская", "проспект 100-летия Владивостока"
RE_STREET = re.compile(
    r"(?:улиц(?:е|у|ы|а)|ул\.?|проспект(?:е|у|а)?|пр-кт|пр\.|"
    r"переулк(?:е|у|а)?|пер\.?|шоссе|бульвар(?:е|а)?)"
    r"\s+([А-ЯЁA-Z][А-Яа-яёЁA-Za-z0-9\-]+(?:\s+[А-ЯЁA-Z][А-Яа-яёЁA-Za-z0-9\-]+){0,2})",
    re.IGNORECASE,
)

# Стандартизация типа улицы → как в БД
STREET_PREFIX_NORMALIZATION = {
    "улица": "ул",
    "ул.": "ул",
    "ул": "ул",
    "проспект": "пр-кт",
    "пр-кт": "пр-кт",
    "пр.": "пр-кт",
    "переулок": "пер",
    "пер.": "пер",
    "пер": "пер",
    "шоссе": "ш",
    "бульвар": "б-р",
}


def extract_streets_from_text(text: str) -> list[str]:
    """
    Из текста поста извлекает упоминания улиц/проспектов/переулков.
    Возвращает нормализованные строки в формате БД: ['ул Шилкинская', 'пр-кт Океанский'].
    """
    if not text:
        return []

    found: set[str] = set()
    for match in RE_STREET.finditer(text):
        full = match.group(0)
        name_part = match.group(1).strip()

        # Определяем префикс по началу матча
        prefix_match = re.match(r"^[А-Яа-яёЁA-Za-z\.\-]+", full)
        prefix_raw = prefix_match.group(0).lower() if prefix_match else "ул"
        prefix_norm = STREET_PREFIX_NORMALIZATION.get(prefix_raw, "ул")

        # Нормализуем падежи названия (приводим к именительному "Светланская")
        # Простая эвристика: если кончается на "ой/ой/ую/ей" — это падежная форма
        # Однако делать корректную лемматизацию здесь дорого.
        # Поэтому в matching сравниваем по подстроке.
        candidate = f"{prefix_norm} {name_part}"
        found.add(candidate)

    return sorted(found)


# ────────────────────────────────────────────────────────────────────
# Mapping STRONG-keywords из NER → em_type БД
# ────────────────────────────────────────────────────────────────────

# Если в тексте поста встретилось одно из ключевых слов,
# em_type БД должен начинаться/содержать соответствующую подстроку.
KEYWORDS_TO_EM_TYPE = [
    (
        [
            "столкновени",
            "совершил столкнов",
            "допустил столкнов",
            "столкнулся с автомоб",
            "столкнулся с автомаш",
        ],
        "Столкновение",
    ),
    (["наезд на пешеход", "сбил", "сбила", "наехал", "наехала"], "Наезд на пешехода"),
    (["опрокидыван", "опрокинул", "опрокинулся"], "Опрокидывание"),
    (["съезд с дороги", "съехал", "съехала"], "Съезд с дороги"),
    (["наезд на препятстви"], "Наезд на препятствие"),
    (["наезд на стоящее"], "Наезд на стоящее ТС"),
    (["велосипедист"], "Наезд на велосипедиста"),
    (
        ["мотоциклист"],
        # мотоциклы относятся обычно к Столкновению/Наезду — оставим без сильного matchа
        None,
    ),
]


def detect_expected_em_type(text: str) -> Optional[str]:
    """
    По тексту поста определяет ожидаемый em_type БД.
    Возвращает None если из текста нельзя однозначно определить.
    """
    if not text:
        return None
    text_lower = text.lower()
    for keywords, em_type in KEYWORDS_TO_EM_TYPE:
        if em_type is None:
            continue
        for kw in keywords:
            if kw in text_lower:
                return em_type
    return None


# ────────────────────────────────────────────────────────────────────
# Severity matching: NER-keywords → severity БД
# ────────────────────────────────────────────────────────────────────


def detect_severity_from_text(text: str) -> Optional[str]:
    """
    По тексту поста определяет ожидаемое severity БД.
    Возвращает 'dead' / 'severe' / 'light' / None.
    """
    if not text:
        return None
    text_lower = text.lower()

    # Сильные сигналы смерти
    death_markers = [
        "погиб на месте",
        "погибл на месте",
        "погиб в дтп",
        "погибла в дтп",
        "погибли в дтп",
        "скончал",
        "смертельн",
        "смертельный исход",
    ]
    for marker in death_markers:
        if marker in text_lower:
            return "dead"

    # Сильные ранения
    severe_markers = [
        "госпитализирован",
        "получил телесные",
        "тяжёлые травмы",
        "тяжелые травмы",
        "реанима",
    ]
    for marker in severe_markers:
        if marker in text_lower:
            return "severe"

    # Лёгкие травмы
    light_markers = ["амбулаторное лечение", "получил травмы", "получила травмы"]
    for marker in light_markers:
        if marker in text_lower:
            return "light"

    return None


# ────────────────────────────────────────────────────────────────────
# Vehicle marka matching
# ────────────────────────────────────────────────────────────────────


def count_brand_overlap(
    post_brands: list[str],  # ['TOYOTA', 'NISSAN'] из NER
    db_marks: list[str],  # ['TOYOTA', 'MAZDA'] из БД vehicles
) -> int:
    """
    Сколько марок в посте совпадает с марками в БД.
    Регистронезависимо. NISSAN_TYPO → NISSAN в matcher не нужно (уже в NER).
    """
    if not post_brands or not db_marks:
        return 0
    post_set = {b.upper() for b in post_brands}
    db_set = {m.upper() for m in db_marks}
    return len(post_set & db_set)


# ────────────────────────────────────────────────────────────────────
# Главная функция: matching одного поста с очками
# ────────────────────────────────────────────────────────────────────


def match_post(
    post_date: date,
    post_text: str,
    post_locations: list[str],  # нормализованные ['г Владивосток']
    post_kilometers: list[int],
    post_streets: list[str],  # ['ул Светланская', 'пр-кт Океанский']
    post_brands: list[str],  # ['TOYOTA', 'NISSAN']
    candidates: list[dict],
    days_before: int = 3,
    score_threshold: int = 30,  # минимальный score чтобы попасть в результат
) -> list[dict]:
    """
    Скорит каждого кандидата по 5 факторам и возвращает тех у кого
    score >= score_threshold.

    candidates содержат поля:
      id, datetime, np, road_km, street, severity, em_type,
      vehicle_marks: list[str]   ← marks из всех vehicles ДТП

    Каждый матч получает поле:
      'score': int (0-130)
      'score_breakdown': dict с разбивкой
    """
    if not candidates:
        return []

    earliest = post_date - timedelta(days=days_before)
    latest = post_date

    location_set = set(post_locations)
    km_set = set(post_kilometers)
    expected_em = detect_expected_em_type(post_text)
    expected_severity = detect_severity_from_text(post_text)
    streets_set = set(s.lower() for s in post_streets)

    matches: list[dict] = []

    for c in candidates:
        c_date = c["datetime"].date() if c.get("datetime") else None
        if c_date is None:
            continue
        if not (earliest <= c_date <= latest):
            continue

        score = 0
        breakdown: dict = {}

        # 1. np (+30)
        if c.get("np") and c["np"] in location_set:
            score += 30
            breakdown["np"] = 30

        # 2. street (+30) — точное совпадение по подстроке
        c_street = (c.get("street") or "").lower()
        if c_street and streets_set:
            for s in streets_set:
                # Сравнение: ищем имя улицы (после префикса) в обе стороны
                # Пост: "ул Светланская"  → "светланская"
                # БД:   "ул Светланская"  → "светланская"
                s_name = re.sub(r"^(ул|пр-кт|пер|ш|б-р)\s+", "", s).strip()
                c_name = re.sub(r"^(ул|пр-кт|пер|ш|б-р)\s+", "", c_street).strip()
                if s_name and c_name and (s_name in c_name or c_name in s_name):
                    score += 30
                    breakdown["street"] = 30
                    break

        # 3. em_type (+20)
        c_em = c.get("em_type") or ""
        if expected_em and expected_em in c_em:
            score += 20
            breakdown["em_type"] = 20

        # 4. severity (+15)
        if expected_severity and c.get("severity"):
            # severity_multiple → совпадает с severity
            c_sev = c["severity"]
            if expected_severity == "dead" and c_sev == "dead":
                score += 15
                breakdown["severity"] = 15
            elif expected_severity == "severe" and c_sev in (
                "severe",
                "severe_multiple",
            ):
                score += 15
                breakdown["severity"] = 15
            elif expected_severity == "light" and c_sev == "light":
                score += 15
                breakdown["severity"] = 15

        # 5. vehicle marks (+25 за каждую совпавшую марку, max +50)
        marks_in_db = c.get("vehicle_marks", [])
        overlap = count_brand_overlap(post_brands, marks_in_db)
        if overlap > 0:
            mark_score = min(overlap * 25, 50)
            score += mark_score
            breakdown["vehicle_marks"] = mark_score

        # 6. road_km (+10)
        c_km = c.get("road_km") or 0.0
        if km_set and c_km > 0:
            for km in km_set:
                if abs(c_km - km) <= 2.0:
                    score += 10
                    breakdown["road_km"] = 10
                    break

        if score >= score_threshold:
            matches.append(
                {
                    **c,
                    "score": score,
                    "score_breakdown": breakdown,
                }
            )

    # Сортируем по убыванию score
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches
