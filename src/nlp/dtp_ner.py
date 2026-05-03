# -*- coding: utf-8 -*-
"""
NER-извлечение сущностей из ДТП-постов УМВД Приморья.

Гибридный подход:
- Natasha NER (NewsNERTagger) → LOC, ORG, PER из коробки
- Регулярные выражения → марки ТС, возрасты, время суток, стаж вождения
- Постобработка → дедупликация, фильтр PER от ложных срабатываний на марках

Главная функция:
    extract_all(text) -> dict с ключами:
        - vehicles:  список марок ТС (Toyota, Subaru, ...) с моделями
        - ages:      возраст + контекст (водитель, пешеход)
        - times:     "23:20", "около полуночи"
        - locations: географические объекты из Natasha
        - orgs:      организации (МВД, ГИБДД)
        - persons:   имена людей (отфильтрованы от марок ТС)
        - stats:     служебная информация
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Segmenter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BRAND_REFERENCE_CSV = PROJECT_ROOT / "data" / "raw" / "vehicle_brand_steering.csv"


# ────────────────────────────────────────────────────────────────────
# Лениво создаваемые компоненты Natasha
# ────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_natasha_components() -> tuple[Segmenter, NewsMorphTagger, NewsNERTagger, MorphVocab]:
    """Создаёт компоненты Natasha. Кэшируется один раз за процесс."""
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    ner_tagger = NewsNERTagger(emb)
    morph_vocab = MorphVocab()
    return segmenter, morph_tagger, ner_tagger, morph_vocab


# ────────────────────────────────────────────────────────────────────
# Список марок ТС из нашего справочника
# ────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_brand_list() -> list[str]:
    """Загружает марки из справочника, сортирует по длине (длинные первыми)."""
    df = pd.read_csv(BRAND_REFERENCE_CSV, encoding="utf-8")
    brands = df["brand"].dropna().tolist()
    brands.sort(key=len, reverse=True)
    return brands


# Альтернативные написания на русском (как пишут в постах УМВД)
RUSSIAN_BRAND_ALIASES: dict[str, list[str]] = {
    "TOYOTA": ["тойота", "тоёта"],
    "NISSAN": ["ниссан", "нисан"],
    "HONDA": ["хонда"],
    "MAZDA": ["мазда"],
    "MITSUBISHI": ["мицубиси", "митсубиси", "митсубиши"],
    "SUBARU": ["субару", "субари"],
    "SUZUKI": ["сузуки", "судзуки"],
    "DAIHATSU": ["дайхатсу"],
    "ISUZU": ["исузу", "исудзу"],
    "LEXUS": ["лексус"],
    "INFINITI": ["инфинити"],
    "ACURA": ["акура"],
    "BMW": ["бмв"],
    "MERCEDES": ["мерседес"],
    "MERCEDES-BENZ": ["мерседес-бенц", "мерседес бенц"],
    "AUDI": ["ауди"],
    "VOLKSWAGEN": ["фольксваген", "вольксваген"],
    "PORSCHE": ["порше"],
    "FORD": ["форд"],
    "CHEVROLET": ["шевроле"],
    "HYUNDAI": ["хёндай", "хундай", "хюндай"],
    "KIA": ["киа"],
    "DAEWOO": ["дэу", "дэо", "деу"],
    "VOLVO": ["вольво", "волво"],
    "RENAULT": ["рено"],
    "PEUGEOT": ["пежо"],
    "CITROEN": ["ситроен"],
    "FIAT": ["фиат"],
    "SCANIA": ["скания"],
    "MAN": ["ман"],
    "IVECO": ["ивеко"],
    "DAF": ["даф"],
    "FREIGHTLINER": ["фрейтлайнер", "фредлайнер"],
    "HOWO": ["хово"],
    "SHACMAN": ["шакман"],
    "LAND ROVER": ["ленд ровер", "лэнд ровер"],
    "JAGUAR": ["ягуар"],
    "MINI": ["мини купер"],  # без "мини" — слишком частое слово
    "JEEP": ["джип"],
    # Китайские марки (часто в постах УМВД)
    "CHERY": ["чери", "черри", "чери тигго"],
    "HAVAL": ["хавал", "хаваль"],
    "EXEED": ["эксид"],
    "GEELY": ["джили", "джилли"],
    "GREAT WALL": ["грейт волл", "грейт уолл"],
    "LIFAN": ["лифан"],
    "CHANGAN": ["чанган"],
    "JAC": ["жак"],
    "FAW": ["фав"],
    "BYD": ["бид", "бьюд"],
    "DONGFENG": ["донгфенг", "донфенг"],
    "ZEEKR": ["зикр"],
    # Российские марки
    "LADA": [
        "лада",
        "ваз",
        "жигули",
        "приора",
        "калина",
        "веста",
        "гранта",
        "нива",
        "шевроле нива",
    ],
    "UAZ": ["уаз", "патриот"],
    "GAZ": ["газ", "газель", "соболь", "волга"],
    "KAMAZ": ["камаз"],
    "KRAZ": ["краз"],
    "MAZ": ["маз"],
    "MOSKVICH": ["москвич"],
    "IZH": ["иж"],
    "ZIL": ["зил"],
    "BELAZ": ["белаз"],
    # Опечатки и альтернативные написания
    "NISSAN_TYPO": ["нисан"],  # без двух "с" — частая опечатка
}


# ────────────────────────────────────────────────────────────────────
# Регулярные выражения
# ────────────────────────────────────────────────────────────────────

# "63-летний водитель", "23-летняя женщина" — берём прилагательное возраста
# и СУЩЕСТВИТЕЛЬНОЕ после (без хвостов "не справился", "за рулём" и т.п.)
RE_AGE_HYPHEN = re.compile(
    r"(\d{1,3})[-‐]летн(?:ий|яя|его|ей|ему|ие|их)\s+"
    r"([А-Яа-яёЁ]+)",  # ровно одно существительное-кандидат
    re.IGNORECASE,
)

# "47 лет водитель", "44 года женщина", "70-летнюю женщину" не покрывает —
# покрывает прямой формат число+существительное-указатель.
RE_AGE_NUM_NOUN = re.compile(
    r"(\d{1,3})\s+(?:лет|год|года|годика?)\s+"
    r"(водитель|водительниц|пешеход|мужчин|женщин|мальчик|девочк|"
    r"подросток|пенсионер|пенсионерк|ребен|ребён|школьник|школьниц|"
    r"парен|девушк)",
    re.IGNORECASE,
)

# "в возрасте 47 лет", "возрастом 32 года"
RE_AGE_VOZRASTE = re.compile(
    r"возраст(?:е|ом)?\s+(\d{1,3})\s+(?:лет|год|года)",
    re.IGNORECASE,
)

# Время в формате "23:20", "23.20", "07:00".
# ВАЖНО: с двоеточием — почти всегда время, с точкой — может быть номером статьи КоАП.
# Поэтому:
# - "23:20" — берём всегда (двоеточие)
# - "23.20" — берём только если рядом контекст времени ("в", "около", "в ночь" и т.п.)
RE_TIME_COLON = re.compile(r"\b(\d{1,2}):(\d{2})\b")
RE_TIME_DOT = re.compile(
    r"(?:\b(?:в|около|примерно|приблизительно)\s+)"  # обязательный временной маркер ПЕРЕД
    r"(\d{1,2})\.(\d{2})\b"
    r"(?!\s*(?:КоАП|УК|ст\.|ч\.))",  # и НЕ перед "КоАП"/"УК"/"ст."/"ч."
    re.IGNORECASE,
)
# Альтернативный паттерн с точкой: "В 12.30 в дежурную часть..." — точка с временем,
# но без явного "в", "около". Распознаём по контексту "час", "сутки", "минут" рядом.
RE_TIME_DOT_BACKWARD = re.compile(
    r"\b(\d{1,2})\.(\d{2})\b" r"(?=\s*(?:в|на|произошло|поступило|случилось|зафиксирован))",
    re.IGNORECASE,
)
# Запрещаем формат если перед числом стоит "ст." / "ч." / "КоАП" — это статья
RE_KOAP_BLOCK = re.compile(
    r"(?:ст\.|ч\.|часть|статья|КоАП|УК)\s*[\d\.]*",
    re.IGNORECASE,
)

# Стаж вождения — несколько форматов:
# - "стаж вождения 5 лет"
# - "стаж управления — 12 лет"
# - "водительский стаж составляет 28 лет"
# - "стаж — 30 лет" / "со стажем 8 лет"
RE_DRIVING_EXPERIENCE = re.compile(
    r"(?:водительск(?:ий|им|ого)\s+)?"
    r"стаж(?:\s+(?:вождения|управления|водителя))?"
    r"(?:\s+(?:составляет|составил|равен))?"
    r"\s*[\-–—]?\s*(\d{1,2})\s*(?:год|лет|года)",
    re.IGNORECASE,
)

# Километровка трассы: "на 599 км", "82 км автодороги"
RE_KILOMETER = re.compile(r"(?:на\s+)?(\d{1,4})\s*км\b", re.IGNORECASE)

# Слова которые ИГНОРИРУЕМ как "контекст возраста" — служебные/глаголы
AGE_CONTEXT_BLACKLIST = {
    "не",
    "ни",
    "за",
    "с",
    "со",
    "в",
    "и",
    "или",
    "а",
    "но",
    "ему",
    "ей",
    "его",
    "её",
    "из",
    "у",
    "к",
    "от",
    "до",
    "на",
    "по",
    "при",
    "про",
    "был",
    "была",
    "будет",
    "был",
    "стало",
    "стал",
    "также",
    "тоже",
}


# ────────────────────────────────────────────────────────────────────
# Извлечение марок ТС (с дедупликацией)
# ────────────────────────────────────────────────────────────────────


def extract_vehicles(text: str) -> list[dict]:
    """
    Извлекает упоминания марок ТС с моделями.

    Дедуплицируется по позиции бренда. Каждое отдельное вхождение бренда
    в текст возвращается с попыткой захватить его модель.

    Возвращает: [{"brand": "TOYOTA", "matched": "Toyota Land Cruiser",
    "position": 123}, ...]
    """
    if not text:
        return []

    brands = _load_brand_list()
    found: list[dict] = []
    # Множество хранит ТОЛЬКО позиции уже найденных БРЕНДОВ (а не моделей),
    # чтобы один и тот же бренд не сматчился по латинице и по алиасу,
    # но модель захватывалась независимо для каждого вхождения.
    used_brand_positions: set[int] = set()

    def capture_model(end_pos: int) -> tuple[str, int]:
        """
        Захватывает модель — до 3 слов после марки.
        Возвращает (текст_модели_с_пробелом_впереди, новая_конечная_позиция).
        Если модель не нашлась — возвращает ("", end_pos).
        """
        after = text[end_pos : end_pos + 80]
        model_match = re.match(
            r"\s+([A-Za-zА-Яа-яёЁ0-9\-]+(?:\s+[A-Za-zА-Яа-яёЁ0-9\-]+){0,2})",
            after,
        )
        if not model_match:
            return "", end_pos

        model_text = model_match.group(1)
        model_words = model_text.split()
        # Принимаем модель только если все слова "разумные":
        # короткие или с цифрами/дефисом, или начинаются с заглавной
        ok_model = all(
            len(w) <= 12
            and (w.isupper() or w[:1].isupper() or any(c.isdigit() for c in w) or "-" in w)
            for w in model_words
        )
        if ok_model:
            return " " + model_text, end_pos + len(model_match.group(0))
        return "", end_pos

    def try_match(pattern: re.Pattern, brand_name: str) -> None:
        for match in pattern.finditer(text):
            start = match.start()
            if start in used_brand_positions:
                continue
            end = match.end()
            model_str, _ = capture_model(end)
            full_match = (match.group(0) + model_str).strip()
            found.append(
                {
                    "brand": brand_name,
                    "matched": full_match,
                    "position": start,
                }
            )
            used_brand_positions.add(start)

    # 1. Латинские бренды (длинные первыми)
    for brand in brands:
        pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)
        try_match(pattern, brand.upper())

    # 2. Русские алиасы
    for brand_eng, aliases in RUSSIAN_BRAND_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            try_match(pattern, brand_eng)

    # Нормализация опечаток: NISSAN_TYPO → NISSAN
    BRAND_NORMALIZE = {
        "NISSAN_TYPO": "NISSAN",
    }
    for v in found:
        if v["brand"] in BRAND_NORMALIZE:
            v["brand"] = BRAND_NORMALIZE[v["brand"]]

    found.sort(key=lambda x: x["position"])
    return found


# ────────────────────────────────────────────────────────────────────
# Извлечение возрастов (с улучшенным контекстом)
# ────────────────────────────────────────────────────────────────────


def extract_ages(text: str) -> list[dict]:
    """
    Извлекает возрасты с контекстом из 3 разных форматов:
    - "63-летний водитель" / "23-летняя женщина"
    - "47 лет водитель" / "44 года женщина"
    - "в возрасте 32 года"

    Возвращает: [{"age": 63, "context": "водитель",
                  "match": "63-летний водитель"}]
    """
    if not text:
        return []

    found: list[dict] = []
    used_positions: set[int] = set()

    def add_age(age: int, context: str, match_text: str, position: int) -> None:
        if age < 1 or age > 110:
            return
        # Дедуп — если в этой позиции уже что-то есть
        if any(abs(position - p) < 5 for p in used_positions):
            return
        if context in AGE_CONTEXT_BLACKLIST:
            context = "?"
        found.append(
            {
                "age": age,
                "context": context,
                "match": match_text,
            }
        )
        used_positions.add(position)

    # 1. "63-летний водитель"
    for match in RE_AGE_HYPHEN.finditer(text):
        try:
            age = int(match.group(1))
        except ValueError:
            continue
        context = match.group(2).lower().strip()
        add_age(age, context, match.group(0).strip(), match.start())

    # 2. "47 лет водитель", "44 года женщина"
    for match in RE_AGE_NUM_NOUN.finditer(text):
        try:
            age = int(match.group(1))
        except ValueError:
            continue
        context = match.group(2).lower().strip()
        add_age(age, context, match.group(0).strip(), match.start())

    # 3. "в возрасте 47 лет"
    for match in RE_AGE_VOZRASTE.finditer(text):
        try:
            age = int(match.group(1))
        except ValueError:
            continue
        add_age(age, "?", match.group(0).strip(), match.start())

    return found


# ────────────────────────────────────────────────────────────────────
# Время суток
# ────────────────────────────────────────────────────────────────────


def extract_times(text: str) -> list[dict]:
    """
    Извлекает время суток ДТП в формате HH:MM или HH.MM.

    Стратегия:
    - "23:20" (с двоеточием) — это всегда время.
    - "23.20" (с точкой) — может быть номером статьи КоАП. Поэтому требуем
      контекстный маркер ("в", "около") и исключаем КоАП-маркеры.
    """
    if not text:
        return []

    found: list[dict] = []
    used_positions: set[tuple[int, int]] = set()

    # 1. С двоеточием — всегда берём
    for match in RE_TIME_COLON.finditer(text):
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
        except ValueError:
            continue
        if hour > 24 or minute > 59:
            continue
        found.append(
            {
                "hour": hour,
                "minute": minute,
                "match": match.group(0),
            }
        )
        used_positions.add((match.start(), match.end()))

    # 2. С точкой и явным временным маркером ("в 12.30")
    for match in RE_TIME_DOT.finditer(text):
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
        except ValueError:
            continue
        if hour > 24 or minute > 59:
            continue
        # Проверим что это не вложено в уже найденное
        time_start = match.start(1)  # позиция самого числа, не маркера "в"
        time_end = match.end(2)
        if any(s <= time_start < e for s, e in used_positions):
            continue
        found.append(
            {
                "hour": hour,
                "minute": minute,
                "match": text[time_start:time_end],
            }
        )
        used_positions.add((time_start, time_end))

    return found


# ────────────────────────────────────────────────────────────────────
# Стаж вождения и километровка
# ────────────────────────────────────────────────────────────────────


def extract_driving_experience(text: str) -> list[int]:
    if not text:
        return []
    return [
        int(m.group(1)) for m in RE_DRIVING_EXPERIENCE.finditer(text) if 0 <= int(m.group(1)) <= 80
    ]


def extract_kilometers(text: str) -> list[int]:
    if not text:
        return []
    return [int(m.group(1)) for m in RE_KILOMETER.finditer(text) if 0 <= int(m.group(1)) <= 9999]


# ────────────────────────────────────────────────────────────────────
# Natasha NER + фильтр персон от марок ТС
# ────────────────────────────────────────────────────────────────────
# Стоп-слова для фильтрации галлюцинаций Persons
# Часто Natasha находит как PER слова с заглавной буквы, которые на самом деле
# топонимы или прилагательные ("Приморья", "Владивостока", ...)
PERSON_BLACKLIST_SUBSTRINGS = {
    "примор",
    "владивост",
    "уссурийск",
    "находк",
    "чугуев",
    "хасан",
    "артем",
    "арсеньев",
    "лесозавод",
    "дальнегор",
    "дальнереч",
    "спасск",
    "партизан",
    "фокино",
    "кавалер",
    "пожар",
    "хорол",
    "октябр",
    "погранич",
    "красноарм",
    "ольгин",
    "тернейск",
    "ханкайск",
    "лазов",
    "яковлев",
    "анучин",
    "михайлов",
    "надеждин",
    "шкотов",
    "ленинск",
    "первореч",
    "первомай",
    "советск",
    "фрунзен",
    "россии",
    "россия",
    "приморь",
    "тойот",
    "ниссан",
    "хонда",
    "мазда",
    "субар",
    "сузуки",
    "мерседес",
    "тойота",
    "форд",
    "шевроле",
    "субарy",
}


def _looks_like_person(span_text: str, full_text: str) -> bool:
    """
    Проверяет является ли span реально именем человека (не галлюцинацией).

    Критерии:
    1. span_text должен присутствовать в full_text дословно (cross-validation)
    2. span_text не содержит запрещённых подстрок (топонимы, марки)
    3. span_text не короче 3 символов
    """
    if len(span_text) < 3:
        return False

    if span_text not in full_text:
        return False  # галлюцинация — Natasha вернула то чего нет

    span_lower = span_text.lower()
    for blacklisted in PERSON_BLACKLIST_SUBSTRINGS:
        if blacklisted in span_lower:
            return False

    return True


def extract_natasha_entities(text: str, vehicle_strings: set[str] | None = None) -> dict:
    """
    Прогоняет текст через Natasha. Фильтрует Persons:
    - Исключает спаны, текст которых не найден в исходном тексте дословно
    - Исключает спаны со стоп-словами (топонимы, марки ТС)
    - Исключает спаны совпадающие с найденными марками ТС
    """
    if not text:
        return {"locations": [], "orgs": [], "persons": []}

    segmenter, morph_tagger, ner_tagger, morph_vocab = _get_natasha_components()
    vehicle_strings = vehicle_strings or set()
    vehicle_strings_lower = {vs.lower() for vs in vehicle_strings}

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)

    locations: list[dict] = []
    orgs: list[dict] = []
    persons: list[dict] = []

    for span in doc.spans:
        entry = {
            "text": span.text,
            "start": span.start,
            "stop": span.stop,
        }
        if span.type == "LOC":
            locations.append(entry)
        elif span.type == "ORG":
            orgs.append(entry)
        elif span.type == "PER":
            # 1. Cross-validation: текст должен быть в исходнике
            # 2. Не топоним и не марка
            if not _looks_like_person(span.text, text):
                continue

            # 3. Не совпадает с найденной маркой
            span_text_lower = span.text.lower().strip()
            is_vehicle = any(
                vs in span_text_lower or span_text_lower in vs for vs in vehicle_strings_lower
            )
            if is_vehicle:
                continue

            persons.append(entry)

    return {
        "locations": locations,
        "orgs": orgs,
        "persons": persons,
    }


# ────────────────────────────────────────────────────────────────────
# Главная функция — extract_all
# ────────────────────────────────────────────────────────────────────


def extract_all(text: str) -> dict:
    """Полный pipeline извлечения сущностей из текста."""
    if not text:
        return {
            "vehicles": [],
            "ages": [],
            "times": [],
            "driving_experience_years": [],
            "kilometers": [],
            "locations": [],
            "orgs": [],
            "persons": [],
            "stats": {"text_length": 0, "total_entities": 0},
        }

    vehicles = extract_vehicles(text)
    ages = extract_ages(text)
    times = extract_times(text)
    driving_exp = extract_driving_experience(text)
    kilometers = extract_kilometers(text)

    # Передаём в Natasha список найденных марок,
    # чтобы она не возвращала их как Persons
    vehicle_strings = {v["matched"] for v in vehicles}
    natasha_ents = extract_natasha_entities(text, vehicle_strings)

    total = (
        len(vehicles)
        + len(ages)
        + len(times)
        + len(driving_exp)
        + len(kilometers)
        + len(natasha_ents["locations"])
        + len(natasha_ents["orgs"])
        + len(natasha_ents["persons"])
    )

    return {
        "vehicles": vehicles,
        "ages": ages,
        "times": times,
        "driving_experience_years": driving_exp,
        "kilometers": kilometers,
        "locations": natasha_ents["locations"],
        "orgs": natasha_ents["orgs"],
        "persons": natasha_ents["persons"],
        "stats": {
            "text_length": len(text),
            "total_entities": total,
        },
    }
