# -*- coding: utf-8 -*-
"""
Обогащение БД возрастами участников ДТП через Telegram NLP-pipeline.

Работает строго на gold-парах (top_score >= 90 && top_count == 1) из.
Цель — точность, а не покрытие: лучше пропустить спорный случай, чем
поставить неправильный возраст.

Pipeline:
    1. Загружаем gold-пары из telegram_db_matches.jsonl
    2. Загружаем NER-возраста (с контекстом) из telegram_ner_results.jsonl
    3. Для каждого поста:
       а. Проверяем уникальность ролей в ages — если 2+ возрастов с одинаковой
          ролью (двое "водитель"), пост описывает несколько ДТП → skip целиком
       б. Загружаем участников связанного ДТП (participants + accident_pedestrians)
       в. Для каждого age классифицируем context → ищем кандидатов в ДТП:
          - Если ровно 1 кандидат — назначаем
          - Если 0 или ≥2 — skip этот age (privacy preferred over coverage)
          - Уже назначенные кандидаты исключаются из последующих age в этом ДТП
    4. UPDATE participants/accident_pedestrians с возрастом + аудит-следом
       (age_source, age_match_context, age_match_post_id)

Колонки результата:
    - age_from_telegram     INT — собственно возраст
    - age_source            VARCHAR(50) — 'telegram_gold'
    - age_match_context     VARCHAR(50) — оригинальный NER-контекст
    - age_match_post_id     INT — tg_id поста (трассировка)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

# =====================================================================
# Константы и конфигурация
# =====================================================================

# Источник возраста — единственное значение для текущего pipeline
SOURCE_GOLD = "telegram_gold"

# Минимальный score чтобы пара пост↔ДТП считалась "gold"
GOLD_TOP_SCORE = 90

# Категории контекста — приоритет для распознавания
# Каждый ключ маппится в роль в БД + опционально пол / признак "ребёнок"
# Порядок словаря важен — приоритет первого совпадения регулярки.

# Регулярки для классификации NER-контекста.
# context — это лемматизированное слово после возраста, например:
#   "водитель", "водительница", "пешеход", "мужчина", "пенсионерка",
#   "мальчик", "девочка", "школьник", "подросток", "ребёнок"
# Используем re.search для устойчивости к окончаниям.
CONTEXT_PATTERNS: list[tuple[re.Pattern, dict]] = [
    # Водители
    (
        re.compile(r"^водител", re.IGNORECASE),
        {"category": "driver", "sex": None, "is_child": False},
    ),
    # Пассажиры
    (
        re.compile(r"^пассажир", re.IGNORECASE),
        {"category": "passenger", "sex": None, "is_child": False},
    ),
    # Пешеходы (общий)
    (
        re.compile(r"^пешеход", re.IGNORECASE),
        {"category": "pedestrian", "sex": None, "is_child": False},
    ),
    # Велосипедисты
    (
        re.compile(r"^велосипедист", re.IGNORECASE),
        {"category": "cyclist", "sex": None, "is_child": False},
    ),
    # Мотоциклисты
    (
        re.compile(r"^мотоцикл", re.IGNORECASE),
        {"category": "motorcyclist", "sex": None, "is_child": False},
    ),
    # Дети-мальчики (всегда пешеходы или пассажиры — но в посте обычно пешеход)
    (
        re.compile(r"^мальчик", re.IGNORECASE),
        {"category": "child_pedestrian", "sex": "Мужской", "is_child": True},
    ),
    # Дети-девочки
    (
        re.compile(r"^девочк", re.IGNORECASE),
        {"category": "child_pedestrian", "sex": "Женский", "is_child": True},
    ),
    # Школьники / школьницы (sex по существу)
    (
        re.compile(r"^школьник", re.IGNORECASE),
        {"category": "child_pedestrian", "sex": "Мужской", "is_child": True},
    ),
    (
        re.compile(r"^школьниц", re.IGNORECASE),
        {"category": "child_pedestrian", "sex": "Женский", "is_child": True},
    ),
    # Подросток (пол неизвестен)
    (
        re.compile(r"^подростк?", re.IGNORECASE),
        {"category": "child_any", "sex": None, "is_child": True},
    ),
    # Ребёнок (пол неизвестен)
    (
        re.compile(r"^реб[её]н", re.IGNORECASE),
        {"category": "child_any", "sex": None, "is_child": True},
    ),
    # Пенсионер / пенсионерка — пол по существу, любая роль
    (
        re.compile(r"^пенсионер[ка]?$|^пенсионер$", re.IGNORECASE),
        {"category": "any_role", "sex": "Мужской", "is_child": False},
    ),
    (
        re.compile(r"^пенсионерк", re.IGNORECASE),
        {"category": "any_role", "sex": "Женский", "is_child": False},
    ),
    # Мужской пол
    (
        re.compile(r"^мужчин|^парен|^парн", re.IGNORECASE),
        {"category": "any_role", "sex": "Мужской", "is_child": False},
    ),
    # Женский пол
    (
        re.compile(r"^женщин|^девушк", re.IGNORECASE),
        {"category": "any_role", "sex": "Женский", "is_child": False},
    ),
]


def classify_context(context: str) -> Optional[dict]:
    """
    Определяет категорию участника по контексту NER.

    Возвращает dict с полями:
        - category: 'driver' / 'passenger' / 'pedestrian' / 'cyclist' /
                    'motorcyclist' / 'child_pedestrian' / 'child_any' /
                    'any_role'
        - sex: 'Мужской' / 'Женский' / None
        - is_child: bool

    Возвращает None если контекст не распознан (контекст='?' или
    шумовое слово — например, "автомобиль", "иномарка").
    """
    if not context or context.strip() in ("", "?"):
        return None
    norm = context.strip().lower()
    for pattern, classification in CONTEXT_PATTERNS:
        if pattern.match(norm):
            return classification
    return None


# =====================================================================
# Загрузка входных данных
# =====================================================================


@dataclass
class PostData:
    """Один Telegram-пост с матчем и NER-возрастами."""

    tg_id: int
    accident_id: int  # db_id из gold-матча
    top_score: int
    ages: list[dict]  # [{age, context, match}, ...]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gold_posts(
    matches_path: Path,
    ner_path: Path,
    gold_top_score: int = GOLD_TOP_SCORE,
) -> list[PostData]:
    """
    Загружает gold-пары пост↔ДТП и сшивает их с NER-возрастами.

    Gold-критерий:
        - top_score >= gold_top_score (90 по умолчанию)
        - top_count == 1 (один лучший кандидат, без неоднозначности)
        - n_matches >= 1
        - в посте есть хотя бы один age с непустым контекстом
    """
    # NER → tg_id → ages
    ner_by_id: dict[int, list[dict]] = {}
    for ner in iter_jsonl(ner_path):
        if ner.get("ages"):
            ner_by_id[ner["tg_id"]] = ner["ages"]

    posts: list[PostData] = []
    for match in iter_jsonl(matches_path):
        if match.get("top_count") != 1:
            continue
        if match.get("top_score", 0) < gold_top_score:
            continue
        if match.get("n_matches", 0) < 1:
            continue
        ages = ner_by_id.get(match["tg_id"], [])
        if not ages:
            continue
        # Только возраста с распознаваемым контекстом
        usable = [a for a in ages if classify_context(a.get("context", ""))]
        if not usable:
            continue
        accident_id = match["matches"][0]["db_id"]
        posts.append(
            PostData(
                tg_id=match["tg_id"],
                accident_id=accident_id,
                top_score=match["top_score"],
                ages=usable,
            )
        )
    return posts


# =====================================================================
# Кандидаты в ДТП: загружаем участников связанного ДТП
# =====================================================================


@dataclass
class Candidate:
    """Кандидат для назначения возраста — участник или пешеход."""

    table: str  # 'participants' / 'accident_pedestrians'
    row_id: int
    part_type: Optional[str]
    sex: Optional[str]
    vehicle_prod_type: Optional[str] = None  # для motorcyclist


def load_accident_candidates(session: Session, accident_id: int) -> list[Candidate]:
    """
    Загружает всех участников ДТП:
    - participants внутри vehicles (водители, пассажиры, велосипедисты)
    - accident_pedestrians (пешеходы и пр.)

    Возвращает список Candidate с метаданными для matching.
    """
    candidates: list[Candidate] = []

    # Участники внутри ТС
    rows = session.execute(
        text(
            """
            SELECT p.id, p.part_type, p.sex, v.prod_type
            FROM participants p
            JOIN vehicles v ON v.id = p.vehicle_id
            WHERE v.accident_id = :acc_id
            """
        ),
        {"acc_id": accident_id},
    ).fetchall()
    for row in rows:
        candidates.append(
            Candidate(
                table="participants",
                row_id=row.id,
                part_type=row.part_type,
                sex=row.sex,
                vehicle_prod_type=row.prod_type,
            )
        )

    # Пешеходы (и иные участники без ТС)
    rows = session.execute(
        text(
            """
            SELECT id, part_type, sex
            FROM accident_pedestrians
            WHERE accident_id = :acc_id
            """
        ),
        {"acc_id": accident_id},
    ).fetchall()
    for row in rows:
        candidates.append(
            Candidate(
                table="accident_pedestrians",
                row_id=row.id,
                part_type=row.part_type,
                sex=row.sex,
            )
        )

    return candidates


# =====================================================================
# Логика отбора кандидатов по классификации контекста
# =====================================================================


def is_pedestrian_part_type(part_type: Optional[str]) -> bool:
    if not part_type:
        return False
    return "Пешеход" in part_type


def is_motorcycle(prod_type: Optional[str]) -> bool:
    if not prod_type:
        return False
    pt = prod_type.lower()
    return ("мотоцикл" in pt) or ("мопед" in pt) or ("мотороллер" in pt) or ("мототранспортн" in pt)


# Минимальные возраста для valid-назначения по роли:
# - Водитель легкового/грузового авто:  16 (B — с 18, но запасом до 16)
# - Водитель мотоцикла/мопеда:          14 (мопед в РФ — с 16, но запас)
# - Велосипедист:                        4
# - Пассажир / пешеход:                  0
MIN_AGE_DRIVER_CAR = 16
MIN_AGE_DRIVER_MOTO = 14
MIN_AGE_CYCLIST = 4


def validate_assignment(age: int, candidate: "Candidate") -> bool:
    """
    Sanity-check: не противоречит ли назначаемый возраст роли участника.

    Возраст 6 у "водителя Toyota" — явная ошибка matching (возраст из NER
    относился к ребёнку-пассажиру, но мульти-категорийная регулярка
    привязала его к водителю). Отбрасываем.
    """
    if candidate.part_type == "Водитель":
        if is_motorcycle(candidate.vehicle_prod_type):
            return age >= MIN_AGE_DRIVER_MOTO
        return age >= MIN_AGE_DRIVER_CAR
    if candidate.part_type and "Велосипедист" in candidate.part_type:
        return age >= MIN_AGE_CYCLIST
    # Пассажиры, пешеходы — нет нижнего ограничения
    return True


def filter_candidates(candidates: list[Candidate], classification: dict) -> list[Candidate]:
    """
    Фильтрует кандидатов по классификации контекста.

    Пол сравнивается строго (если задан в classification и в кандидате) —
    но если у кандидата sex='Не определен' (значение из источника), мы НЕ
    отсекаем его, т.к. в посте возможен matching по другим признакам.
    Однако: если sex задан явно (М/Ж), а у кандидата явно противоположный —
    исключаем.
    """
    cat = classification["category"]
    target_sex = classification.get("sex")

    def sex_matches(cand_sex: Optional[str]) -> bool:
        """Совместим ли пол кандидата с целевым (None у любой стороны = ок)."""
        if target_sex is None:
            return True
        if cand_sex is None or cand_sex == "Не определен":
            return True  # ослабляем — нет данных в БД
        return cand_sex == target_sex

    result: list[Candidate] = []

    for c in candidates:
        if cat == "driver":
            if c.table == "participants" and c.part_type == "Водитель":
                if sex_matches(c.sex):
                    result.append(c)
        elif cat == "passenger":
            if c.table == "participants" and c.part_type == "Пассажир":
                if sex_matches(c.sex):
                    result.append(c)
        elif cat == "pedestrian":
            if (
                c.table == "accident_pedestrians"
                and is_pedestrian_part_type(c.part_type)
                and sex_matches(c.sex)
            ):
                result.append(c)
        elif cat == "cyclist":
            if c.table == "participants" and c.part_type and ("Велосипедист" in c.part_type):
                if sex_matches(c.sex):
                    result.append(c)
        elif cat == "motorcyclist":
            # Водитель мотоцикла или мопеда
            if (
                c.table == "participants"
                and c.part_type == "Водитель"
                and is_motorcycle(c.vehicle_prod_type)
                and sex_matches(c.sex)
            ):
                result.append(c)
        elif cat == "child_pedestrian":
            # Ребёнок-пешеход с заданным полом
            if (
                c.table == "accident_pedestrians"
                and is_pedestrian_part_type(c.part_type)
                and sex_matches(c.sex)
            ):
                result.append(c)
        elif cat == "child_any":
            # Ребёнок без указания роли — пешеход или пассажир
            if c.table == "accident_pedestrians" and is_pedestrian_part_type(c.part_type):
                if sex_matches(c.sex):
                    result.append(c)
            elif c.table == "participants" and c.part_type == "Пассажир":
                if sex_matches(c.sex):
                    result.append(c)
        elif cat == "any_role":
            # Любая роль с заданным полом (мужчина / женщина / пенсионер[ка])
            if c.table == "participants" and c.part_type in (
                "Водитель",
                "Пассажир",
            ):
                if sex_matches(c.sex):
                    result.append(c)
            elif c.table == "accident_pedestrians" and is_pedestrian_part_type(c.part_type):
                if sex_matches(c.sex):
                    result.append(c)
    return result


# =====================================================================
# Главный pipeline: process_post + enrich
# =====================================================================


@dataclass
class Assignment:
    """Назначение возраста кандидату с аудит-следом."""

    table: str
    row_id: int
    age: int
    context: str
    post_id: int


@dataclass
class EnrichmentStats:
    posts_total: int = 0
    posts_skipped_multi_dtp: int = 0
    posts_skipped_no_accident_data: int = 0
    posts_processed: int = 0

    ages_total: int = 0
    ages_assigned: int = 0
    ages_skipped_no_candidate: int = 0
    ages_skipped_ambiguous: int = 0
    ages_skipped_unknown_context: int = 0
    ages_skipped_age_role_conflict: int = 0
    cross_post_collisions_consistent: int = 0
    cross_post_collisions_dropped: int = 0

    by_category: Counter = field(default_factory=Counter)
    by_table: Counter = field(default_factory=Counter)
    age_distribution: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "posts_total": self.posts_total,
            "posts_skipped_multi_dtp": self.posts_skipped_multi_dtp,
            "posts_skipped_no_accident_data": self.posts_skipped_no_accident_data,
            "posts_processed": self.posts_processed,
            "ages_total": self.ages_total,
            "ages_assigned": self.ages_assigned,
            "ages_skipped_no_candidate": self.ages_skipped_no_candidate,
            "ages_skipped_ambiguous": self.ages_skipped_ambiguous,
            "ages_skipped_unknown_context": self.ages_skipped_unknown_context,
            "ages_skipped_age_role_conflict": self.ages_skipped_age_role_conflict,
            "cross_post_collisions_consistent": self.cross_post_collisions_consistent,
            "cross_post_collisions_dropped": self.cross_post_collisions_dropped,
            "by_category": dict(self.by_category),
            "by_table": dict(self.by_table),
            "age_min": min(self.age_distribution) if self.age_distribution else None,
            "age_max": max(self.age_distribution) if self.age_distribution else None,
            "age_mean": (
                sum(self.age_distribution) / len(self.age_distribution)
                if self.age_distribution
                else None
            ),
        }


def detect_multi_dtp_post(ages: list[dict]) -> bool:
    """
    Эвристика: если в одном посте 2+ возрастов с одинаковой ролью
    (например, два "водитель"), пост описывает разные ДТП.
    Gold-матч связал пост с ОДНИМ из этих ДТП — назначать всех на него
    некорректно. Skip весь пост.

    NB: 2 возраста с разными ролями (водитель + пешеход) — нормально
    (одно ДТП "наезд на пешехода").
    """
    counter = Counter()
    for age in ages:
        cls = classify_context(age.get("context", ""))
        if cls is None:
            continue
        # Группируем по category — driver/pedestrian/passenger/...
        # any_role и child_any не считаем (они "общие")
        cat = cls["category"]
        if cat in ("any_role", "child_any"):
            continue
        # Различаем child_pedestrian с разным полом (мальчик+девочка — нормально)
        sex = cls.get("sex") or ""
        counter[(cat, sex)] += 1
    return any(count >= 2 for count in counter.values())


def process_post(post: PostData, candidates: list[Candidate]) -> tuple[list[Assignment], dict]:
    """
    Обрабатывает один пост: для каждого age классифицирует и назначает
    однозначному кандидату. Уже назначенные исключаются из последующих age.

    Возвращает (assignments, per_age_stats).
    """
    assignments: list[Assignment] = []
    used_keys: set[tuple[str, int]] = set()
    stats = {
        "no_candidate": 0,
        "ambiguous": 0,
        "unknown_context": 0,
        "age_role_conflict": 0,
        "by_category": Counter(),
    }

    for age_entry in post.ages:
        age_value = age_entry.get("age")
        context = age_entry.get("context", "")
        cls = classify_context(context)
        if cls is None:
            stats["unknown_context"] += 1
            continue

        # Исключаем уже назначенных
        free = [c for c in candidates if (c.table, c.row_id) not in used_keys]
        matches = filter_candidates(free, cls)

        if len(matches) == 1:
            target = matches[0]
            # Sanity-check: возраст соответствует роли (не 6-летний водитель Toyota)
            if not validate_assignment(int(age_value), target):
                stats["age_role_conflict"] += 1
                continue
            assignments.append(
                Assignment(
                    table=target.table,
                    row_id=target.row_id,
                    age=int(age_value),
                    context=context,
                    post_id=post.tg_id,
                )
            )
            used_keys.add((target.table, target.row_id))
            stats["by_category"][cls["category"]] += 1
        elif len(matches) == 0:
            stats["no_candidate"] += 1
        else:
            stats["ambiguous"] += 1

    return assignments, stats


def reset_existing_assignments(session: Session, source: str) -> dict:
    """Идемпотентность: обнуляем все ранее назначенные возрасты с этим source."""
    result_p = session.execute(
        text(
            """
            UPDATE participants
            SET age_from_telegram = NULL,
                age_source = NULL,
                age_match_context = NULL,
                age_match_post_id = NULL
            WHERE age_source = :src
            """
        ),
        {"src": source},
    )
    result_ap = session.execute(
        text(
            """
            UPDATE accident_pedestrians
            SET age_from_telegram = NULL,
                age_source = NULL,
                age_match_context = NULL,
                age_match_post_id = NULL
            WHERE age_source = :src
            """
        ),
        {"src": source},
    )
    return {
        "participants_reset": result_p.rowcount,
        "accident_pedestrians_reset": result_ap.rowcount,
    }


def apply_assignments(session: Session, assignments: list[Assignment], source: str) -> None:
    """UPDATE-батч для всех назначений."""
    if not assignments:
        return
    by_table: dict[str, list[Assignment]] = defaultdict(list)
    for a in assignments:
        by_table[a.table].append(a)

    for table_name, items in by_table.items():
        # SQLAlchemy executemany через :параметры
        rows = [
            {
                "row_id": a.row_id,
                "age": a.age,
                "context": a.context[:50],
                "post_id": a.post_id,
                "src": source,
            }
            for a in items
        ]
        session.execute(
            text(
                f"""
                UPDATE {table_name}
                SET age_from_telegram = :age,
                    age_source = :src,
                    age_match_context = :context,
                    age_match_post_id = :post_id
                WHERE id = :row_id
                """
            ),
            rows,
        )


def enrich(
    session: Session,
    matches_path: Path,
    ner_path: Path,
    gold_top_score: int = GOLD_TOP_SCORE,
    source: str = SOURCE_GOLD,
) -> EnrichmentStats:
    """Полный pipeline обогащения. Идемпотентен."""
    stats = EnrichmentStats()

    logger.info(f"Загружаем gold-пары: matches={matches_path.name}, ner={ner_path.name}")
    posts = load_gold_posts(matches_path, ner_path, gold_top_score)
    stats.posts_total = len(posts)
    logger.info(f"  Получили {stats.posts_total} gold-постов с возрастами")

    logger.info(f"Сбрасываем предыдущие назначения с source={source!r}")
    reset_info = reset_existing_assignments(session, source)
    logger.info(f"  Обнулено: {reset_info}")

    all_assignments: list[Assignment] = []

    for post in posts:
        # Защита от мульти-ДТП-постов
        if detect_multi_dtp_post(post.ages):
            stats.posts_skipped_multi_dtp += 1
            continue

        candidates = load_accident_candidates(session, post.accident_id)
        if not candidates:
            stats.posts_skipped_no_accident_data += 1
            continue

        assignments, post_stats = process_post(post, candidates)

        stats.ages_total += len(post.ages)
        stats.ages_assigned += len(assignments)
        stats.ages_skipped_no_candidate += post_stats["no_candidate"]
        stats.ages_skipped_ambiguous += post_stats["ambiguous"]
        stats.ages_skipped_unknown_context += post_stats["unknown_context"]
        stats.ages_skipped_age_role_conflict += post_stats["age_role_conflict"]
        for cat, n in post_stats["by_category"].items():
            stats.by_category[cat] += n

        for a in assignments:
            stats.by_table[a.table] += 1
            stats.age_distribution.append(a.age)

        all_assignments.extend(assignments)
        stats.posts_processed += 1

    # Кросс-постовая дедупликация (precision-first):
    # один и тот же participant/pedestrian может попасть под 2+ разных
    # gold-поста. Если все они согласованы по возрасту — оставляем один.
    # Если конфликтуют — отбрасываем все (лучше не знать чем неправильно).
    deduplicated = _deduplicate_assignments(all_assignments, stats)

    logger.info(
        f"Применяем {len(deduplicated)} назначений в БД (батчем) — "
        f"кросс-постовых коллизий: {stats.cross_post_collisions_consistent} consistent / "
        f"{stats.cross_post_collisions_dropped} dropped"
    )
    apply_assignments(session, deduplicated, source)
    logger.info("  UPDATE завершён.")

    # Пересчитываем итоговые числа после дедупликации
    stats.ages_assigned = len(deduplicated)
    stats.by_table = Counter()
    stats.age_distribution = []
    for a in deduplicated:
        stats.by_table[a.table] += 1
        stats.age_distribution.append(a.age)

    return stats


def _deduplicate_assignments(
    assignments: list[Assignment], stats: EnrichmentStats
) -> list[Assignment]:
    """
    Кросс-постовая дедупликация. Группирует по (table, row_id):
    - 1 assignment        → оставляем
    - 2+ с одинаковым age → оставляем первый (consistent, +counter)
    - 2+ с разным age     → выбрасываем все (collision, +counter)
    """
    grouped: dict[tuple[str, int], list[Assignment]] = defaultdict(list)
    for a in assignments:
        grouped[(a.table, a.row_id)].append(a)

    result: list[Assignment] = []
    for key, items in grouped.items():
        if len(items) == 1:
            result.append(items[0])
            continue
        ages = {it.age for it in items}
        if len(ages) == 1:
            stats.cross_post_collisions_consistent += 1
            result.append(items[0])
        else:
            stats.cross_post_collisions_dropped += 1
    return result
