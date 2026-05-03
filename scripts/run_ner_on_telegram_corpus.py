# -*- coding: utf-8 -*-
"""
Массовый прогон NER на корпусе Telegram-постов про ДТП.

Читает все JSONL из data/raw/telegram_prim_police/, фильтрует
ДТП-релевантные посты и применяет extract_all() к каждому.

Сохраняет:
- data/processed/telegram_ner_results.jsonl  — результаты NER по каждому посту
- data/processed/telegram_ner_summary.json   — агрегированная статистика

Запуск:
    python -m scripts.run_ner_on_telegram_corpus
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from loguru import logger

from src.nlp.dtp_ner import extract_all

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "telegram_prim_police"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = OUTPUT_DIR / "telegram_ner_results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "telegram_ner_summary.json"


def load_dtp_posts() -> list[dict]:
    """Читает все JSONL и фильтрует только ДТП-посты."""
    posts = []
    for path in sorted(INPUT_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("is_dtp_related"):
                    posts.append(rec)
    return posts


def process_post(post: dict) -> dict:
    """NER + краткий вывод результата."""
    text = post["text"]
    entities = extract_all(text)
    return {
        "tg_id": post["tg_id"],
        "date_published": post["date_published"],
        "text_length": len(text),
        "vehicles": entities["vehicles"],
        "ages": entities["ages"],
        "times": entities["times"],
        "driving_experience_years": entities["driving_experience_years"],
        "kilometers": entities["kilometers"],
        "locations": entities["locations"],
        "orgs": entities["orgs"],
        "persons": entities["persons"],
        "total_entities": entities["stats"]["total_entities"],
    }


def aggregate_stats(results: list[dict]) -> dict:
    """Считает агрегированную статистику по корпусу."""
    brand_counter = Counter()
    age_distribution = Counter()
    age_context_counter = Counter()
    hour_counter = Counter()
    location_counter = Counter()
    org_counter = Counter()
    drivers_with_experience = []

    posts_with_vehicle = 0
    posts_with_age = 0
    posts_with_time = 0
    posts_with_location = 0
    posts_no_entities = 0

    for r in results:
        if r["vehicles"]:
            posts_with_vehicle += 1
            for v in r["vehicles"]:
                brand_counter[v["brand"]] += 1

        if r["ages"]:
            posts_with_age += 1
            for a in r["ages"]:
                age_distribution[a["age"]] += 1
                age_context_counter[a["context"]] += 1

        if r["times"]:
            posts_with_time += 1
            for t in r["times"]:
                hour_counter[t["hour"]] += 1

        if r["locations"]:
            posts_with_location += 1
            for loc in r["locations"]:
                location_counter[loc["text"]] += 1

        for org in r["orgs"]:
            org_counter[org["text"]] += 1

        drivers_with_experience.extend(r["driving_experience_years"])

        if r["total_entities"] == 0:
            posts_no_entities += 1

    return {
        "total_posts": len(results),
        "posts_with_vehicle": posts_with_vehicle,
        "posts_with_age": posts_with_age,
        "posts_with_time": posts_with_time,
        "posts_with_location": posts_with_location,
        "posts_no_entities": posts_no_entities,
        "top_30_brands": brand_counter.most_common(30),
        "age_distribution_top": age_distribution.most_common(50),
        "age_context_top": age_context_counter.most_common(20),
        "hour_distribution": dict(sorted(hour_counter.items())),
        "top_30_locations": location_counter.most_common(30),
        "top_20_orgs": org_counter.most_common(20),
        "driver_experience_count": len(drivers_with_experience),
        "driver_experience_avg": (
            sum(drivers_with_experience) / len(drivers_with_experience)
            if drivers_with_experience
            else 0
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Массовый NER-прогон по корпусу Telegram-постов")
    logger.info("=" * 70)

    posts = load_dtp_posts()
    logger.info(f"Загружено ДТП-постов: {len(posts)}")

    if not posts:
        logger.error("Нет постов. Сначала запусти telegram_export_loader.")
        return

    # Прогрев Natasha (первый прогон долгий)
    logger.info("Инициализация Natasha (~20-30 сек)...")
    _ = extract_all(posts[0]["text"])
    logger.info("Natasha готова, начинаем массовый прогон")

    results = []
    start_time = time.time()
    last_report_time = start_time

    with RESULTS_PATH.open("w", encoding="utf-8") as out_f:
        for i, post in enumerate(posts, 1):
            try:
                result = process_post(post)
                results.append(result)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Ошибка на tg_id={post['tg_id']}: {e}")
                continue

            now = time.time()
            if now - last_report_time > 10 or i == len(posts):
                elapsed = now - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(posts) - i) / rate if rate > 0 else 0
                logger.info(
                    f"  [{i:5d}/{len(posts)}] {rate:.1f} постов/сек, " f"осталось ~{eta:.0f} сек"
                )
                last_report_time = now

    elapsed = time.time() - start_time
    logger.success(
        f"Обработано {len(results)} постов за {elapsed:.1f} сек "
        f"({len(results)/elapsed:.1f} постов/сек)"
    )

    summary = aggregate_stats(results)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Сводка сохранена: {SUMMARY_PATH}")

    logger.info("\n" + "=" * 70)
    logger.info("ИТОГ")
    logger.info("=" * 70)
    logger.info(f"  Всего постов:                {summary['total_posts']}")
    logger.info(f"  С маркой ТС:                 {summary['posts_with_vehicle']}")
    logger.info(f"  С возрастом:                 {summary['posts_with_age']}")
    logger.info(f"  С временем (HH:MM):          {summary['posts_with_time']}")
    logger.info(f"  С геолокацией:               {summary['posts_with_location']}")
    logger.info(f"  Без сущностей:               {summary['posts_no_entities']}")
    logger.info(
        f"  Стажей вождения извлечено:   {summary['driver_experience_count']} "
        f"(средний {summary['driver_experience_avg']:.1f} лет)"
    )

    logger.info("\n--- ТОП-15 марок ТС ---")
    for brand, count in summary["top_30_brands"][:15]:
        logger.info(f"  {brand:20s} {count:5d}")

    logger.info("\n--- ТОП-15 локаций ---")
    for loc, count in summary["top_30_locations"][:15]:
        logger.info(f"  {loc:40s} {count:5d}")

    logger.info("\n--- Распределение по часу суток ---")
    if summary["hour_distribution"]:
        max_count = max(summary["hour_distribution"].values())
        for hour in sorted(summary["hour_distribution"].keys()):
            count = summary["hour_distribution"][hour]
            bar = "█" * (count * 30 // max_count)
            logger.info(f"  {hour:02d}:00  {count:4d}  {bar}")

    logger.success("Готово!")


if __name__ == "__main__":
    main()
