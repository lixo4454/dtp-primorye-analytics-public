# -*- coding: utf-8 -*-
"""
Оркестратор обогащения БД возрастами из Telegram NLP.

Берёт gold-пары пост↔ДТП (top_score >= 90 && top_count == 1, 482 поста)
классифицирует возрастные NER-метки по контексту и назначает
их однозначным кандидатам в БД.

Идемпотентен: повторный запуск обнуляет предыдущие назначения с
age_source='telegram_gold' и записывает заново.

Запуск:
    python -m scripts.enrich_db_with_telegram_ages

Артефакты:
    data/processed/telegram_age_enrichment_summary.json — статистика прогона
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.analysis.telegram_age_enrichment import (
    GOLD_TOP_SCORE,
    SOURCE_GOLD,
    enrich,
)
from src.database import SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_db_matches.jsonl"
NER_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_ner_results.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_age_enrichment_summary.json"


def main() -> None:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(f"Не найдено: {MATCHES_PATH}")
    if not NER_PATH.exists():
        raise FileNotFoundError(f"Не найдено: {NER_PATH}")

    logger.info("=" * 70)
    logger.info("Обогащение БД возрастами через Telegram NLP")
    logger.info("=" * 70)
    logger.info(f"Источник matchа: {MATCHES_PATH}")
    logger.info(f"Источник NER:    {NER_PATH}")
    logger.info(f"Порог gold:      top_score >= {GOLD_TOP_SCORE} && top_count == 1")
    logger.info(f"age_source:      {SOURCE_GOLD!r}")

    session = SessionLocal()
    try:
        stats = enrich(
            session=session,
            matches_path=MATCHES_PATH,
            ner_path=NER_PATH,
            gold_top_score=GOLD_TOP_SCORE,
            source=SOURCE_GOLD,
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    summary = {
        "run_at": datetime.now().isoformat(),
        "source": SOURCE_GOLD,
        "gold_top_score_threshold": GOLD_TOP_SCORE,
        **stats.as_dict(),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("Итоги:")
    logger.info(f"  Постов всего (gold с возрастами): {stats.posts_total}")
    logger.info(f"  Постов skip (мульти-ДТП):          {stats.posts_skipped_multi_dtp}")
    logger.info(f"  Постов skip (нет участников в БД): {stats.posts_skipped_no_accident_data}")
    logger.info(f"  Постов обработано:                  {stats.posts_processed}")
    logger.info("")
    logger.info(f"  Возрастов всего:                    {stats.ages_total}")
    logger.info(f"  Возрастов назначено:                {stats.ages_assigned}")
    logger.info(f"  Возрастов skip (нет кандидата):     {stats.ages_skipped_no_candidate}")
    logger.info(f"  Возрастов skip (неоднозначно):      {stats.ages_skipped_ambiguous}")
    logger.info(f"  Возрастов skip (неизв. контекст):   {stats.ages_skipped_unknown_context}")
    logger.info(f"  Возрастов skip (age/role конфликт): {stats.ages_skipped_age_role_conflict}")
    logger.info(f"  Кросс-пост. коллизий consistent:    {stats.cross_post_collisions_consistent}")
    logger.info(f"  Кросс-пост. коллизий dropped:       {stats.cross_post_collisions_dropped}")
    logger.info("")
    logger.info(f"  По категориям:                      {dict(stats.by_category)}")
    logger.info(f"  По таблицам:                        {dict(stats.by_table)}")
    if stats.age_distribution:
        logger.info(
            f"  Возраст: min={min(stats.age_distribution)}, "
            f"max={max(stats.age_distribution)}, "
            f"mean={sum(stats.age_distribution) / len(stats.age_distribution):.1f}"
        )
    logger.info("")
    logger.info(f"Сводка сохранена: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
