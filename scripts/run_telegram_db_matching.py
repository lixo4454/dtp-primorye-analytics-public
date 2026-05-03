# -*- coding: utf-8 -*-
"""
Production matching Telegram-NLP с БД ДТП (3 уровня + бонус):
- np (город), street (улица), em_type, severity, vehicle marks, road_km

Score system 0-130. Считаем high-precision matches: score >= 60.

Запуск:
    python -m scripts.run_telegram_db_matching
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from src.analysis.telegram_db_matcher import (
    extract_normalized_locations,
    extract_streets_from_text,
    match_post,
)
from src.database import SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NER_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_ner_results.jsonl"
TELEGRAM_DIR = PROJECT_ROOT / "data" / "raw" / "telegram_prim_police"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_db_matches.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_db_matching_summary.json"

HIGH_PRECISION_THRESHOLD = 60
GOLD_THRESHOLD = 90


def load_ner_results_with_text() -> list[dict]:
    """Загружает NER-результаты + соединяет с текстом постов."""
    # 1. Тексты из JSONL telegram_prim_police
    text_index: dict[int, str] = {}
    for path in sorted(TELEGRAM_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("is_dtp_related"):
                    text_index[rec["tg_id"]] = rec["text"]

    # 2. NER-результаты
    results = []
    with NER_RESULTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["text"] = text_index.get(rec["tg_id"], "")
            results.append(rec)
    return results


def load_candidates_index() -> dict[date, list[dict]]:
    """
    Грузим все ДТП за период + аггрегируем vehicle marks одним JOIN.
    """
    logger.info("Загружаем ДТП с marks из БД (один SQL с JOIN)...")
    index: dict[date, list[dict]] = {}

    with SessionLocal() as s:
        # Один большой запрос: все ДТП + список marks + остальные поля
        rows = s.execute(
            text(
                "SELECT a.id, a.datetime, a.np, a.road_km, a.street, "
                "       a.severity, a.em_type, a.parent_region, "
                "       a.house, a.place, "
                "       COALESCE(array_agg(DISTINCT v.mark) "
                "                FILTER (WHERE v.mark IS NOT NULL), '{}') AS vehicle_marks "
                "FROM accidents a "
                "LEFT JOIN vehicles v ON v.accident_id = a.id "
                "WHERE a.datetime::date BETWEEN '2022-01-01' AND '2026-12-31' "
                "GROUP BY a.id"
            )
        ).all()

        for r in rows:
            d = r.datetime.date() if r.datetime else None
            if d is None:
                continue
            entry = {
                "id": r.id,
                "datetime": r.datetime,
                "np": r.np,
                "road_km": r.road_km,
                "street": r.street,
                "severity": r.severity,
                "em_type": r.em_type,
                "parent_region": r.parent_region,
                "house": r.house,
                "place": r.place,
                "vehicle_marks": list(r.vehicle_marks) if r.vehicle_marks else [],
            }
            index.setdefault(d, []).append(entry)

    total = sum(len(v) for v in index.values())
    logger.info(f"  Индекс собран: {total} ДТП за {len(index)} уникальных дат")
    return index


def get_candidates(
    index: dict[date, list[dict]], post_date: date, days_before: int = 3
) -> list[dict]:
    candidates: list[dict] = []
    for offset in range(0, days_before + 1):
        check_date = post_date - timedelta(days=offset)
        if check_date in index:
            candidates.extend(index[check_date])
    return candidates


def main() -> None:
    logger.info("=" * 70)
    logger.info("Production Telegram → DB matching (5 уровней + scoring)")
    logger.info("=" * 70)

    if not NER_RESULTS_PATH.exists():
        logger.error("Сначала запусти: python -m scripts.run_ner_on_telegram_corpus")
        return

    posts = load_ner_results_with_text()
    logger.info(f"Загружено NER-постов: {len(posts)}")

    db_index = load_candidates_index()

    # Статистика
    posts_no_candidates = 0
    posts_no_match = 0
    posts_low_score = 0  # score < HIGH_PRECISION_THRESHOLD
    posts_high_precision = 0  # score >= HIGH_PRECISION_THRESHOLD
    posts_gold = 0  # score >= GOLD_THRESHOLD
    posts_with_1_top = 0
    score_distribution = Counter()
    breakdown_counter = Counter()

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for post in posts:
            try:
                post_date = datetime.fromisoformat(post["date_published"]).date()
            except (ValueError, TypeError):
                continue

            text_value = post.get("text", "")
            normalized_locs = extract_normalized_locations(post["locations"])
            kilometers = post.get("kilometers", [])
            streets = extract_streets_from_text(text_value)
            brands = [v["brand"] for v in post.get("vehicles", [])]

            candidates = get_candidates(db_index, post_date, days_before=3)
            if not candidates:
                posts_no_candidates += 1
                continue

            matches = match_post(
                post_date=post_date,
                post_text=text_value,
                post_locations=normalized_locs,
                post_kilometers=kilometers,
                post_streets=streets,
                post_brands=brands,
                candidates=candidates,
                score_threshold=30,
            )

            if not matches:
                posts_no_match += 1
                continue

            top_score = matches[0]["score"]
            score_bucket = (top_score // 10) * 10  # 0, 10, 20, ..., 130
            score_distribution[score_bucket] += 1

            if top_score >= GOLD_THRESHOLD:
                posts_gold += 1
            if top_score >= HIGH_PRECISION_THRESHOLD:
                posts_high_precision += 1
            else:
                posts_low_score += 1

            # Сколько кандидатов имеют top score (определяет ambiguity)
            top_count = sum(1 for m in matches if m["score"] == top_score)
            if top_count == 1:
                posts_with_1_top += 1

            # Считаем какие критерии чаще срабатывают
            for m in matches:
                for k in m["score_breakdown"].keys():
                    breakdown_counter[k] += 1

            # Сохраняем (только top-3 для компактности)
            out_record = {
                "tg_id": post["tg_id"],
                "date_published": post["date_published"],
                "normalized_locations": normalized_locs,
                "extracted_streets": streets,
                "extracted_brands": brands,
                "kilometers": kilometers,
                "n_candidates": len(candidates),
                "n_matches": len(matches),
                "top_score": top_score,
                "top_count": top_count,
                "matches": [
                    {
                        "db_id": m["id"],
                        "datetime": m["datetime"].isoformat(),
                        "np": m["np"],
                        "place": m["place"],
                        "street": m["street"],
                        "house": m["house"],
                        "road_km": m["road_km"],
                        "severity": m["severity"],
                        "em_type": m["em_type"],
                        "vehicle_marks": m["vehicle_marks"],
                        "score": m["score"],
                        "score_breakdown": m["score_breakdown"],
                    }
                    for m in matches[:3]
                ],
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False, default=str) + "\n")

    # Сводка
    summary = {
        "total_posts": len(posts),
        "posts_no_candidates": posts_no_candidates,
        "posts_no_match": posts_no_match,
        "posts_low_score": posts_low_score,
        "posts_high_precision": posts_high_precision,
        "posts_gold": posts_gold,
        "posts_with_unique_top": posts_with_1_top,
        "score_distribution": dict(sorted(score_distribution.items())),
        "criterion_usage": dict(breakdown_counter.most_common()),
        "thresholds": {
            "high_precision": HIGH_PRECISION_THRESHOLD,
            "gold": GOLD_THRESHOLD,
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    matched_total = posts_low_score + posts_high_precision
    coverage = matched_total / len(posts) * 100 if posts else 0

    logger.info("\n" + "=" * 70)
    logger.info("ИТОГ")
    logger.info("=" * 70)
    logger.info(f"  Всего постов:                        {len(posts)}")
    logger.info(f"  Без кандидатов в окне:               {posts_no_candidates}")
    logger.info(f"  С кандидатами но без матча:          {posts_no_match}")
    logger.info(
        f"  С low score (<{HIGH_PRECISION_THRESHOLD}):                   " f"{posts_low_score}"
    )
    logger.info(
        f"  ⭐ HIGH PRECISION (≥{HIGH_PRECISION_THRESHOLD}):"
        f"               {posts_high_precision}"
    )
    logger.info(f"  💎 GOLD (≥{GOLD_THRESHOLD}):                       " f"{posts_gold}")
    logger.info(f"  ───")
    logger.info(f"  Уникальный топ-кандидат:             {posts_with_1_top}")
    logger.info(
        f"  ОБЩЕЕ ПОКРЫТИЕ:                      {matched_total}/" f"{len(posts)} ({coverage:.1f}%)"
    )

    logger.info("\n--- Распределение score ---")
    for bucket in sorted(score_distribution.keys()):
        count = score_distribution[bucket]
        bar = "█" * (count * 30 // max(score_distribution.values()))
        logger.info(f"  {bucket:3d}+  {count:5d}  {bar}")

    logger.info("\n--- Срабатывание критериев (всего по кандидатам) ---")
    for criterion, count in breakdown_counter.most_common():
        logger.info(f"  {criterion:20s} {count:6d}")

    logger.success(f"Сохранено: {OUTPUT_PATH}")
    logger.success(f"Сводка: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
