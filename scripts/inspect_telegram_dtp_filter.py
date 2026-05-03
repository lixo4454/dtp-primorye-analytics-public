# -*- coding: utf-8 -*-
"""
Sanity check фильтра ДТП в постах Telegram-канала @prim_police.

Показывает:
- 10 случайных постов помеченных как ДТП (is_dtp_related=True)
- 10 случайных постов помеченных как НЕ-ДТП (is_dtp_related=False)
- Статистику по совпавшим ключевым словам
- Длину постов в обеих группах

Цель: убедиться что keyword-фильтр отбирает реальные ДТП-посты,
а не профилактические мероприятия или акции.

Запуск:
    python scripts/inspect_telegram_dtp_filter.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONL_DIR = PROJECT_ROOT / "data" / "raw" / "telegram_prim_police"

random.seed(42)  # воспроизводимость


def load_all_posts() -> list[dict]:
    """Читает все JSONL-файлы из telegram_prim_police/."""
    posts = []
    for path in sorted(JSONL_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    posts.append(json.loads(line))
    return posts


def truncate(text: str, max_len: int = 600) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # сжимаем множественные пробелы
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def show_samples(posts: list[dict], label: str, n: int = 10) -> None:
    sample = random.sample(posts, min(n, len(posts)))
    print(f"\n{'='*80}")
    print(f"{label} (показываю {len(sample)} из {len(posts)})")
    print(f"{'='*80}")
    for i, post in enumerate(sample, 1):
        print(f"\n--- [{i}] {post['date_published']} | tg_id={post['tg_id']} ---")
        print(f"keywords matched: {post['matched_keywords']}")
        print(f"text ({post['char_count']} chars, {post['word_count']} words):")
        print(truncate(post["text"]))


def show_keyword_stats(dtp_posts: list[dict]) -> None:
    """Какие ключевые слова срабатывают чаще всего."""
    counter = Counter()
    for post in dtp_posts:
        for kw in post["matched_keywords"]:
            counter[kw] += 1

    print(f"\n{'='*80}")
    print("ЧАСТОТА КЛЮЧЕВЫХ СЛОВ (в ДТП-постах)")
    print(f"{'='*80}")
    for kw, count in counter.most_common():
        bar = "█" * (count * 50 // counter.most_common(1)[0][1])
        print(f"  {kw:30s} {count:5d} {bar}")


def show_length_stats(dtp_posts: list[dict], non_dtp_posts: list[dict]) -> None:
    """Сравнение длин постов в группах."""

    def stats(posts: list[dict]) -> tuple[int, int, int]:
        lens = [p["char_count"] for p in posts]
        return min(lens), sum(lens) // len(lens), max(lens)

    dtp_min, dtp_avg, dtp_max = stats(dtp_posts)
    non_min, non_avg, non_max = stats(non_dtp_posts)

    print(f"\n{'='*80}")
    print("ДЛИНА ПОСТОВ (символов)")
    print(f"{'='*80}")
    print(f"  ДТП-посты:    min={dtp_min:5d}  avg={dtp_avg:5d}  max={dtp_max:5d}")
    print(f"  Не-ДТП:       min={non_min:5d}  avg={non_avg:5d}  max={non_max:5d}")


def main() -> None:
    print("=" * 80)
    print("Sanity check ДТП-фильтра Telegram-постов")
    print("=" * 80)

    posts = load_all_posts()
    print(f"\nЗагружено всего постов: {len(posts)}")

    dtp_posts = [p for p in posts if p["is_dtp_related"]]
    non_dtp_posts = [p for p in posts if not p["is_dtp_related"]]

    print(f"  ДТП-посты:  {len(dtp_posts)} ({len(dtp_posts) / len(posts) * 100:.1f}%)")
    print(f"  Не-ДТП:     {len(non_dtp_posts)} " f"({len(non_dtp_posts) / len(posts) * 100:.1f}%)")

    show_length_stats(dtp_posts, non_dtp_posts)
    show_keyword_stats(dtp_posts)

    show_samples(dtp_posts, "🟢 ДТП-ПОСТЫ", n=10)
    show_samples(non_dtp_posts, "🔴 НЕ-ДТП ПОСТЫ", n=10)


if __name__ == "__main__":
    main()
