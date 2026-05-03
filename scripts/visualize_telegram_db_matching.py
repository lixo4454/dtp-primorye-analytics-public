# -*- coding: utf-8 -*-
"""
Визуализация результатов matching Telegram-NLP с БД ДТП.

Создаёт single-page summary 2x2:
  [1] Покрытие по месяцам (gold/high/low/no-match)
  [2] Распределение score (гистограмма)
  [3] Срабатывание критериев scoring (горизонтальный бар)
  [4] Топ городов с gold-matches (горизонтальный бар)

Сохраняет: data/processed/telegram_db_matching_overview.png

Запуск:
    python -m scripts.visualize_telegram_db_matching
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_db_matches.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_db_matching_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_db_matching_overview.png"

HIGH_PRECISION_THRESHOLD = 60
GOLD_THRESHOLD = 90

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_matches() -> list[dict]:
    matches = []
    with MATCHES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                matches.append(json.loads(line))
    return matches


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def aggregate_for_plotting(matches: list[dict], summary: dict) -> dict:
    # 1. Покрытие по месяцам
    by_month = defaultdict(lambda: {"gold": 0, "high": 0, "low": 0})
    for m in matches:
        date_str = m["date_published"]
        month_key = date_str[:7]
        score = m["top_score"]
        if score >= GOLD_THRESHOLD:
            by_month[month_key]["gold"] += 1
        elif score >= HIGH_PRECISION_THRESHOLD:
            by_month[month_key]["high"] += 1
        else:
            by_month[month_key]["low"] += 1

    # 2. Топ городов с gold-matches
    gold_locations = Counter()
    for m in matches:
        if m["top_score"] >= GOLD_THRESHOLD:
            for loc in m.get("normalized_locations", []):
                # Убираем префикс «г »/«с »/«п »
                clean = (
                    loc.replace("г ", "").replace("с ", "").replace("п ", "").replace("пгт ", "")
                )
                gold_locations[clean] += 1

    return {
        "by_month": dict(sorted(by_month.items())),
        "score_distribution": summary.get("score_distribution", {}),
        "criterion_usage": summary.get("criterion_usage", {}),
        "gold_locations": gold_locations.most_common(12),
    }


def plot_monthly_coverage(ax, by_month: dict) -> None:
    """График 1: stacked bar — gold/high/low по месяцам."""
    months = list(by_month.keys())
    if not months:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return

    month_dates = [datetime.strptime(m, "%Y-%m") for m in months]
    gold = [by_month[m]["gold"] for m in months]
    high = [by_month[m]["high"] for m in months]
    low = [by_month[m]["low"] for m in months]

    width = 18
    ax.bar(
        month_dates,
        gold,
        width=width,
        color="#ffc107",
        label=f"Gold (score≥{GOLD_THRESHOLD})",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.bar(
        month_dates,
        high,
        width=width,
        bottom=gold,
        color="#28a745",
        label=f"High (≥{HIGH_PRECISION_THRESHOLD})",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.bar(
        month_dates,
        low,
        width=width,
        bottom=[g + h for g, h in zip(gold, high)],
        color="#6c757d",
        label="Low (<60)",
        edgecolor="white",
        linewidth=0.3,
    )

    ax.set_title("Покрытие matching по месяцам", fontsize=12, fontweight="bold")
    ax.set_ylabel("Постов с матчем")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    total_gold = sum(gold)
    total_high = sum(high)
    total_low = sum(low)
    ax.text(
        0.98,
        0.95,
        f"Gold: {total_gold}\n" f"High: {total_high}\n" f"Low: {total_low}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85),
    )


def plot_score_distribution(ax, score_distribution: dict) -> None:
    """График 2: распределение score-buckets."""
    if not score_distribution:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return

    buckets = sorted([int(k) for k in score_distribution.keys()])
    counts = [
        score_distribution[str(b) if isinstance(list(score_distribution.keys())[0], str) else b]
        for b in buckets
    ]

    # Цвет по уровню качества
    def bucket_color(b: int) -> str:
        if b >= GOLD_THRESHOLD:
            return "#ffc107"  # gold
        elif b >= HIGH_PRECISION_THRESHOLD:
            return "#28a745"  # green
        else:
            return "#6c757d"  # gray

    colors = [bucket_color(b) for b in buckets]
    ax.bar(
        buckets,
        counts,
        width=8,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        align="edge",
    )

    # Линии порогов
    ax.axvline(
        HIGH_PRECISION_THRESHOLD,
        color="#28a745",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.text(
        HIGH_PRECISION_THRESHOLD + 1,
        ax.get_ylim()[1] * 0.9,
        f"high={HIGH_PRECISION_THRESHOLD}",
        color="#28a745",
        fontsize=9,
        fontweight="bold",
    )
    ax.axvline(GOLD_THRESHOLD, color="#ffc107", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(
        GOLD_THRESHOLD + 1,
        ax.get_ylim()[1] * 0.95,
        f"gold={GOLD_THRESHOLD}",
        color="#ff9800",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_title("Распределение score (top match каждого поста)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Score (10-bucket)")
    ax.set_ylabel("Постов")


def plot_criterion_usage(ax, criterion_usage: dict) -> None:
    """График 3: какие критерии чаще срабатывали."""
    if not criterion_usage:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        return

    # Перевод названий на русский
    names = {
        "vehicle_marks": "Совпадение марок ТС",
        "np": "Совпадение города (np)",
        "em_type": "Совпадение типа ДТП",
        "severity": "Совпадение тяжести",
        "road_km": "Совпадение километровки",
        "street": "Совпадение улицы",
    }

    items = sorted(criterion_usage.items(), key=lambda x: x[1], reverse=True)
    labels = [names.get(k, k) for k, _ in items]
    counts = [v for _, v in items]

    # Палитра
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bar_colors = colors[: len(items)]

    bars = ax.barh(labels, counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_title("Срабатывание критериев scoring", fontsize=12, fontweight="bold")
    ax.set_xlabel("Срабатываний (по всем кандидатам)")

    # Цифры на барах
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )


def plot_gold_locations(ax, gold_locations: list[tuple[str, int]]) -> None:
    """График 4: топ городов с gold-matches."""
    if not gold_locations:
        ax.text(0.5, 0.5, "Нет gold-матчей", ha="center", va="center")
        ax.set_title("Топ городов с gold-matches")
        return

    cities = [g[0] for g in gold_locations]
    counts = [g[1] for g in gold_locations]

    bars = ax.barh(cities, counts, color="#ffc107", edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_title("Топ городов с gold-matches (score≥90)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gold-матчей")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )


def main() -> None:
    logger.info("=" * 70)
    logger.info("Визуализация Telegram → DB matching")
    logger.info("=" * 70)

    if not MATCHES_PATH.exists():
        logger.error(
            "Файл matches не найден. " "Запусти: python -m scripts.run_telegram_db_matching"
        )
        return

    matches = load_matches()
    summary = load_summary()
    logger.info(f"Загружено matches: {len(matches)}")
    logger.info(f"  Gold (≥{GOLD_THRESHOLD}):  {summary['posts_gold']}")
    logger.info(f"  High (≥{HIGH_PRECISION_THRESHOLD}):  {summary['posts_high_precision']}")

    data = aggregate_for_plotting(matches, summary)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Production matching: Telegram-NLP ↔ структурная БД ДТП\n"
        f"{summary['posts_high_precision']} high-precision · "
        f"{summary['posts_gold']} gold · scoring 0-130",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plot_monthly_coverage(axes[0, 0], data["by_month"])
    plot_score_distribution(axes[0, 1], data["score_distribution"])
    plot_criterion_usage(axes[1, 0], data["criterion_usage"])
    plot_gold_locations(axes[1, 1], data["gold_locations"])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    logger.success(f"Сохранено: {OUTPUT_PATH}")
    logger.info(f"Размер: {OUTPUT_PATH.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
