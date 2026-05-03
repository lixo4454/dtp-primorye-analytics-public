# -*- coding: utf-8 -*-
"""
Визуализация результатов NER-анализа корпуса @prim_police.

Создаёт single-page summary 2x2:
  [1] Динамика ДТП-постов по месяцам (line chart, 50 месяцев)
  [2] Топ-12 марок ТС в постах (bar chart)
  [3] Распределение возрастов участников ДТП (histogram)
  [4] Распределение по часу суток (bar chart, двугорбая кривая)

Сохраняет: data/processed/telegram_ner_overview.png

Запуск:
    python -m scripts.visualize_telegram_ner
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
NER_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_ner_results.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "telegram_ner_overview.png"

# Стилизация для презентации
plt.rcParams["font.family"] = "DejaVu Sans"  # с поддержкой кириллицы
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_ner_results() -> list[dict]:
    """Читает все NER-результаты из JSONL."""
    results = []
    with NER_RESULTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def aggregate_for_plotting(results: list[dict]) -> dict:
    """Агрегирует данные в формат удобный для графиков."""
    # 1. Динамика по месяцам
    posts_by_month = Counter()
    for r in results:
        date_str = r["date_published"]  # YYYY-MM-DD
        month_key = date_str[:7]  # YYYY-MM
        posts_by_month[month_key] += 1

    # 2. Топ марок
    brand_counter = Counter()
    for r in results:
        for v in r["vehicles"]:
            brand_counter[v["brand"]] += 1

    # 3. Возрасты
    ages_list = []
    for r in results:
        for a in r["ages"]:
            ages_list.append(a["age"])

    # 4. Часы суток
    hour_counter = Counter()
    for r in results:
        for t in r["times"]:
            hour_counter[t["hour"]] += 1

    return {
        "posts_by_month": dict(sorted(posts_by_month.items())),
        "top_brands": brand_counter.most_common(12),
        "ages": ages_list,
        "hour_distribution": dict(sorted(hour_counter.items())),
    }


def plot_monthly_dynamics(ax, posts_by_month: dict) -> None:
    """График 1: динамика ДТП-постов по месяцам."""
    months = list(posts_by_month.keys())
    counts = list(posts_by_month.values())
    # Преобразуем строки YYYY-MM в datetime для красивых засечек
    month_dates = [datetime.strptime(m, "%Y-%m") for m in months]

    ax.plot(month_dates, counts, marker="o", linewidth=2, color="#1f77b4", markersize=4)
    ax.fill_between(month_dates, counts, alpha=0.2, color="#1f77b4")
    ax.set_title("Динамика ДТП-постов в @prim_police по месяцам", fontsize=12, fontweight="bold")
    ax.set_ylabel("Постов в месяц")
    ax.set_ylim(bottom=0)

    # Засечки — каждые 6 месяцев
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    # Аннотация — общая статистика
    total = sum(counts)
    avg_per_month = total / len(counts) if counts else 0
    ax.text(
        0.02,
        0.95,
        f"Всего: {total} постов\n"
        f"Период: {len(counts)} мес.\n"
        f"Среднее: {avg_per_month:.0f}/мес.",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def plot_top_brands(ax, top_brands: list[tuple[str, int]]) -> None:
    """График 2: топ марок ТС."""
    brands = [b[0] for b in top_brands]
    counts = [b[1] for b in top_brands]

    # Японцы — отдельным цветом для подчёркивания доминирования
    japanese = {
        "TOYOTA",
        "NISSAN",
        "HONDA",
        "MAZDA",
        "MITSUBISHI",
        "SUBARU",
        "SUZUKI",
        "DAIHATSU",
        "ISUZU",
        "LEXUS",
        "INFINITI",
        "ACURA",
    }
    colors = ["#d62728" if b in japanese else "#7f7f7f" for b in brands]

    bars = ax.barh(brands, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()  # самая частая — наверху
    ax.set_title("Топ-12 марок ТС в постах УМВД", fontsize=12, fontweight="bold")
    ax.set_xlabel("Упоминаний")

    # Цифры на барах
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )

    # Легенда
    from matplotlib.patches import Patch

    legend = [
        Patch(facecolor="#d62728", label="Японские (78%+)"),
        Patch(facecolor="#7f7f7f", label="Прочие"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, frameon=True)


def plot_ages(ax, ages: list[int]) -> None:
    """График 3: распределение возрастов."""
    if not ages:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Возрасты участников ДТП", fontsize=12, fontweight="bold")
        return

    # Возраст 1-100 с ячейками по 5 лет
    bins = list(range(0, 101, 5))
    ax.hist(ages, bins=bins, color="#2ca02c", edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.set_title("Распределение возрастов участников ДТП", fontsize=12, fontweight="bold")
    ax.set_xlabel("Возраст, лет")
    ax.set_ylabel("Упоминаний")
    ax.set_xlim(0, 100)

    # Статистика
    avg = sum(ages) / len(ages)
    median = sorted(ages)[len(ages) // 2]
    ax.axvline(avg, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(
        avg + 1,
        ax.get_ylim()[1] * 0.9,
        f"среднее = {avg:.1f}",
        color="#d62728",
        fontsize=9,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.95,
        f"Всего: {len(ages)} упоминаний\n"
        f"Среднее: {avg:.1f} лет\n"
        f"Медиана: {median} лет\n"
        f"Мин/Макс: {min(ages)}/{max(ages)}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def plot_hour_distribution(ax, hour_distribution: dict) -> None:
    """График 4: распределение по часу суток."""
    hours = list(range(24))
    counts = [hour_distribution.get(h, 0) for h in hours]

    # Цвета по фазам суток
    def hour_color(h: int) -> str:
        if h in (0, 1, 2, 3, 4):
            return "#1f3a5f"  # ночь — тёмный
        elif h in (5, 6, 7, 8, 9, 10):
            return "#f1c40f"  # утро — жёлтый
        elif h in (11, 12, 13, 14, 15, 16, 17):
            return "#3498db"  # день — синий
        else:
            return "#8e44ad"  # вечер — фиолетовый

    colors = [hour_color(h) for h in hours]
    ax.bar(hours, counts, color=colors, edgecolor="white", linewidth=0.5, width=0.85)
    ax.set_title("Распределение ДТП по часу суток", fontsize=12, fontweight="bold")
    ax.set_xlabel("Час")
    ax.set_ylabel("ДТП с указанным временем")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)

    # Аннотация — всего извлечено
    total = sum(counts)
    ax.text(
        0.02,
        0.95,
        f"Всего: {total} временных меток\n" f"(только посты с явным HH:MM)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def main() -> None:
    logger.info("=" * 70)
    logger.info("Визуализация NER-результатов корпуса @prim_police")
    logger.info("=" * 70)

    if not NER_RESULTS_PATH.exists():
        logger.error(f"Файл не найден: {NER_RESULTS_PATH}")
        logger.error("Сначала запусти: python -m scripts.run_ner_on_telegram_corpus")
        return

    results = load_ner_results()
    logger.info(f"Загружено NER-результатов: {len(results)}")

    data = aggregate_for_plotting(results)
    logger.info(
        f"  Месяцев в выгрузке: {len(data['posts_by_month'])}\n"
        f"  Топ-марок: {len(data['top_brands'])}\n"
        f"  Возрастных меток: {len(data['ages'])}\n"
        f"  Временных меток: {sum(data['hour_distribution'].values())}"
    )

    # 4 субплота 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "ДТП в Приморье: NLP-анализ корпуса @prim_police (УМВД)\n"
        f"{len(results)} ДТП-постов · 4+ года · извлечение через Natasha NER",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plot_monthly_dynamics(axes[0, 0], data["posts_by_month"])
    plot_top_brands(axes[0, 1], data["top_brands"])
    plot_ages(axes[1, 0], data["ages"])
    plot_hour_distribution(axes[1, 1], data["hour_distribution"])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    logger.success(f"Сохранено: {OUTPUT_PATH}")
    logger.info(f"Размер файла: {OUTPUT_PATH.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
