# -*- coding: utf-8 -*-
"""
Загрузчик постов из экспорта Telegram-канала (формат result.json).

Источник: канал @prim_police (Полиция Приморья), ~22000 публичных постов.
Экспорт получается через стандартную функцию Telegram Desktop:
"Экспорт истории чата" → формат JSON, без медиа.

Структура result.json:
{
  "name": "...",
  "type": "public_channel",
  "id": ...,
  "messages": [
    {"id": ..., "type": "message", "date": "...", "text": "..." | [...], ...}
  ]
}

Запуск:
    python -m src.loaders.telegram_export_loader \
        --input "C:/path/to/ChatExport_2026-04-28/result.json"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "telegram_prim_police"
METADATA_PATH = OUTPUT_DIR / "_metadata.json"

# ─── ДТП-маркеры ─────────────────────────────────────────────────────
# v3 — двухуровневые маркеры + NEGATIVE-фильтр (исключения)
#
# STRONG  — явный признак инцидента ДТП. Достаточно одного для попадания.
# WEAK    — общая лексика, для статистики.
# NEGATIVE — если пост ПРОШЁЛ через STRONG, но содержит NEGATIVE-маркер,
#            он ОТБРАСЫВАЕТСЯ (мошенничество, поздравления, некрологи).

STRONG_KEYWORDS = [
    # Глаголы инцидента
    "наезд",
    "наехал",
    "наехала",
    "опрокидыван",
    "опрокинул",
    "опрокинулся",
    "съехал",
    "съехала",
    "съезд с дороги",
    "выехал на встреч",
    "выехала на встреч",
    "вылетел",
    "вылетела",
    "врезал",
    "врезалась",
    # "Столкнов*" в грамматических формах для ДТП
    # (не голое "столкнулся" — оно часто метафорическое)
    "столкновени",
    "допустил столкнов",
    "совершил столкнов",
    "столкнулся с автомоб",
    "столкнулся с автомаш",
    "столкнулся с встречн",
    # Жертвы / последствия
    "получил травмы",
    "получила травмы",
    "получил телесные",
    "травмирован",
    "пострадал в дтп",
    "пострадала в дтп",
    "пострадали в дтп",
    "госпитализирован",
    "скончал",  # скончался, скончалась
    "смертельн",
    "погиб на месте",
    "погибл на месте",
    "погиб в дтп",
    "погибла в дтп",
    "погибли в дтп",
    # Специфические участники
    "мотоциклист",
    "велосипедист",
    "сбил",
    "сбила",
]

WEAK_KEYWORDS = [
    "дтп",
    "дорожно-транспортн",
    "пдд",
    "правил дорожного движени",
    "автомобил",
    "автомашин",
    "водитель",
    "водительниц",
    "пешеход",
    "погиб",
    "погибш",  # без специального контекста — слабые
    "пострадал",
    "пострадала",
]

# Если эти фразы встречаются — отбрасываем пост даже при наличии STRONG.
# Это ловушки русского языка: "столкнулись с мошенниками",
# "погибших при исполнении", "поздравляю с днём".
NEGATIVE_PHRASES = [
    # Мошенничество и кражи (часто в "столкнулись с мошенниками")
    "мошенник",
    "обманом",
    "обманул",
    "обманула",
    "обманули",
    "доверившись",
    "под предлогом",
    "позвонил",
    "позвонила",  # часто стартовое "ей позвонил мошенник"
    # Поздравления / некрологи / служебные мероприятия
    "поздравляю вас",
    "поздравляем вас",
    "при исполнении служебных",
    "при исполнении служебного",
    "погибших сотрудников",
    "погибшим сотрудникам",
    "погибших коллег",
    "погибшим коллегам",
    "ветераны",
    "ветеран",
    "день сотрудника",
    "день полиции",
    "день памяти",
    "минута молчания",
    "светлая память",
    "вечная память",
]

ALL_KEYWORDS = STRONG_KEYWORDS + WEAK_KEYWORDS


# ────────────────────────────────────────────────────────────────────
# Структура данных
# ────────────────────────────────────────────────────────────────────


@dataclass
class TelegramPost:
    """Один пост из экспорта Telegram-канала."""

    tg_id: int  # id поста в канале
    date_published: date  # дата публикации
    text: str  # текст поста (объединённый)
    is_dtp_related: bool = False  # отметка про ДТП по ключевым словам
    matched_keywords: list[str] = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0

    def to_dict(self) -> dict:
        return {
            "tg_id": self.tg_id,
            "date_published": self.date_published.isoformat(),
            "text": self.text,
            "is_dtp_related": self.is_dtp_related,
            "matched_keywords": self.matched_keywords,
            "char_count": self.char_count,
            "word_count": self.word_count,
        }


# ────────────────────────────────────────────────────────────────────
# Извлечение текста из поста
# ────────────────────────────────────────────────────────────────────


def extract_text(message: dict) -> str:
    """
    Telegram хранит текст поста в поле 'text' одним из двух способов:
    - строка: "text": "просто текст"
    - массив сегментов: "text": [{"type": "plain", "text": "..."}, ...]

    Эта функция объединяет их в одну плоскую строку.
    """
    text = message.get("text", "")

    if isinstance(text, str):
        return text.strip()

    if isinstance(text, list):
        parts: list[str] = []
        for segment in text:
            if isinstance(segment, str):
                parts.append(segment)
            elif isinstance(segment, dict):
                # форматированный сегмент: bold, italic, link, mention, hashtag
                seg_text = segment.get("text", "")
                if seg_text:
                    parts.append(seg_text)
        return "".join(parts).strip()

    return ""


# ────────────────────────────────────────────────────────────────────
# ДТП-фильтр по ключевым словам
# ────────────────────────────────────────────────────────────────────


def detect_dtp(text: str) -> tuple[bool, list[str]]:
    """
    v3 трёхуровневый keyword-фильтр.

    Логика:
    1. Если в тексте есть >=1 NEGATIVE-фраза (мошенники, поздравления,
       некрологи) — пост ИСКЛЮЧАЕМ, независимо от STRONG.
    2. Если есть >=1 STRONG-слово — пост ВКЛЮЧАЕМ.
    3. Иначе — НЕ ДТП.

    Возвращает (является_ли_ДТП, список_совпавших_keywords).
    """
    if not text:
        return False, []

    text_lower = text.lower()

    # 1. Проверяем NEGATIVE — это «вето»
    for phrase in NEGATIVE_PHRASES:
        if phrase in text_lower:
            return False, []

    # 2. Считаем STRONG/WEAK совпадения
    strong_matched = [kw for kw in STRONG_KEYWORDS if kw in text_lower]
    weak_matched = [kw for kw in WEAK_KEYWORDS if kw in text_lower]

    is_dtp = len(strong_matched) >= 1
    return is_dtp, strong_matched + weak_matched


# ────────────────────────────────────────────────────────────────────
# Парсинг
# ────────────────────────────────────────────────────────────────────


def parse_export(input_path: Path) -> list[TelegramPost]:
    """Читает result.json и возвращает список TelegramPost."""
    logger.info(f"Читаю экспорт: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    channel_name = data.get("name", "?")
    channel_type = data.get("type", "?")
    raw_messages = data.get("messages", [])

    logger.info(f"Канал: '{channel_name}' (type={channel_type})")
    logger.info(f"Сырых сообщений: {len(raw_messages)}")

    posts: list[TelegramPost] = []
    skipped_no_text = 0
    skipped_service = 0
    skipped_no_date = 0

    for msg in raw_messages:
        # type=='service' — это служебные сообщения (создание канала и т.п.)
        if msg.get("type") != "message":
            skipped_service += 1
            continue

        text = extract_text(msg)
        if not text:
            skipped_no_text += 1
            continue

        date_str = msg.get("date", "")
        if not date_str:
            skipped_no_date += 1
            continue

        try:
            # Формат: "2024-03-15T10:30:00"
            published_at = datetime.fromisoformat(date_str).date()
        except (ValueError, TypeError):
            skipped_no_date += 1
            continue

        is_dtp, matched = detect_dtp(text)

        post = TelegramPost(
            tg_id=msg.get("id", 0),
            date_published=published_at,
            text=text,
            is_dtp_related=is_dtp,
            matched_keywords=matched,
            char_count=len(text),
            word_count=len(text.split()),
        )
        posts.append(post)

    logger.info(f"Распарсено постов с текстом: {len(posts)}")
    logger.info(
        f"Пропущено: service={skipped_service}, "
        f"без текста={skipped_no_text}, без даты={skipped_no_date}"
    )

    return posts


# ────────────────────────────────────────────────────────────────────
# Запись в JSONL по месяцам
# ────────────────────────────────────────────────────────────────────


def write_to_jsonl(posts: list[TelegramPost]) -> dict:
    """Записывает посты в JSONL по месяцам. Возвращает статистику."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_month: dict[str, list[TelegramPost]] = defaultdict(list)
    for p in posts:
        month_key = p.date_published.strftime("%Y-%m")
        by_month[month_key].append(p)

    counts = {}
    dtp_counts = {}
    for month_key, month_posts in sorted(by_month.items()):
        path = OUTPUT_DIR / f"{month_key}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p in month_posts:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        counts[month_key] = len(month_posts)
        dtp_counts[month_key] = sum(1 for p in month_posts if p.is_dtp_related)
        logger.debug(
            f"  {month_key}.jsonl: {len(month_posts)} постов "
            f"(из них ДТП: {dtp_counts[month_key]})"
        )

    return {
        "by_month": counts,
        "dtp_by_month": dtp_counts,
        "total_posts": len(posts),
        "total_dtp_posts": sum(1 for p in posts if p.is_dtp_related),
    }


def write_metadata(stats: dict, posts: list[TelegramPost], source: str) -> None:
    """Сохраняет _metadata.json."""
    earliest = min((p.date_published for p in posts), default=None)
    latest = max((p.date_published for p in posts), default=None)

    metadata = {
        "version": "1.0",
        "source": "Telegram Desktop export",
        "source_file": source,
        "channel": "Полиция Приморья (@prim_police)",
        "imported_at": datetime.now().isoformat(),
        "total_posts": stats["total_posts"],
        "total_dtp_posts": stats["total_dtp_posts"],
        "dtp_ratio": round(stats["total_dtp_posts"] / max(stats["total_posts"], 1) * 100, 2),
        "earliest_post": earliest.isoformat() if earliest else None,
        "latest_post": latest.isoformat() if latest else None,
        "by_month": stats["by_month"],
        "dtp_by_month": stats["dtp_by_month"],
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram export loader")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Путь к result.json от Telegram Desktop",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Файл не найден: {args.input}")
        return

    logger.info("=" * 70)
    logger.info("Telegram export loader (@prim_police)")
    logger.info("=" * 70)

    posts = parse_export(args.input)
    if not posts:
        logger.warning("Нет постов для записи")
        return

    stats = write_to_jsonl(posts)
    write_metadata(stats, posts, str(args.input))

    logger.info("\n" + "=" * 70)
    logger.info("ИТОГ")
    logger.info("=" * 70)
    logger.info(f"  Всего постов: {stats['total_posts']}")
    logger.info(
        f"  Из них про ДТП: {stats['total_dtp_posts']} "
        f"({stats['total_dtp_posts'] / max(stats['total_posts'], 1) * 100:.1f}%)"
    )
    logger.info(f"  Месяцев в выгрузке: {len(stats['by_month'])}")
    logger.info(f"  Файлы: {OUTPUT_DIR}")
    logger.info(f"  Метаданные: {METADATA_PATH}")
    logger.success("Готово.")


if __name__ == "__main__":
    main()
