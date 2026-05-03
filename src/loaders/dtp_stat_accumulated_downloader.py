"""
Скачивание единого аккумулированного архива dtp-stat.ru за 2015-2026.

Источник: https://dtp-stat.ru/media/years/2015-2026.json.7z
Размер архива: ~117 МБ
Размер распакованного: ~5.9 ГБ
Записей по всей России: ~1.6 млн
Записей по Приморью: ~29 400

Стратегия:
1. Скачать архив (если ещё нет)
2. Распаковать в data/raw/accumulated/
3. Потоково прочитать через ijson, извлечь только Приморье
4. Сохранить в data/raw/primorye_accumulated_2015_2026.json (~80 МБ)
5. Удалить большие файлы (опционально)
"""

import json
import time
from decimal import Decimal
from pathlib import Path
from urllib.request import urlretrieve

import ijson
import py7zr
from loguru import logger


class DecimalJSONEncoder(json.JSONEncoder):
    """JSON encoder, который умеет сериализовать Decimal как float.

    Нужен потому что ijson возвращает все числа с плавающей точкой как Decimal
    для сохранения точности. Стандартный json.dump на это падает.
    """

    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
ACCUM_DIR = DATA_RAW / "accumulated"

URL_ALL = "https://dtp-stat.ru/media/years/2015-2026.json.7z"
ARCHIVE_PATH = ACCUM_DIR / "all_years_2015_2026.json.7z"
PRIMORYE_OUTPUT = DATA_RAW / "primorye_accumulated_2015_2026.json"


def download_archive(force: bool = False) -> Path:
    """Скачивает архив 2015-2026."""
    ACCUM_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists() and not force:
        size_mb = ARCHIVE_PATH.stat().st_size / 1024 / 1024
        logger.info(f"Архив уже скачан: {size_mb:.1f} МБ")
        return ARCHIVE_PATH

    logger.info(f"Скачиваю {URL_ALL}...")
    t0 = time.time()
    urlretrieve(URL_ALL, ARCHIVE_PATH)
    elapsed = time.time() - t0
    size_mb = ARCHIVE_PATH.stat().st_size / 1024 / 1024
    logger.info(f"Скачано: {size_mb:.1f} МБ за {elapsed:.1f} сек")

    return ARCHIVE_PATH


def extract_archive() -> Path:
    """Распаковывает архив, возвращает путь к большому JSON."""
    logger.info(f"Распаковываю {ARCHIVE_PATH.name}...")
    t0 = time.time()

    with py7zr.SevenZipFile(ARCHIVE_PATH, mode="r") as z:
        names = z.getnames()
        logger.info(f"Файлы в архиве: {names}")
        z.extractall(path=ACCUM_DIR)

    elapsed = time.time() - t0
    logger.info(f"Распаковка за {elapsed:.0f} сек")

    candidates = sorted(ACCUM_DIR.glob("*.json"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise RuntimeError("После распаковки JSON не найден")

    big_json = candidates[0]
    size_gb = big_json.stat().st_size / 1024 / 1024 / 1024
    logger.info(f"Найден файл: {big_json.name}, {size_gb:.2f} ГБ")
    return big_json


def extract_primorye(big_json: Path) -> int:
    """Извлекает записи Приморья потоково через ijson, сохраняет в PRIMORYE_OUTPUT."""
    logger.info(f"Извлекаю Приморье из {big_json.name}...")
    t0 = time.time()

    primorye_records = []
    total = 0

    with open(big_json, "rb") as f:
        for rec in ijson.items(f, "item"):
            total += 1
            regions = rec.get("REGIONS") or []
            if any("Приморский" in r for r in regions):
                primorye_records.append(rec)

            if total % 200000 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  Обработано {total:,} | Приморье: {len(primorye_records):,} ({elapsed:.0f} сек)"
                )

    elapsed = time.time() - t0
    logger.info(
        f"Прочитано {total:,} записей. Приморье: {len(primorye_records):,} (за {elapsed:.0f} сек)"
    )

    logger.info(f"Сохраняю в {PRIMORYE_OUTPUT}...")
    t0 = time.time()
    with open(PRIMORYE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(primorye_records, f, ensure_ascii=False, cls=DecimalJSONEncoder)
    elapsed = time.time() - t0
    size_mb = PRIMORYE_OUTPUT.stat().st_size / 1024 / 1024
    logger.info(f"Сохранено за {elapsed:.0f} сек. Размер: {size_mb:.1f} МБ")

    return len(primorye_records)


def cleanup_large_files() -> None:
    """Удаляет большие файлы после извлечения Приморья."""
    if ACCUM_DIR.exists():
        for f in ACCUM_DIR.glob("*.json"):
            size_gb = f.stat().st_size / 1024 / 1024 / 1024
            f.unlink()
            logger.info(f"Удалён: {f.name} ({size_gb:.2f} ГБ освобождено)")
        logger.info(f"Архив {ARCHIVE_PATH.name} оставлен.")


def prepare_data(force: bool = False) -> Path:
    """Главная функция: скачивает, распаковывает, извлекает Приморье, чистит."""
    if PRIMORYE_OUTPUT.exists() and not force:
        size_mb = PRIMORYE_OUTPUT.stat().st_size / 1024 / 1024
        logger.info(f"Финальный файл уже готов: {PRIMORYE_OUTPUT} ({size_mb:.1f} МБ)")
        return PRIMORYE_OUTPUT

    download_archive(force=force)
    big_json = extract_archive()
    extract_primorye(big_json)
    cleanup_large_files()

    return PRIMORYE_OUTPUT


if __name__ == "__main__":
    path = prepare_data()
    logger.info(f"Готово. Финальный файл: {path}")
