"""
Реестр версий ML-моделей.

Каждый retrain выполняет двух-шаговое обновление:
1. Сохранить новый файл `models/<name>_<timestamp>.<ext>` (immutable snapshot).
2. Атомарно переключить `is_current` в таблице ``model_versions`` И
   скопировать (atomic os.replace на NTFS) файл в "alias" path
   ``models/<name>.<ext>`` — путь, по которому FastAPI грузит модель.

Зачем atomic copy вместо symlink: на Windows symlink требует
admin-прав или Developer Mode, надёжный atomic-replace на NTFS
(Path.replace == os.replace == MoveFileExW с MOVEFILE_REPLACE_EXISTING)
работает без них.

Параллельный безопасный паттерн:
    with model_swap("prophet_dtp", new_path, alias_path,
                    metadata={"mape_pct": 12.5, ...},
                    train_size=3653) as version:
        ...  # дополнительные действия после swap, например POST /admin/reload_models

При исключении внутри блока — alias-копию НЕ откатываем (новый файл
лучше старого по нашему sanity-check'у), но is_current=FALSE в БД,
чтобы health-check мог увидеть рассогласование.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from loguru import logger
from sqlalchemy import update

from src.database import ModelVersion, SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def timestamped_filename(model_name: str, ext: str) -> Path:
    """`prophet_dtp_20260502_154301.pkl` (UTC).

    Минуты и секунды нужны: на быстром CI можно получить два retrain'а
    за минуту (например, при ручном debug). Уникальность имени —
    гарантия idempotency файлов.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return MODELS_DIR / f"{model_name}_{ts}.{ext.lstrip('.')}"


def register_version(
    model_name: str,
    version_path: Path,
    metadata: dict[str, Any],
    train_size: int | None,
    make_current: bool = True,
) -> int:
    """Записывает строку в model_versions. Если make_current=True —
    атомарно UNSET'ит предыдущий is_current и SET'ит для новой.

    Возвращает id новой версии.
    """
    # Если файл лежит внутри PROJECT_ROOT — пишем относительный путь;
    # иначе абсолютный (например, в pytest tmp-директории).
    try:
        rel_path = str(version_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        rel_path = str(version_path).replace("\\", "/")

    with SessionLocal() as s:
        # 1. Снимаем флаг с предыдущего active
        if make_current:
            s.execute(
                update(ModelVersion)
                .where(ModelVersion.model_name == model_name)
                .where(ModelVersion.is_current.is_(True))
                .values(is_current=False)
            )
        # 2. Вставляем новую версию
        new_version = ModelVersion(
            model_name=model_name,
            version_path=rel_path,
            train_size=train_size,
            metadata_json=metadata,
            is_current=make_current,
            trained_at=datetime.utcnow(),
        )
        s.add(new_version)
        s.commit()
        s.refresh(new_version)
        logger.info(
            f"[model_registry] registered {model_name} version_id={new_version.id} "
            f"path={rel_path} is_current={make_current}"
        )
        return new_version.id


def atomic_swap_alias(version_path: Path, alias_path: Path) -> None:
    """Atomic-replace: alias_path ← version_path.

    Сначала копируем в temp в той же директории, потом os.replace —
    это гарантирует atomic без cross-filesystem move. На NTFS os.replace
    == MoveFileExW(MOVEFILE_REPLACE_EXISTING). На POSIX rename(2).
    Если version_path не существует — FileNotFoundError из copy2.
    """
    tmp = alias_path.with_suffix(alias_path.suffix + ".tmp")
    shutil.copy2(version_path, tmp)
    os.replace(tmp, alias_path)
    logger.info(f"[model_registry] alias swapped: {alias_path.name} ← {version_path.name}")


@contextmanager
def model_swap(
    model_name: str,
    version_path: Path,
    alias_path: Path,
    metadata: dict[str, Any],
    train_size: int | None,
) -> Iterator[int]:
    """Контекст: register_version → atomic_swap_alias → yield version_id.

    Если внутри блока возникнет исключение — alias уже подменён
    (модель уже rolled out), но это не критично: health-check заметит
    проблему по error в task_runs.
    """
    version_id = register_version(model_name, version_path, metadata, train_size, make_current=True)
    atomic_swap_alias(version_path, alias_path)
    try:
        yield version_id
    except Exception:
        logger.exception(
            f"model_swap[{model_name}] block failed AFTER swap — "
            "alias переключен, но post-swap action упал"
        )
        raise


def get_current(model_name: str) -> ModelVersion | None:
    """Текущая активная версия модели (для UI/footer)."""
    with SessionLocal() as s:
        from sqlalchemy import select

        return s.execute(
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.is_current.is_(True))
        ).scalar_one_or_none()


def try_reload_api(label: str = "") -> None:
    """Best-effort POST /admin/reload_models после atomic-swap.

    Любая ошибка — warning (swap уже в БД, модель в файле). 5-секундный
    timeout: reload в API — это pickle.load + dict-swap, занимает <1с;
    больше 5с — API лежит, нет смысла блокировать worker.
    """
    import os

    import httpx

    api_url = os.getenv("DTP_API_URL_INTERNAL", "http://api:8000")
    token = os.getenv("ADMIN_RELOAD_TOKEN", "")
    try:
        r = httpx.post(
            f"{api_url}/admin/reload_models",
            headers={"X-Admin-Token": token} if token else {},
            timeout=5.0,
        )
        prefix = f"[{label}] " if label else ""
        logger.info(f"{prefix}/admin/reload_models → {r.status_code} {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"[{label}] reload_models не дёрнулся: {exc}")
