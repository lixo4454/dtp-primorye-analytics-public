"""
Регистрация уже существующих ML-моделей в `model_versions`.

Ранее модели жили просто файлами в `models/`. Теперь
у нас есть таблица `model_versions` для версионирования. Этот скрипт
регистрирует текущие активные файлы как первые is_current версии,
чтобы UI footer (читающий model_versions) сразу показал их даты.

Запуск:
    python -m scripts.seed_model_versions

Идемпотентно: если is_current версия уже есть для этой модели —
ничего не делает.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database import ModelVersion, SessionLocal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_model_versions")


def file_meta(path: Path) -> dict:
    """trained_at = mtime файла; train_size попробуем взять из соседнего .json."""
    if not path.exists():
        return {}
    return {
        "trained_at": datetime.utcfromtimestamp(path.stat().st_mtime),
        "size_kb": round(path.stat().st_size / 1024, 1),
    }


def register_if_absent(model_name: str, file_path: Path, sidecar_json: Path | None = None) -> bool:
    if not file_path.exists():
        logger.warning(f"[skip] {file_path} not found")
        return False

    with SessionLocal() as s:
        from sqlalchemy import select

        existing = s.execute(
            select(ModelVersion).where(
                ModelVersion.model_name == model_name,
                ModelVersion.is_current.is_(True),
            )
        ).scalar_one_or_none()
        if existing is not None:
            logger.info(f"[skip] {model_name} already registered: id={existing.id}")
            return False

        meta = file_meta(file_path)
        sidecar_data = {}
        if sidecar_json and sidecar_json.exists():
            try:
                sidecar_data = json.loads(sidecar_json.read_text(encoding="utf-8"))
            except Exception:
                pass

        rel_path = str(file_path.relative_to(ROOT)).replace("\\", "/")
        train_size = sidecar_data.get("train_size") or sidecar_data.get("n_train")
        version = ModelVersion(
            model_name=model_name,
            version_path=rel_path,
            trained_at=meta.get("trained_at", datetime.utcnow()),
            train_size=train_size,
            metadata_json={
                "seeded_from": "scripts.seed_model_versions",
                "size_kb": meta.get("size_kb"),
                "sidecar": sidecar_data if sidecar_data else None,
            },
            is_current=True,
        )
        s.add(version)
        s.commit()
        s.refresh(version)
        logger.info(f"[ok] registered {model_name} id={version.id} trained_at={version.trained_at}")
        return True


def main() -> None:
    models_dir = ROOT / "models"

    items = [
        (
            "prophet_dtp",
            models_dir / "prophet_dtp.pkl",
            ROOT / "data" / "processed" / "forecast_summary.json",
        ),
        (
            "catboost_severity_v2",
            models_dir / "catboost_severity_v2.cbm",
            models_dir / "catboost_severity_v2.json",
        ),
        (
            "catboost_severity_v1_calibrated",
            models_dir / "catboost_severity_v1_calibrated.pkl",
            None,
        ),
        (
            "bertopic_dtp",
            models_dir / "bertopic_dtp.pkl",
            ROOT / "data" / "processed" / "topic_summary.json",
        ),
    ]

    n_added = 0
    for name, path, sidecar in items:
        if register_if_absent(name, path, sidecar):
            n_added += 1
    logger.info(f"Готово, зарегистрировано: {n_added}")


if __name__ == "__main__":
    main()
