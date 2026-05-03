"""
FastAPI dependencies + ModelRegistry для ML-моделей.

Что здесь:
- get_db() — async SQLAlchemy session (через src.database.async_session)
- ModelRegistry — singleton-контейнер всех загруженных моделей и
  предзагруженных артефактов (topic_assignments, hotspots_summary)
- get_models() — dependency, возвращает ModelRegistry из app.state

Singleton-паттерн: модели инициализируются один раз в lifespan
(src.api.main:lifespan) и хранятся в app.state.models. Все эндпоинты
получают тот же экземпляр через Depends(get_models). Это убирает
повторную загрузку моделей и даёт стабильный warmup.

Важно: хранение моделей в app.state, а не в module-global, потому
что pytest TestClient с разными конфигурациями (например, без
ML-моделей) может переопределить регистр через app.dependency_overrides.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.async_session import get_async_session

logger = logging.getLogger(__name__)


# =====================================================================
# Пути артефактов (относительно корня проекта)
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CATBOOST_V2_PATH = MODELS_DIR / "catboost_severity_v2.cbm"
CATBOOST_CALIBRATED_PATH = MODELS_DIR / "catboost_severity_v1_calibrated.pkl"
PROPHET_PATH = MODELS_DIR / "prophet_dtp.pkl"
BERTOPIC_PATH = MODELS_DIR / "bertopic_dtp.pkl"

CATBOOST_FEATURES_PATH = DATA_PROCESSED_DIR / "catboost_features.pkl"
TOPIC_ASSIGNMENTS_PATH = DATA_PROCESSED_DIR / "topic_assignments.jsonl"
TOPIC_SUMMARY_PATH = DATA_PROCESSED_DIR / "topic_summary.json"
TELEGRAM_EMBEDDINGS_PATH = DATA_PROCESSED_DIR / "telegram_embeddings.npz"
HOTSPOTS_SUMMARY_PATH = DATA_PROCESSED_DIR / "hotspots_summary.json"
FORECAST_SUMMARY_PATH = DATA_PROCESSED_DIR / "forecast_summary.json"
TELEGRAM_DB_MATCHES_PATH = DATA_PROCESSED_DIR / "telegram_db_matches.jsonl"
# Сырые помесячные jsonl с текстами постов (источник)
TELEGRAM_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "telegram_prim_police"

# Порог top_score для gold-пар (Дни 5 + 8 — gold = top_score >= 75)
GOLD_PAIR_SCORE_THRESHOLD = 75


# =====================================================================
# ModelRegistry — singleton-контейнер
# =====================================================================


@dataclass
class ModelRegistry:
    """Все загруженные модели и сериализованные артефакты в одном месте.

    Поля Optional, потому что в pytest-сценариях мы можем подгружать
    только подмножество (например, только БД без BERTopic).
    """

    catboost_v2: Any = None  # CatBoostClassifier
    catboost_calibrated: Any = None  # IsotonicCalibratedClassifier
    catboost_feature_columns: list[str] = field(default_factory=list)
    catboost_cat_features: list[str] = field(default_factory=list)

    prophet: Any = None  # prophet.Prophet
    bertopic: Any = None  # BERTopic

    # Предзагруженные артефакты (read-only, общие для всех запросов)
    topic_assignments: dict[int, int] = field(default_factory=dict)  # tg_id → topic_id
    topic_summary: dict[str, Any] = field(default_factory=dict)
    hotspots_summary: dict[str, Any] = field(default_factory=dict)
    forecast_summary: dict[str, Any] = field(default_factory=dict)
    # tg_id → accident.id (gold-пары: top_score >= 75)
    telegram_gold_pairs: dict[int, int] = field(default_factory=dict)
    # tg_id → краткий текст поста (первые 300 символов) для UI
    telegram_post_previews: dict[int, str] = field(default_factory=dict)
    # Эмбеддинги Telegram-постов для semantic search.
    # Формат: (n_posts, 384) float32, L2-нормализованные. Загрузка
    # лениво в lifespan, ~3 МБ. None если артефакт отсутствует.
    telegram_embeddings: Any = None  # np.ndarray
    telegram_embedding_tg_ids: Any = None  # np.ndarray
    # sentence-transformer для encode(query) на лету. Lazy-init —
    # только при первом обращении к /nlp/search, чтобы cold-start
    # FastAPI не платил +5 сек.
    sentence_transformer: Any = None


def load_model_registry() -> ModelRegistry:
    """Создаёт ModelRegistry, загружая все модели и артефакты с диска.

    Вызывается ровно один раз в lifespan приложения. Если файла нет —
    логируется warning, соответствующее поле остаётся None/{} и
    эндпоинты возвращают 503 Service Unavailable.
    """
    reg = ModelRegistry()

    # ---- CatBoost v2 (нативный .cbm) ----
    if CATBOOST_V2_PATH.exists():
        from catboost import CatBoostClassifier

        m = CatBoostClassifier()
        m.load_model(str(CATBOOST_V2_PATH))
        reg.catboost_v2 = m
        logger.info("CatBoost v2 loaded: %d features", len(m.feature_names_))
    else:
        logger.warning("CatBoost v2 not found at %s", CATBOOST_V2_PATH)

    # ---- IsotonicCalibratedClassifier (pickle) ----
    if CATBOOST_CALIBRATED_PATH.exists():
        # Pickle-загрузка требует, чтобы класс IsotonicCalibratedClassifier
        # был импортируем по тому же пути, под которым был сохранён.
        # Он определён в src.ml.severity_classifier — импорт ниже
        # достаточно для unpickle.
        import src.ml.severity_classifier  # noqa: F401  (регистрирует класс)

        with open(CATBOOST_CALIBRATED_PATH, "rb") as f:
            reg.catboost_calibrated = pickle.load(f)
        logger.info("CatBoost calibrated loaded: classes=%s", reg.catboost_calibrated.classes_)
    else:
        logger.warning("CatBoost calibrated not found at %s", CATBOOST_CALIBRATED_PATH)

    # ---- Feature schema (порядок и типы фичей для CatBoost) ----
    if CATBOOST_FEATURES_PATH.exists():
        with open(CATBOOST_FEATURES_PATH, "rb") as f:
            features_pkl = pickle.load(f)
        x_train = features_pkl["X_train"]
        reg.catboost_feature_columns = x_train.columns.tolist()
        reg.catboost_cat_features = list(features_pkl["cat_features"])
        logger.info(
            "Feature schema loaded: %d features (%d categorical)",
            len(reg.catboost_feature_columns),
            len(reg.catboost_cat_features),
        )

    # ---- Prophet (pickle) ----
    if PROPHET_PATH.exists():
        with open(PROPHET_PATH, "rb") as f:
            reg.prophet = pickle.load(f)
        logger.info("Prophet loaded: %s", type(reg.prophet).__name__)
    else:
        logger.warning("Prophet not found at %s", PROPHET_PATH)

    # ---- BERTopic (нативный) ----
    if BERTOPIC_PATH.exists():
        from bertopic import BERTopic

        reg.bertopic = BERTopic.load(str(BERTOPIC_PATH))
        n_topics = len(reg.bertopic.get_topic_info())
        logger.info("BERTopic loaded: %d topics", n_topics)
    else:
        logger.warning("BERTopic not found at %s", BERTOPIC_PATH)

    # ---- topic_assignments.jsonl ----
    if TOPIC_ASSIGNMENTS_PATH.exists():
        with open(TOPIC_ASSIGNMENTS_PATH, "r", encoding="utf-8") as f:
            reg.topic_assignments = {
                (row := json.loads(line))["tg_id"]: row["topic_id"] for line in f if line.strip()
            }
        logger.info("topic_assignments: %d posts", len(reg.topic_assignments))

    # ---- topic_summary.json ----
    if TOPIC_SUMMARY_PATH.exists():
        reg.topic_summary = json.loads(TOPIC_SUMMARY_PATH.read_text(encoding="utf-8"))
        logger.info("topic_summary: %d topics", len(reg.topic_summary.get("topics", [])))

    # ---- hotspots_summary.json ----
    if HOTSPOTS_SUMMARY_PATH.exists():
        reg.hotspots_summary = json.loads(HOTSPOTS_SUMMARY_PATH.read_text(encoding="utf-8"))
        n_clusters = len(reg.hotspots_summary.get("top_clusters", []))
        logger.info("hotspots_summary: %d top clusters", n_clusters)

    # ---- forecast_summary.json ----
    if FORECAST_SUMMARY_PATH.exists():
        reg.forecast_summary = json.loads(FORECAST_SUMMARY_PATH.read_text(encoding="utf-8"))
        logger.info("forecast_summary loaded")

    # ---- telegram_db_matches.jsonl: tg_id → accident.id (берём top match с score >= 75) ----
    if TELEGRAM_DB_MATCHES_PATH.exists():
        with open(TELEGRAM_DB_MATCHES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                tg_id = row.get("tg_id")
                top_score = row.get("top_score", 0)
                matches = row.get("matches") or []
                if tg_id is not None and top_score >= GOLD_PAIR_SCORE_THRESHOLD and matches:
                    db_id = matches[0].get("db_id")
                    if db_id is not None:
                        reg.telegram_gold_pairs[int(tg_id)] = int(db_id)
        logger.info(
            "telegram_gold_pairs (top_score >= %d): %d",
            GOLD_PAIR_SCORE_THRESHOLD,
            len(reg.telegram_gold_pairs),
        )

    # ---- Telegram embeddings (numpy npz) ----
    if TELEGRAM_EMBEDDINGS_PATH.exists():
        import numpy as np

        npz = np.load(TELEGRAM_EMBEDDINGS_PATH)
        emb = npz["embeddings"].astype(np.float32)
        # L2-нормализация: cosine similarity = dot product (быстрее)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        reg.telegram_embeddings = emb / norms
        reg.telegram_embedding_tg_ids = npz["tg_ids"]
        logger.info(
            "telegram_embeddings: %d posts × %d dims",
            reg.telegram_embeddings.shape[0],
            reg.telegram_embeddings.shape[1],
        )

    # ---- Текст постов из raw/telegram_prim_police/*.jsonl (только DTP-related) ----
    if TELEGRAM_RAW_DIR.exists():
        # 2122 поста — приемлемое для in-memory; ~1-2 МБ суммарно
        for jsonl_path in sorted(TELEGRAM_RAW_DIR.glob("*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tg_id = row.get("tg_id")
                    if tg_id is None:
                        continue
                    # ограничиваемся DTP-related постами (~2122 шт.)
                    if not row.get("is_dtp_related", False):
                        continue
                    text = (row.get("text") or "")[:300]
                    reg.telegram_post_previews[int(tg_id)] = text
        logger.info(
            "telegram_post_previews: %d (DTP-related only)", len(reg.telegram_post_previews)
        )

    return reg


# =====================================================================
# FastAPI dependencies
# =====================================================================


async def get_db() -> AsyncIterator[AsyncSession]:
    """Async SQLAlchemy session — обёртка, чтобы не импортировать из database напрямую в routers."""
    async for s in get_async_session():
        yield s


def get_models(request: Request) -> ModelRegistry:
    """Возвращает ModelRegistry, инициализированный в lifespan.

    Если регистр не инициализирован (например, ошибка startup) —
    подсовывает пустой ModelRegistry; эндпоинты возвращают 503,
    проверяя нужное поле через is None.
    """
    reg = getattr(request.app.state, "models", None)
    if reg is None:
        logger.error("ModelRegistry не инициализирован в app.state")
        return ModelRegistry()
    return reg


# Type aliases для удобства (FastAPI понимает Annotated типы автоматически)
DbSession = AsyncSession
Models = ModelRegistry
