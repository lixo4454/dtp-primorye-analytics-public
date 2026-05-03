"""
Router /nlp — BERTopic-темы и Telegram-посты.

Эндпоинты:
- GET /nlp/topics — список 7 тем + шум (-1) с топ-словами и примерами
- GET /nlp/posts/{tg_id} — какая тема у поста + accident-связь (если в gold-парах)

Архитектурное решение (см. предупреждение из промпта3):
- НЕ делаем BERTopic.transform на новом тексте в этом эндпоинте.
  Transform на сохранённой модели без embedding-model в памяти медленный
  (~1с/текст) и требует sentence-transformer на ~500МБ.
- Вместо этого ходим в `topic_assignments.jsonl` — lookup tg_id → topic_id
  (мгновенно). Текст поста — из telegram_post_previews (предзагружено).

Это покрывает 99% бизнес-случая «дай тему этого поста». Для произвольного
текста (классификация на лету) — отдельный эндпоинт `/nlp/classify` появится
в4-15 после аккуратной интеграции sentence-transformer в lifespan.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import ModelRegistry, get_db, get_models
from src.api.schemas import (
    PostTopicMatch,
    SemanticSearchHit,
    SemanticSearchResponse,
    TopicEmTypeLink,
    TopicOut,
    TopicsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nlp", tags=["nlp"])


# Человекочитаемые подписи тем — заполнятся в4 после ручной интерпретации.
# Пока подписи отсутствуют (None), фронт показывает auto-name.
TOPIC_LABELS: dict[int, str] = {
    -1: "Шум / разнотемье",
}


@router.get(
    "/topics",
    response_model=TopicsResponse,
    summary="Список BERTopic-тем (7 + шум) с топ-словами и примерами",
    responses={503: {"description": "topic_summary не загружен"}},
)
async def list_topics(
    models: Annotated[ModelRegistry, Depends(get_models)],
    include_noise: bool = Query(True, description="Включать ли тему -1 (шум)"),
) -> TopicsResponse:
    """Возвращает темы из предсчитанного topic_summary.json."""
    s = models.topic_summary
    if not s:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "topic_summary not loaded")

    raw_topics = s.get("topics", [])
    items: list[TopicOut] = []
    for t in raw_topics:
        topic_id = int(t["topic_id"])
        if topic_id == -1 and not include_noise:
            continue
        em_link = t.get("em_type_link")
        em_link_obj = None
        if em_link:
            em_link_obj = TopicEmTypeLink(
                n_gold_posts=int(em_link.get("n_gold_posts", 0)),
                shares_pct={k: float(v) for k, v in em_link.get("shares_pct", {}).items()},
                dominant_em_type=em_link.get("dominant_em_type"),
                dominant_share_pct=(
                    float(em_link["dominant_share_pct"])
                    if em_link.get("dominant_share_pct") is not None
                    else None
                ),
            )
        items.append(
            TopicOut(
                topic_id=topic_id,
                size=int(t.get("size", 0)),
                share_pct=float(t.get("share_pct", 0.0)),
                name=t.get("name", f"topic_{topic_id}"),
                label=TOPIC_LABELS.get(topic_id),
                top_words=list(t.get("top_words", []))[:10],
                examples=list(t.get("examples", []))[:5],
                em_type_link=em_link_obj,
            )
        )

    return TopicsResponse(
        n_posts=int(s.get("n_posts", 0)),
        n_topics=int(s.get("n_topics", 0)),
        n_noise=int(s.get("n_noise", 0)),
        noise_share_pct=float(s.get("noise_share_pct", 0.0)),
        embedding_model=str(s.get("embedding_model", "")),
        items=items,
    )


@router.get(
    "/search",
    response_model=SemanticSearchResponse,
    summary="Semantic search Telegram-постов через sentence-transformer + cosine",
    responses={503: {"description": "Эмбеддинги или sentence-transformer не загружены"}},
)
async def semantic_search(
    models: Annotated[ModelRegistry, Depends(get_models)],
    q: str = Query(..., min_length=2, description="Текст-запрос для поиска похожих"),
    top_k: int = Query(5, ge=1, le=20, description="Количество результатов"),
) -> SemanticSearchResponse:
    """Cosine similarity между encoded(query) и L2-нормализованными
    эмбеддингами 2 122 Telegram-постов.

    Lazy-загружает sentence-transformer на первое обращение
    (paraphrase-multilingual-MiniLM-L12-v2, ~470 МБ + ~5 сек cold-start).
    Последующие запросы — ~50 мс на encode + ~5 мс на dot-product.
    """
    import time

    if models.telegram_embeddings is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "telegram_embeddings.npz not loaded",
        )

    # Lazy-init модели — на первый /nlp/search в процессе FastAPI.
    if models.sentence_transformer is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Lazy-loading sentence-transformer paraphrase-MiniLM-L12-v2...")
        models.sentence_transformer = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        logger.info("sentence-transformer loaded")

    import numpy as np

    t0 = time.perf_counter()
    qvec = models.sentence_transformer.encode(
        [q], convert_to_numpy=True, normalize_embeddings=True
    )[0].astype(np.float32)

    sims = models.telegram_embeddings @ qvec  # (n,)
    top_idx = np.argsort(-sims)[:top_k]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    items: list[SemanticSearchHit] = []
    for i in top_idx:
        tg_id = int(models.telegram_embedding_tg_ids[i])
        items.append(
            SemanticSearchHit(
                tg_id=tg_id,
                similarity=float(sims[i]),
                topic_id=models.topic_assignments.get(tg_id),
                text_preview=models.telegram_post_previews.get(tg_id),
            )
        )

    return SemanticSearchResponse(
        query=q,
        top_k=top_k,
        elapsed_ms=round(elapsed_ms, 2),
        items=items,
    )


@router.get(
    "/posts/{tg_id}",
    response_model=PostTopicMatch,
    summary="Тема Telegram-поста + accident-связь (если в gold-парах)",
    responses={
        404: {"description": "Пост не найден в topic_assignments"},
    },
)
async def post_topic(
    tg_id: int,
    session: Annotated[AsyncSession, Depends(get_db)],
    models: Annotated[ModelRegistry, Depends(get_models)],
) -> PostTopicMatch:
    """Lookup tg_id → topic_id через предсчитанный topic_assignments.jsonl,
    + если есть в gold-парах (top_score>=75) — возвращаем accident_id и external_id."""
    topic_id = models.topic_assignments.get(tg_id)
    if topic_id is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"tg_id={tg_id} не найден в topic_assignments"
        )

    # Имя темы из summary (если есть)
    topic_name: str | None = None
    for t in models.topic_summary.get("topics", []):
        if int(t.get("topic_id", -999)) == topic_id:
            topic_name = t.get("name")
            break

    # Связь с ДТП через gold-пары (db_id = accident.id)
    matched_accident_id = models.telegram_gold_pairs.get(tg_id)
    matched_external_id: int | None = None
    if matched_accident_id is not None:
        r = await session.execute(
            text("SELECT external_id FROM accidents WHERE id = :id"),
            {"id": matched_accident_id},
        )
        row = r.first()
        if row is not None:
            matched_external_id = int(row.external_id)

    return PostTopicMatch(
        tg_id=tg_id,
        topic_id=int(topic_id),
        topic_name=topic_name,
        topic_label=TOPIC_LABELS.get(int(topic_id)),
        post_text_preview=models.telegram_post_previews.get(tg_id),
        matched_accident_id=matched_accident_id,
        matched_accident_external_id=matched_external_id,
    )
