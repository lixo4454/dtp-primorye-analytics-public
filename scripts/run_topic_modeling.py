# -*- coding: utf-8 -*-
"""
Оркестратор тематического моделирования ДТП-постов.

Что делает:
1. Собирает корпус 2122 ДТП-релевантных постов (data/raw/ + tg_id из NER).
2. Считает sentence-transformers эмбеддинги, кэширует в .npz.
3. Обучает BERTopic, получает 10-15 тем + шумовой кластер -1.
4. Описывает темы (top words, размер, примеры).
5. Связывает темы с em_type через 482 gold-пары.
6. Сохраняет всё на диск + UMAP-проекцию 2D для визуализации.

Зачем нужно:
- Воспроизводимый pipeline для NLP-блока портфолио.
- Артефакты для собеседования — как темы из текста соотносятся с категориями БД.

Запуск:
    python -m scripts.run_topic_modeling
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.nlp.topic_modeling import (
    build_bertopic,
    compute_embeddings,
    describe_topics,
    fit_topic_model,
    link_topics_to_em_type,
    load_gold_pairs,
    prepare_corpus,
    umap_2d_projection,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

EMBEDDINGS_PATH = PROCESSED_DIR / "telegram_embeddings.npz"
ASSIGNMENTS_PATH = PROCESSED_DIR / "topic_assignments.jsonl"
SUMMARY_PATH = PROCESSED_DIR / "topic_summary.json"
UMAP_2D_PATH = PROCESSED_DIR / "telegram_umap_2d.npz"
MODEL_PATH = MODELS_DIR / "bertopic_dtp.pkl"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42
MIN_CLUSTER_SIZE = 30


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Корпус
    logger.info("Шаг 1/6: собираем корпус ДТП-постов")
    t0 = time.time()
    records = prepare_corpus()
    texts = [r.text_clean for r in records]
    tg_ids = [r.tg_id for r in records]
    logger.info(
        f"  → {len(records)} постов, средняя длина {np.mean([len(t) for t in texts]):.0f} симв "
        f"({time.time() - t0:.1f}с)"
    )

    # 2. Эмбеддинги
    logger.info("Шаг 2/6: эмбеддинги через {}", EMBEDDING_MODEL)
    if EMBEDDINGS_PATH.exists():
        cached = np.load(EMBEDDINGS_PATH)
        cached_ids = cached["tg_ids"].tolist()
        if cached_ids == tg_ids:
            embeddings = cached["embeddings"]
            logger.info(
                f"  → загружены кэшированные эмбеддинги {embeddings.shape}, dtype={embeddings.dtype}"
            )
        else:
            logger.warning("  кэш .npz не совпадает по tg_ids — пересчитываем")
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        t0 = time.time()
        embeddings = compute_embeddings(texts, model_name=EMBEDDING_MODEL)
        logger.info(
            f"  → эмбеддинги {embeddings.shape}, dtype={embeddings.dtype} "
            f"({time.time() - t0:.1f}с)"
        )
        np.savez_compressed(
            EMBEDDINGS_PATH,
            embeddings=embeddings.astype(np.float32),
            tg_ids=np.array(tg_ids, dtype=np.int64),
        )
        logger.info(f"  → сохранено в {EMBEDDINGS_PATH.name}")

    # 3. BERTopic
    logger.info(
        "Шаг 3/6: BERTopic (UMAP-10D → HDBSCAN min_cluster_size={}) → c-TF-IDF",
        MIN_CLUSTER_SIZE,
    )
    t0 = time.time()
    topic_model = build_bertopic(
        embedding_model_name=EMBEDDING_MODEL,
        min_cluster_size=MIN_CLUSTER_SIZE,
        random_state=RANDOM_STATE,
    )
    topics, _ = fit_topic_model(topic_model, texts, embeddings)

    n_topics = len(set(topics) - {-1})
    n_noise = int((topics == -1).sum())
    logger.info(
        f"  → {n_topics} тем + шум={n_noise} ({100 * n_noise / len(topics):.1f}%), "
        f"обучение {time.time() - t0:.1f}с"
    )

    # Sanity-check
    if n_topics < 5:
        logger.warning("  ⚠️ всего {} тем — возможно, min_cluster_size слишком велик", n_topics)
    if n_noise / len(topics) > 0.40:
        logger.warning("  ⚠️ шума >40% — возможно, эмбеддинги плохие или кластеры слишком плотные")

    # 4. Описание тем
    logger.info("Шаг 4/6: описание тем (top-10 слов c-TF-IDF + 5 примеров)")
    descriptions = describe_topics(topic_model, texts, topics, k_words=10, k_examples=5)

    # 5. Связь с em_type через gold-пары
    logger.info("Шаг 5/6: связь тем с em_type через 482 gold-пары")
    gold_pairs = load_gold_pairs()
    logger.info(f"  → загружено gold-пар: {len(gold_pairs)}")
    crosstab, em_summary = link_topics_to_em_type(tg_ids, topics, gold_pairs)
    if not crosstab.empty:
        logger.info(f"  → cross-tab {crosstab.shape}, em_types: {list(crosstab.columns)}")

    # Объединяем описание + em_type в финальный summary
    final_summary = {
        "n_posts": len(texts),
        "n_topics": n_topics,
        "n_noise": n_noise,
        "noise_share_pct": round(100 * n_noise / len(texts), 2),
        "embedding_model": EMBEDDING_MODEL,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "random_state": RANDOM_STATE,
        "n_gold_pairs_used": len(gold_pairs),
        "topics": [],
    }
    for desc in descriptions:
        em_link = em_summary.get(desc["topic_id"], {})
        final_summary["topics"].append({**desc, "em_type_link": em_link})

    SUMMARY_PATH.write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  → {SUMMARY_PATH.name}")

    # Сохранить crosstab отдельно (для heatmap-визуализации)
    if not crosstab.empty:
        crosstab.to_json(
            PROCESSED_DIR / "topic_em_type_crosstab.json",
            orient="split",
            force_ascii=False,
        )

    # 6. UMAP 2D + сохранение модели
    logger.info("Шаг 6/6: UMAP 2D-проекция + сохранение модели")
    t0 = time.time()
    umap_2d = umap_2d_projection(embeddings, random_state=RANDOM_STATE)
    np.savez_compressed(
        UMAP_2D_PATH,
        coords=umap_2d.astype(np.float32),
        topics=topics.astype(np.int32),
        tg_ids=np.array(tg_ids, dtype=np.int64),
    )
    logger.info(f"  → UMAP-2D {umap_2d.shape} ({time.time() - t0:.1f}с)")

    # topic_assignments.jsonl
    with ASSIGNMENTS_PATH.open("w", encoding="utf-8") as f:
        for tid, t in zip(tg_ids, topics):
            f.write(json.dumps({"tg_id": int(tid), "topic_id": int(t)}) + "\n")
    logger.info(f"  → {ASSIGNMENTS_PATH.name}")

    # Модель — без embedding_model (тяжёлый)
    topic_model.save(
        str(MODEL_PATH),
        serialization="pickle",
        save_embedding_model=False,
    )
    logger.info(f"  → {MODEL_PATH}")

    # Финальная сводка в консоль
    logger.info("=" * 60)
    logger.info("Темы (отсортированы по размеру):")
    sorted_topics = sorted(
        [d for d in descriptions if d["topic_id"] != -1],
        key=lambda d: -d["size"],
    )
    for d in sorted_topics:
        em_link = em_summary.get(d["topic_id"], {})
        em_str = (
            f" → em_type='{em_link['dominant_em_type']}' "
            f"({em_link['dominant_share_pct']:.0f}% из {em_link['n_gold_posts']} gold)"
            if em_link
            else ""
        )
        logger.info(
            f"  Topic {d['topic_id']:>2} | n={d['size']:>4} ({d['share_pct']:>5.2f}%) "
            f"| {', '.join(d['top_words'][:5])}{em_str}"
        )

    noise_d = next((d for d in descriptions if d["topic_id"] == -1), None)
    if noise_d:
        logger.info(f"  Шум        | n={noise_d['size']:>4} ({noise_d['share_pct']:>5.2f}%)")


if __name__ == "__main__":
    main()
