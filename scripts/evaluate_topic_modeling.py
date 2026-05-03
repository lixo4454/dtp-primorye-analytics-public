# -*- coding: utf-8 -*-
"""
Количественная оценка качества тематической модели.

Что делает:
1. Метрики качества на baseline-кластеризации (MiniLM):
   - Silhouette на UMAP-10D и raw 384D эмбеддингах (без шума -1).
   - DBCV (density-based clustering validation) через hdbscan.validity_index.
2. Bootstrap-стабильность тем: 5 запусков на 90%-bootstrap-сэмплах с разными
   seed → попарный Adjusted Rand Index (ARI) между запусками. Высокий ARI
   = темы устойчивы к малым возмущениям выборки.
3. Cross-encoder check: повторный pipeline на multilingual-e5-small (другая
   архитектура), посчитать ARI с MiniLM-кластеризацией. Высокий ARI
   = темы не артефакт конкретной embedding-модели.

Зачем нужно:
- Академическая строгость для Роспатента — числа в JSON вместо «выглядит ОК».
- Cross-validation выводов: модель не переподогнана к одному encoder.

Запуск:
    python -m scripts.evaluate_topic_modeling
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_PATH = PROCESSED_DIR / "telegram_embeddings.npz"
ASSIGNMENTS_PATH = PROCESSED_DIR / "topic_assignments.jsonl"
SUMMARY_PATH = PROCESSED_DIR / "topic_summary.json"
QUALITY_PATH = PROCESSED_DIR / "topic_quality_metrics.json"
EMBEDDINGS_E5_PATH = PROCESSED_DIR / "telegram_embeddings_e5.npz"

E5_MODEL = "intfloat/multilingual-e5-small"
RANDOM_STATE = 42
MIN_CLUSTER_SIZE = 30


def load_minilm_state() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Загружает MiniLM эмбеддинги и topics из baseline-прогона."""
    cached = np.load(EMBEDDINGS_PATH)
    emb = cached["embeddings"]
    tg_ids = cached["tg_ids"].tolist()

    # topics из jsonl
    topic_by_id: dict[int, int] = {}
    with ASSIGNMENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            topic_by_id[r["tg_id"]] = r["topic_id"]
    topics = np.array([topic_by_id[t] for t in tg_ids], dtype=np.int32)
    return emb, topics, tg_ids


def cluster_umap_hdbscan(
    embeddings: np.ndarray,
    seed: int = RANDOM_STATE,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """UMAP-10D + HDBSCAN. Возвращает (reduced_10d, labels)."""
    from hdbscan import HDBSCAN
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    reduced = reducer.fit_transform(embeddings)
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(reduced)
    return reduced, labels


# ────────────────────────────────────────────────────────────────────
# 1. Метрики качества baseline
# ────────────────────────────────────────────────────────────────────


def compute_quality_metrics(
    embeddings_raw: np.ndarray,
    embeddings_10d: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Silhouette на raw и 10D + DBCV. Без точек с label=-1."""
    from hdbscan.validity import validity_index
    from sklearn.metrics import silhouette_score

    mask = labels != -1
    n_total = len(labels)
    n_clustered = int(mask.sum())
    n_noise = int(n_total - n_clustered)

    if n_clustered < 2 or len(set(labels[mask])) < 2:
        return {"error": "Недостаточно кластеров для метрик"}

    sil_10d = float(silhouette_score(embeddings_10d[mask], labels[mask], metric="euclidean"))
    sil_raw_cos = float(silhouette_score(embeddings_raw[mask], labels[mask], metric="cosine"))

    # DBCV — на UMAP-10D (где работал HDBSCAN)
    try:
        dbcv = float(
            validity_index(
                embeddings_10d[mask].astype(np.float64),
                labels[mask],
                metric="euclidean",
            )
        )
    except Exception as e:
        logger.warning(f"DBCV failed: {e}")
        dbcv = None

    return {
        "n_total": n_total,
        "n_clustered": n_clustered,
        "n_noise": n_noise,
        "noise_share_pct": round(100 * n_noise / n_total, 2),
        "silhouette_umap10d_euclidean": round(sil_10d, 4),
        "silhouette_raw384d_cosine": round(sil_raw_cos, 4),
        "dbcv_umap10d": round(dbcv, 4) if dbcv is not None else None,
        "interpretation": {
            "silhouette": (
                "Близко к 0 — кластеры пересекаются, ближе к 1 — компактны и далеки. "
                "На текстовых эмбеддингах после UMAP типично 0.3-0.6."
            ),
            "dbcv": (
                "Density-based: ближе к 1 = плотные хорошо разделённые кластеры; "
                "0 = случайная структура; <0 = плохая кластеризация."
            ),
        },
    }


# ────────────────────────────────────────────────────────────────────
# 2. Bootstrap-стабильность
# ────────────────────────────────────────────────────────────────────


def bootstrap_stability(
    embeddings: np.ndarray,
    seeds: list[int],
    sample_frac: float = 0.9,
) -> dict:
    """Запускаем UMAP+HDBSCAN на 90%-bootstrap-сэмплах с разными seeds.
    Считаем попарный ARI между запусками на их пересечении."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    n = len(embeddings)
    sample_size = int(n * sample_frac)

    runs: list[dict] = []
    for seed in seeds:
        idx = rng.choice(n, size=sample_size, replace=False)
        idx.sort()
        t0 = time.time()
        _, labels = cluster_umap_hdbscan(embeddings[idx], seed=seed)
        dur = time.time() - t0
        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        logger.info(
            f"  seed={seed}: {n_clusters} тем + шум {n_noise} "
            f"({100 * n_noise / sample_size:.1f}%) [{dur:.1f}с]"
        )
        runs.append(
            {
                "seed": int(seed),
                "indices": idx,
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
            }
        )

    # ARI попарно на пересечении индексов
    pairs = []
    for r1, r2 in combinations(runs, 2):
        common = np.intersect1d(r1["indices"], r2["indices"])
        if len(common) < 100:
            continue
        # mapping idx → label для каждого run
        m1 = {int(i): int(l) for i, l in zip(r1["indices"], r1["labels"])}
        m2 = {int(i): int(l) for i, l in zip(r2["indices"], r2["labels"])}
        l1 = np.array([m1[i] for i in common])
        l2 = np.array([m2[i] for i in common])
        ari = adjusted_rand_score(l1, l2)
        # ARI без шума (фильтруем точки, которые шум хотя бы в одном run)
        mask_no_noise = (l1 != -1) & (l2 != -1)
        ari_no_noise = (
            adjusted_rand_score(l1[mask_no_noise], l2[mask_no_noise])
            if mask_no_noise.sum() > 50
            else None
        )
        pairs.append(
            {
                "seeds": [r1["seed"], r2["seed"]],
                "n_common_points": int(len(common)),
                "ari_with_noise": round(float(ari), 4),
                "ari_no_noise": round(float(ari_no_noise), 4) if ari_no_noise is not None else None,
            }
        )

    aris = [p["ari_with_noise"] for p in pairs]
    aris_no_noise = [p["ari_no_noise"] for p in pairs if p["ari_no_noise"] is not None]

    return {
        "n_runs": len(seeds),
        "seeds": seeds,
        "sample_frac": sample_frac,
        "runs_summary": [
            {"seed": r["seed"], "n_clusters": r["n_clusters"], "n_noise": r["n_noise"]}
            for r in runs
        ],
        "pairwise_ari": pairs,
        "ari_with_noise_mean": round(float(np.mean(aris)), 4),
        "ari_with_noise_std": round(float(np.std(aris)), 4),
        "ari_no_noise_mean": round(float(np.mean(aris_no_noise)), 4) if aris_no_noise else None,
        "ari_no_noise_std": round(float(np.std(aris_no_noise)), 4) if aris_no_noise else None,
        "interpretation": (
            "ARI=1 — идеальное совпадение разбиения; ARI=0 — случайное; "
            ">0.5 = темы устойчивы; >0.7 = очень устойчивы; "
            "<0.3 = разбиение не стабильно (артефакт seed)."
        ),
    }


# ────────────────────────────────────────────────────────────────────
# 3. Cross-encoder check (e5-small)
# ────────────────────────────────────────────────────────────────────


def cross_encoder_check(
    texts: list[str],
    minilm_labels: np.ndarray,
) -> dict:
    """Считаем эмбеддинги e5-small, запускаем тот же pipeline,
    сравниваем с MiniLM-кластеризацией через ARI."""
    from sklearn.metrics import adjusted_rand_score

    if EMBEDDINGS_E5_PATH.exists():
        cached = np.load(EMBEDDINGS_E5_PATH)
        e5_emb = cached["embeddings"]
        logger.info(f"  → e5 кэш загружен: {e5_emb.shape}")
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        logger.info(f"  → загружаем e5 ({E5_MODEL})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(E5_MODEL, device=device)
        # e5 требует префикса "passage: " для документов
        prefixed = [f"passage: {t}" for t in texts]
        t0 = time.time()
        e5_emb = model.encode(
            prefixed, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )
        logger.info(f"  → e5 эмбеддинги {e5_emb.shape} ({time.time() - t0:.1f}с)")
        np.savez_compressed(EMBEDDINGS_E5_PATH, embeddings=e5_emb.astype(np.float32))

    t0 = time.time()
    _, e5_labels = cluster_umap_hdbscan(e5_emb, seed=RANDOM_STATE)
    dur = time.time() - t0
    n_clusters = len(set(e5_labels) - {-1})
    n_noise = int((e5_labels == -1).sum())
    logger.info(
        f"  → e5 pipeline: {n_clusters} тем + шум {n_noise} "
        f"({100 * n_noise / len(texts):.1f}%) [{dur:.1f}с]"
    )

    ari_full = float(adjusted_rand_score(minilm_labels, e5_labels))
    mask = (minilm_labels != -1) & (e5_labels != -1)
    ari_no_noise = (
        float(adjusted_rand_score(minilm_labels[mask], e5_labels[mask]))
        if mask.sum() > 100
        else None
    )

    return {
        "encoder_a": "paraphrase-multilingual-MiniLM-L12-v2",
        "encoder_b": E5_MODEL,
        "minilm_n_clusters": int(len(set(minilm_labels) - {-1})),
        "minilm_n_noise": int((minilm_labels == -1).sum()),
        "e5_n_clusters": n_clusters,
        "e5_n_noise": n_noise,
        "ari_with_noise": round(ari_full, 4),
        "ari_no_noise": round(ari_no_noise, 4) if ari_no_noise is not None else None,
        "interpretation": (
            "ARI=1 — оба encoder'а дали идентичное разбиение; "
            ">0.5 — темы не зависят от выбора encoder'а; "
            "<0.3 — модель переподогнана к конкретному encoder'у."
        ),
    }


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main() -> None:
    from src.nlp.topic_modeling import prepare_corpus

    logger.info("Шаг 1/4: загружаем baseline-состояние (MiniLM)")
    embeddings, topics, tg_ids = load_minilm_state()
    logger.info(
        f"  → {len(topics)} постов, {len(set(topics) - {-1})} тем + " f"{(topics == -1).sum()} шум"
    )

    # Нужен UMAP-10D — пересчитаем (быстро) для silhouette/DBCV
    logger.info("  → UMAP-10D на baseline (для silhouette/DBCV)")
    reduced_10d, _ = cluster_umap_hdbscan(embeddings, seed=RANDOM_STATE)
    # ВАЖНО: используем СОХРАНЁННЫЕ topics (из jsonl), не пересчитанные —
    # они привязаны к BERTopic-выводам.

    logger.info("Шаг 2/4: метрики качества (silhouette + DBCV)")
    quality = compute_quality_metrics(embeddings, reduced_10d, topics)
    logger.info(
        f"  → silhouette UMAP-10D = {quality['silhouette_umap10d_euclidean']}, "
        f"silhouette raw cos = {quality['silhouette_raw384d_cosine']}, "
        f"DBCV = {quality['dbcv_umap10d']}"
    )

    logger.info("Шаг 3/4: bootstrap-стабильность (5 seed × 90% sample)")
    seeds = [42, 7, 123, 999, 2024]
    stability = bootstrap_stability(embeddings, seeds=seeds, sample_frac=0.9)
    logger.info(
        f"  → ARI (с шумом) = {stability['ari_with_noise_mean']} ± "
        f"{stability['ari_with_noise_std']}"
    )
    logger.info(
        f"  → ARI (без шума) = {stability['ari_no_noise_mean']} ± "
        f"{stability['ari_no_noise_std']}"
    )

    logger.info("Шаг 4/4: cross-encoder check (e5-small)")
    records = prepare_corpus()
    texts = [r.text_clean for r in records]
    cross = cross_encoder_check(texts, topics)
    logger.info(
        f"  → ARI MiniLM vs e5 (с шумом) = {cross['ari_with_noise']}, "
        f"(без шума) = {cross['ari_no_noise']}"
    )

    final = {
        "baseline_quality": quality,
        "bootstrap_stability": stability,
        "cross_encoder": cross,
    }
    QUALITY_PATH.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  → {QUALITY_PATH.name}")


if __name__ == "__main__":
    main()
