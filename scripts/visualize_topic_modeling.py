# -*- coding: utf-8 -*-
"""
3 PNG-визуализации тематической модели.

Что делает:
- topic_umap.png — UMAP-2D scatter всех 2122 постов, цвет = тема
- topic_summary.png — bar chart размера тем + русские топ-слова
- topic_em_type_heatmap.png — связь topic × em_type через 482 gold-пары

Зачем нужно:
- Артефакты для собеседования и Роспатента (визуализация результатов).
- topic_umap — главная диаграмма для нарратива «темы из текста».
- heatmap — связка двух таксономий (тема vs em_type БД).

Запуск:
    python -m scripts.visualize_topic_modeling
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

UMAP_2D_NPZ = PROCESSED_DIR / "telegram_umap_2d.npz"
SUMMARY_JSON = PROCESSED_DIR / "topic_summary.json"
CROSSTAB_JSON = PROCESSED_DIR / "topic_em_type_crosstab.json"

UMAP_PNG = PROCESSED_DIR / "topic_umap.png"
SUMMARY_PNG = PROCESSED_DIR / "topic_summary.png"
HEATMAP_PNG = PROCESSED_DIR / "topic_em_type_heatmap.png"
QUALITY_PNG = PROCESSED_DIR / "topic_quality.png"
QUALITY_JSON = PROCESSED_DIR / "topic_quality_metrics.json"

# Плотные различимые цвета для 7+1 тем (включая шум серым)
TOPIC_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]


def _topic_label(topic: dict, max_words: int = 4) -> str:
    """T{id} «слово1, слово2, слово3, слово4»"""
    words = ", ".join(topic["top_words"][:max_words])
    return f"T{topic['topic_id']} «{words}»"


def plot_umap(summary: dict) -> None:
    """UMAP-2D scatter, цвет = тема."""
    data = np.load(UMAP_2D_NPZ)
    coords = data["coords"]
    topics = data["topics"]

    fig, ax = plt.subplots(figsize=(13, 9))

    # Шум — серым на фоне
    noise_mask = topics == -1
    ax.scatter(
        coords[noise_mask, 0],
        coords[noise_mask, 1],
        s=8,
        c="#cccccc",
        alpha=0.5,
        label=f"шум (n={noise_mask.sum()})",
        rasterized=True,
    )

    # Темы — поверх
    topic_meta = {t["topic_id"]: t for t in summary["topics"]}
    unique_topics = sorted(set(int(t) for t in topics if t != -1))
    for i, tid in enumerate(unique_topics):
        m = topics == tid
        meta = topic_meta[tid]
        words3 = ", ".join(meta["top_words"][:3])
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            s=14,
            c=TOPIC_PALETTE[i % len(TOPIC_PALETTE)],
            alpha=0.75,
            label=f"T{tid} (n={m.sum()}): {words3}",
            rasterized=True,
        )

    ax.set_title(
        f"UMAP-2D проекция эмбеддингов 2122 ДТП-постов УМВД Приморья\n"
        f"BERTopic: {summary['n_topics']} тем + {summary['noise_share_pct']}% шум "
        f"(min_cluster_size={summary['min_cluster_size']}, "
        f"embedding={summary['embedding_model']})",
        fontsize=11,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(UMAP_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {UMAP_PNG.name}")


def plot_topic_summary(summary: dict) -> None:
    """Bar chart размеров тем + топ-слова на label."""
    topics = sorted(
        [t for t in summary["topics"] if t["topic_id"] != -1],
        key=lambda t: -t["size"],
    )
    noise = next(t for t in summary["topics"] if t["topic_id"] == -1)

    labels = [_topic_label(t, max_words=5) for t in topics]
    sizes = [t["size"] for t in topics]
    shares = [t["share_pct"] for t in topics]

    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(topics) + 2)))
    bars = ax.barh(
        range(len(topics)),
        sizes,
        color=[TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i in range(len(topics))],
        edgecolor="white",
    )
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Количество постов")
    ax.set_title(
        f"Темы BERTopic — размер и топ-слова c-TF-IDF\n"
        f"Корпус: {summary['n_posts']} ДТП-постов, шум: "
        f"{noise['size']} ({noise['share_pct']}%)",
        fontsize=11,
    )
    for i, (bar, n, share) in enumerate(zip(bars, sizes, shares)):
        ax.text(
            bar.get_width() + max(sizes) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{n} ({share:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {SUMMARY_PNG.name}")


def plot_em_type_heatmap(summary: dict) -> None:
    """Heatmap topic × em_type с долями (% от gold-постов темы)."""
    if not CROSSTAB_JSON.exists():
        logger.warning("  ⚠️ {} не найден — пропускаю heatmap", CROSSTAB_JSON.name)
        return

    crosstab = pd.read_json(CROSSTAB_JSON, orient="split")
    # Нормируем по строкам: % внутри каждой темы
    row_sums = crosstab.sum(axis=1).replace(0, 1)
    pct = crosstab.div(row_sums, axis=0) * 100

    # Подписи тем — id + 3 топ-слова
    topic_meta = {t["topic_id"]: t for t in summary["topics"]}
    row_labels = []
    for tid in pct.index:
        meta = topic_meta.get(int(tid), None)
        if meta is None:
            row_labels.append(f"T{tid}")
        else:
            words3 = ", ".join(meta["top_words"][:3])
            row_labels.append(f"T{tid} «{words3}» (n={int(row_sums[tid])})")

    fig, ax = plt.subplots(figsize=(13, max(5, 0.55 * len(pct) + 2)))
    im = ax.imshow(pct.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(pct.columns)))
    ax.set_xticklabels(pct.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pct.index)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.values[i, j]
            if v > 0:
                color = "white" if v > 60 else "black"
                ax.text(
                    j,
                    i,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )

    ax.set_title(
        f"Связь BERTopic-тем с em_type из БД через {summary['n_gold_pairs_used']} gold-пар "
        f"(top_score≥90)\n"
        f"% — доля каждого em_type внутри темы (по gold-постам темы)",
        fontsize=11,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("% от gold-пар темы", fontsize=10)
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {HEATMAP_PNG.name}")


def plot_quality_metrics() -> None:
    """3-панельный summary метрик: baseline + bootstrap stability + cross-encoder."""
    if not QUALITY_JSON.exists():
        logger.warning("  ⚠️ {} не найден — пропускаю quality.png", QUALITY_JSON.name)
        return

    q = json.loads(QUALITY_JSON.read_text(encoding="utf-8"))
    base = q["baseline_quality"]
    boot = q["bootstrap_stability"]
    cross = q["cross_encoder"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Панель 1: baseline silhouette + DBCV
    ax = axes[0]
    metrics = [
        ("Silhouette\nUMAP-10D", base["silhouette_umap10d_euclidean"], "#2ca02c"),
        ("Silhouette\nraw 384D cos", base["silhouette_raw384d_cosine"], "#7f7f7f"),
        ("DBCV\nUMAP-10D", base["dbcv_umap10d"], "#1f77b4"),
    ]
    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]
    bars = ax.bar(names, values, color=colors, edgecolor="white")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(0.3, color="gray", ls="--", lw=0.7, label="0.3 — норма для текстов")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Baseline-качество кластеризации\n(MiniLM, 7 тем + 16% шум)", fontsize=11)
    ax.set_ylabel("значение метрики (выше — лучше)")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Панель 2: bootstrap stability — pairwise ARI heatmap
    ax = axes[1]
    seeds = boot["seeds"]
    n = len(seeds)
    ari_mat = np.full((n, n), np.nan)
    for p in boot["pairwise_ari"]:
        i = seeds.index(p["seeds"][0])
        j = seeds.index(p["seeds"][1])
        ari_mat[i, j] = p["ari_no_noise"]
        ari_mat[j, i] = p["ari_no_noise"]
    np.fill_diagonal(ari_mat, 1.0)

    im = ax.imshow(ari_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        [f"s={s}\n(k={r['n_clusters']})" for s, r in zip(seeds, boot["runs_summary"])], fontsize=9
    )
    ax.set_yticklabels([f"s={s}" for s in seeds], fontsize=9)
    for i in range(n):
        for j in range(n):
            v = ari_mat[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if v < 0.5 else "black",
                    fontweight="bold" if i != j else "normal",
                )
    ax.set_title(
        f"Bootstrap-стабильность (5 seeds × 90% sample)\n"
        f"Pairwise ARI без шума = {boot['ari_no_noise_mean']:.2f} ± {boot['ari_no_noise_std']:.2f}",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ARI")

    # Панель 3: cross-encoder
    ax = axes[2]
    bars_data = [
        ("MiniLM-L12-v2\nbaseline", 7, 16.2, "#2ca02c"),
        (
            "e5-small\ncross-check",
            cross["e5_n_clusters"],
            100 * cross["e5_n_noise"] / 2122,
            "#1f77b4",
        ),
    ]
    labels = [d[0] for d in bars_data]
    n_clusters = [d[1] for d in bars_data]
    noise = [d[2] for d in bars_data]
    colors = [d[3] for d in bars_data]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, n_clusters, w, label="число тем", color=colors, edgecolor="white")
    ax2 = ax.twinx()
    bars2 = ax2.bar(
        x + w / 2,
        noise,
        w,
        label="% шума",
        color=[c + "80" for c in colors],
        edgecolor="white",
        hatch="//",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("число тем", fontsize=10)
    ax2.set_ylabel("% шума (-1)", fontsize=10)
    for b, v in zip(bars1, n_clusters):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.2,
            str(v),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    for b, v in zip(bars2, noise):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.5,
            f"{v:.1f}%",
            ha="center",
            fontsize=9,
        )

    ax.set_title(
        f"Cross-encoder check\n"
        f"ARI MiniLM ↔ e5 (без шума) = {cross['ari_no_noise']:.2f}\n"
        f"(низкий — темы encoder-зависимы)",
        fontsize=11,
    )
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(QUALITY_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {QUALITY_PNG.name}")


def main() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    logger.info("Шаг 1/4: UMAP-2D scatter")
    plot_umap(summary)
    logger.info("Шаг 2/4: bar chart размеров тем")
    plot_topic_summary(summary)
    logger.info("Шаг 3/4: heatmap topic × em_type")
    plot_em_type_heatmap(summary)
    logger.info("Шаг 4/4: 3-панельный quality summary")
    plot_quality_metrics()
    logger.info("Готово.")


if __name__ == "__main__":
    main()
