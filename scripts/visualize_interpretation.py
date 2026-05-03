"""
Визуализации SHAP, калибровки, Optuna и v1↔v2.

Что делает: рисует 5 PNG:
- shap_summary.png — топ-20 признаков по mean(|SHAP|), per-class разбивка
- shap_dead_beeswarm.png — beeswarm для класса dead (топ-20 признаков),
  цвет — нормализованное значение признака (red=high, blue=low),
  показывает направление влияния (positive shap → больше шанс dead)
- calibration_curves.png — 4 reliability-diagram (по классам), до и после
  isotonic-калибровки + ECE в подписи
- optuna_history.png — best ROC-AUC dead vs trial number + per-trial AUC
- metrics_v1_vs_v2.png — bar comparison ключевых метрик

Зачем нужно: SHAP-картинки — главные интерпретируемые артефакты для
собеседования («что повышает шанс смертельного ДТП»). Calibration
показывает что вероятности модели можно подавать диспетчеру 112 без
поправок. Optuna/v1-v2 — доказательство что тюнинг дал бизнес-эффект.

Использование:
    python -m scripts.visualize_interpretation
        [--no-shap] [--no-calibration] [--no-optuna] [--no-v1v2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("viz")

CLASS_ORDER_CANON = ["light", "severe", "severe_multiple", "dead"]
CLASS_COLORS = {
    "light": "#4CAF50",
    "severe": "#FFC107",
    "severe_multiple": "#FF5722",
    "dead": "#B71C1C",
}


# =====================================================================
# 1. SHAP summary (per-class stacked bar)
# =====================================================================
def plot_shap_summary(out_path: Path):
    npz_path = DATA_DIR / "catboost_shap_values.npz"
    if not npz_path.exists():
        logger.warning("Нет %s — пропускаю", npz_path)
        return
    data = np.load(npz_path, allow_pickle=False)
    shap_values = data["shap_values"]  # (n_samples, n_classes, n_features)
    classes = list(data["classes"])
    feature_names = list(data["feature_names"])

    # mean(|SHAP|) per class per feature
    abs_mean = np.mean(np.abs(shap_values), axis=0)  # (n_classes, n_features)
    total = abs_mean.sum(axis=0)
    order = np.argsort(-total)[:20]
    feats_top = [feature_names[i] for i in order]
    abs_mean_top = abs_mean[:, order]  # (n_classes, 20)

    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(feats_top))
    left = np.zeros(len(feats_top))
    for k, cls in enumerate(classes):
        color = CLASS_COLORS.get(cls, "#888888")
        vals = abs_mean_top[k]
        ax.barh(y_pos, vals, left=left, color=color, edgecolor="white", label=cls, height=0.75)
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats_top, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|) — стэк по классам", fontsize=11)
    ax.set_title(
        "Топ-20 признаков по SHAP-важности (per-class разбивка)", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10, title="severity")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (%.0f КБ)", out_path, out_path.stat().st_size / 1024)


# =====================================================================
# 2. SHAP beeswarm для класса dead
# =====================================================================
def plot_shap_dead_beeswarm(out_path: Path, X_test_pickle: Path):
    npz_path = DATA_DIR / "catboost_shap_values.npz"
    if not npz_path.exists():
        logger.warning("Нет %s — пропускаю", npz_path)
        return
    import pickle

    with open(X_test_pickle, "rb") as f:
        d = pickle.load(f)
    X_test = d["X_test"]

    data = np.load(npz_path, allow_pickle=False)
    shap_values = data["shap_values"]
    classes = list(data["classes"])
    feature_names = list(data["feature_names"])
    dead_idx = classes.index("dead")
    shap_dead = shap_values[:, dead_idx, :]  # (n_samples, n_features)

    # Top-20 features для dead (mean abs)
    importance = np.mean(np.abs(shap_dead), axis=0)
    order = np.argsort(-importance)[:20]
    feats_top = [feature_names[i] for i in order]
    shap_top = shap_dead[:, order]  # (n_samples, 20)

    # Feature values (для цвета: нормализуем 0..1).
    # Категориальные → label-encode по частоте; числовые → quantile-rank.
    feat_norm = np.zeros_like(shap_top, dtype=float)
    for j, fname in enumerate(feats_top):
        col = X_test[fname]
        if col.dtype == "object" or col.dtype == "bool":
            # label-encoding по частоте (rare → 0, common → 1)
            codes, _ = col.astype(str).factorize()
            vals = codes.astype(float)
            if np.ptp(vals) == 0:
                feat_norm[:, j] = 0.5
            else:
                feat_norm[:, j] = (vals - vals.min()) / (vals.max() - vals.min())
        else:
            # quantile rank — устойчиво к выбросам
            vals = col.values.astype(float)
            valid_mask = ~np.isnan(vals)
            ranks = np.full(len(vals), np.nan)
            if valid_mask.sum() > 1:
                v = vals[valid_mask]
                # rankdata
                from scipy.stats import rankdata

                r = rankdata(v) / len(v)
                ranks[valid_mask] = r
            ranks[~valid_mask] = 0.5
            feat_norm[:, j] = ranks

    fig, ax = plt.subplots(figsize=(11, 8))
    rng = np.random.default_rng(42)
    n_samples = shap_top.shape[0]

    # Сэмплируем для скорости / читаемости (макс 2000 точек на feature)
    if n_samples > 2000:
        sample_idx = rng.choice(n_samples, 2000, replace=False)
    else:
        sample_idx = np.arange(n_samples)

    cmap = plt.get_cmap("RdBu_r")
    for j, fname in enumerate(feats_top):
        y = j + (rng.random(len(sample_idx)) - 0.5) * 0.6  # jitter
        x = shap_top[sample_idx, j]
        c = feat_norm[sample_idx, j]
        ax.scatter(x, y, c=c, cmap=cmap, alpha=0.55, s=8, vmin=0, vmax=1, edgecolors="none")

    ax.set_yticks(range(len(feats_top)))
    ax.set_yticklabels(feats_top, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("SHAP value для класса 'dead' (positive → ↑ шанс смертельного ДТП)", fontsize=11)
    ax.set_title(
        "Beeswarm SHAP для класса 'dead' — топ-20 признаков\n"
        "Цвет: красный = высокое значение признака, синий = низкое",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("Значение признака (norm)", fontsize=9)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["низкое", "среднее", "высокое"])
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (%.0f КБ)", out_path, out_path.stat().st_size / 1024)


# =====================================================================
# 3. Calibration curves (4 классов, до и после)
# =====================================================================
def plot_calibration_curves(out_path: Path):
    cm_path = DATA_DIR / "catboost_calibration_metrics.json"
    if not cm_path.exists():
        logger.warning("Нет %s — пропускаю", cm_path)
        return
    cm = json.loads(cm_path.read_text())
    rel = cm["reliability_curves"]
    ece_b = cm["ece_before"]
    ece_a = cm["ece_after"]
    classes = cm["classes_order"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    for idx, cls in enumerate(classes):
        ax = axes[idx]
        d = rel[cls]
        # before
        mp_b = [v for v in d["before"]["mean_pred"] if v is not None]
        fp_b = [
            d["before"]["frac_pos"][i]
            for i, v in enumerate(d["before"]["mean_pred"])
            if v is not None
        ]
        # after
        mp_a = [v for v in d["after"]["mean_pred"] if v is not None]
        fp_a = [
            d["after"]["frac_pos"][i]
            for i, v in enumerate(d["after"]["mean_pred"])
            if v is not None
        ]

        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="идеальная")
        ax.plot(
            mp_b,
            fp_b,
            "o-",
            color="#1976D2",
            linewidth=2,
            markersize=7,
            label=f"до калибровки  (ECE={ece_b[cls]:.3f})",
        )
        ax.plot(
            mp_a,
            fp_a,
            "s-",
            color="#D32F2F",
            linewidth=2,
            markersize=7,
            label=f"после isotonic   (ECE={ece_a[cls]:.3f})",
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Средняя предсказанная вероятность")
        ax.set_ylabel("Реальная частота положительных")
        ax.set_title(
            f"Класс: {cls}", fontsize=12, fontweight="bold", color=CLASS_COLORS.get(cls, "black")
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Reliability diagrams: до и после isotonic-калибровки  "
        f"(holdout {cm['calib_holdout_size']} ДТП, eval на test {cm['test_size']})",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (%.0f КБ)", out_path, out_path.stat().st_size / 1024)


# =====================================================================
# 4. Optuna history
# =====================================================================
def plot_optuna_history(out_path: Path):
    h_path = DATA_DIR / "catboost_optuna_history.json"
    if not h_path.exists():
        logger.warning("Нет %s — пропускаю Optuna-визуализацию", h_path)
        return
    history = json.loads(h_path.read_text())
    if not history:
        logger.warning("Optuna history пустая — пропускаю")
        return

    trials = [h["trial"] for h in history]
    aucs = [h["mean_auc"] for h in history]
    best_so_far = np.maximum.accumulate(aucs)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(
        trials,
        aucs,
        alpha=0.55,
        s=40,
        color="#1976D2",
        label="ROC-AUC dead этого trial (CV mean)",
        edgecolors="white",
    )
    ax.plot(trials, best_so_far, "-", color="#D32F2F", linewidth=2.5, label="best-so-far")
    ax.axhline(
        0.734,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="baseline v1 (test ROC-AUC dead = 0.734)",
    )
    ax.axhline(0.78, color="green", linestyle=":", linewidth=1, label="цель ROC-AUC ≥0.78")

    best_trial = int(np.argmax(aucs))
    best_auc = aucs[best_trial]
    ax.scatter(
        [trials[best_trial]],
        [best_auc],
        s=200,
        marker="*",
        color="gold",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        label=f"лучший trial #{trials[best_trial]} (AUC={best_auc:.4f})",
    )

    ax.set_xlabel("Trial number", fontsize=11)
    ax.set_ylabel("ROC-AUC класса 'dead' (3-fold CV)", fontsize=11)
    ax.set_title(
        f"Optuna optimization history — {len(trials)} trials × 3-fold CV\n"
        f"Метрика: ROC-AUC класса dead (бизнес-метрика для диспетчера 112)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (%.0f КБ)", out_path, out_path.stat().st_size / 1024)


# =====================================================================
# 5. v1 vs v2
# =====================================================================
def plot_v1_vs_v2(out_path: Path):
    v_path = DATA_DIR / "catboost_v1_vs_v2.json"
    if not v_path.exists():
        logger.warning("Нет %s — пропускаю v1↔v2-визуализацию", v_path)
        return
    cmp = json.loads(v_path.read_text())
    v1 = cmp["v1"]
    v2 = cmp["v2"]

    # 4 группы метрик: F1-macro, F1-weighted, accuracy, + ROC-AUC per class (4)
    rows = [
        ("F1-macro", v1["f1_macro"], v2["f1_macro"]),
        ("F1-weighted", v1["f1_weighted"], v2["f1_weighted"]),
        ("Accuracy", v1["accuracy"], v2["accuracy"]),
        ("ROC-AUC dead", v1["roc_auc_per_class"].get("dead"), v2["roc_auc_per_class"].get("dead")),
        (
            "ROC-AUC light",
            v1["roc_auc_per_class"].get("light"),
            v2["roc_auc_per_class"].get("light"),
        ),
        (
            "ROC-AUC severe",
            v1["roc_auc_per_class"].get("severe"),
            v2["roc_auc_per_class"].get("severe"),
        ),
        (
            "ROC-AUC sv_mlt",
            v1["roc_auc_per_class"].get("severe_multiple"),
            v2["roc_auc_per_class"].get("severe_multiple"),
        ),
        ("F1 dead", v1["per_class"].get("dead"), v2["per_class"].get("dead")),
        (
            "F1 sv_mlt",
            v1["per_class"].get("severe_multiple"),
            v2["per_class"].get("severe_multiple"),
        ),
    ]
    labels = [r[0] for r in rows]
    v1_vals = [r[1] for r in rows]
    v2_vals = [r[2] for r in rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(
        x - width / 2,
        v1_vals,
        width,
        label="v1 (baseline)",
        color="#90A4AE",
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        v2_vals,
        width,
        label="v2 (Optuna-tuned)",
        color="#1976D2",
        edgecolor="white",
    )

    # подписи Δ
    for i, (l, v1v, v2v) in enumerate(zip(labels, v1_vals, v2_vals)):
        if v1v is None or v2v is None:
            continue
        delta = v2v - v1v
        sign = "+" if delta >= 0 else ""
        ax.text(
            i,
            max(v1v, v2v) + 0.012,
            f"Δ {sign}{delta:.3f}",
            ha="center",
            fontsize=9,
            color="green" if delta > 0 else "red",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("Значение метрики", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Сравнение метрик: baseline → Optuna-tuned v2\n"
        f"ΔF1-macro = {cmp['delta']['f1_macro']:+.4f},  "
        f"ΔROC-AUC dead = {cmp['delta']['roc_auc_dead']:+.4f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (%.0f КБ)", out_path, out_path.stat().st_size / 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--no-optuna", action="store_true")
    ap.add_argument("--no-v1v2", action="store_true")
    args = ap.parse_args()

    if not args.no_shap:
        plot_shap_summary(DATA_DIR / "shap_summary.png")
        plot_shap_dead_beeswarm(
            DATA_DIR / "shap_dead_beeswarm.png",
            DATA_DIR / "catboost_features.pkl",
        )
    if not args.no_calibration:
        plot_calibration_curves(DATA_DIR / "calibration_curves.png")
    if not args.no_optuna:
        plot_optuna_history(DATA_DIR / "optuna_history.png")
    if not args.no_v1v2:
        plot_v1_vs_v2(DATA_DIR / "metrics_v1_vs_v2.png")
    logger.info("Visualizations done.")


if __name__ == "__main__":
    main()
