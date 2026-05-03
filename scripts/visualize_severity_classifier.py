"""
Визуализации baseline-модели CatBoost.

Что строит:
1. confusion_matrix.png   — heatmap, нормализованный по true-классам
2. roc_curves.png         — One-vs-rest ROC для всех 4 классов
3. pr_curves.png          — One-vs-rest Precision-Recall
4. feature_importance_baseline.png — top-20 признаков

Использует датасет (X_test, y_test) из data/processed/catboost_features.pkl
и модель из models/catboost_severity.cbm. Идемпотентно — перезапуск
перезаписывает PNG.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.severity_classifier import SEVERITY_CLASSES, load_model

CLASS_COLORS = {
    "light": "#3498db",
    "severe": "#f39c12",
    "severe_multiple": "#9b59b6",
    "dead": "#e74c3c",
}


def _load_data():
    feat_path = PROJECT_ROOT / "data" / "processed" / "catboost_features.pkl"
    with open(feat_path, "rb") as f:
        bundle = pickle.load(f)
    return (
        bundle["X_train"],
        bundle["X_test"],
        bundle["y_train"],
        bundle["y_test"],
        bundle["cat_features"],
    )


def plot_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=SEVERITY_CLASSES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, m, title, fmt in [
        (axes[0], cm, "Confusion matrix (counts)", "d"),
        (axes[1], cm_norm, "Confusion matrix (row-normalized)", ".2f"),
    ]:
        im = ax.imshow(m, cmap="Blues" if title.startswith("Confusion") else "Blues")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(SEVERITY_CLASSES)))
        ax.set_yticks(range(len(SEVERITY_CLASSES)))
        ax.set_xticklabels(SEVERITY_CLASSES, rotation=30, ha="right")
        ax.set_yticklabels(SEVERITY_CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(SEVERITY_CLASSES)):
            for j in range(len(SEVERITY_CLASSES)):
                value = m[i, j]
                txt = f"{value:{fmt}}"
                color = "white" if value > (m.max() * 0.55) else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("CatBoost severity baseline — Confusion matrix", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_proba, classes_in_model, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in SEVERITY_CLASSES:
        if c not in classes_in_model:
            continue
        col = list(classes_in_model).index(c)
        y_bin = (y_true == c).astype(int).values
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        proba_c = y_proba[:, col]
        fpr, tpr, _ = roc_curve(y_bin, proba_c)
        auc = roc_auc_score(y_bin, proba_c)
        ax.plot(fpr, tpr, label=f"{c}  AUC={auc:.3f}", color=CLASS_COLORS.get(c, "gray"), lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CatBoost severity baseline — ROC (one-vs-rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_pr_curves(y_true, y_proba, classes_in_model, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in SEVERITY_CLASSES:
        if c not in classes_in_model:
            continue
        col = list(classes_in_model).index(c)
        y_bin = (y_true == c).astype(int).values
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        proba_c = y_proba[:, col]
        precision, recall, _ = precision_recall_curve(y_bin, proba_c)
        ap = average_precision_score(y_bin, proba_c)
        baseline = y_bin.sum() / len(y_bin)
        ax.plot(
            recall,
            precision,
            label=f"{c}  AP={ap:.3f} (base={baseline:.3f})",
            color=CLASS_COLORS.get(c, "gray"),
            lw=2,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("CatBoost severity baseline — Precision-Recall (one-vs-rest)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, out_path: Path, top_n: int = 20) -> None:
    fi = model.get_feature_importance(prettified=True).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.34)))
    y_pos = np.arange(len(fi))
    ax.barh(y_pos, fi["Importances"].values, color="#2c3e50")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi["Feature Id"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (PredictionValuesChange)")
    ax.set_title(f"CatBoost severity baseline — top-{top_n} feature importance")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(fi["Importances"].values):
        ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("viz")

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = PROJECT_ROOT / "models" / "catboost_severity.cbm"
    model = load_model(model_path)

    X_train, X_test, y_train, y_test, cat_features = _load_data()
    logger.info("Loaded model + test set: %d examples", len(X_test))

    y_pred = pd.Series(np.array(model.predict(X_test)).ravel(), index=y_test.index)
    y_proba = model.predict_proba(X_test)
    classes_in_model = list(model.classes_)

    plot_confusion_matrix(y_test, y_pred, out_dir / "confusion_matrix.png")
    logger.info("✓ confusion_matrix.png")

    plot_roc_curves(y_test, y_proba, classes_in_model, out_dir / "roc_curves.png")
    logger.info("✓ roc_curves.png")

    plot_pr_curves(y_test, y_proba, classes_in_model, out_dir / "pr_curves.png")
    logger.info("✓ pr_curves.png")

    plot_feature_importance(model, out_dir / "feature_importance_baseline.png", top_n=20)
    logger.info("✓ feature_importance_baseline.png")


if __name__ == "__main__":
    main()
