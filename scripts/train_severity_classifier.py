"""
Обучение CatBoost-классификатора тяжести ДТП (baseline).

Что делает:
1. Достаёт accident-level feature matrix через единый SQL-запрос с
   агрегатами по vehicles + participants + pedestrians.
2. Проверяет sanity: распределение severity, NaN-rate, отсутствие
   leakage-колонок (lost_amount/suffer_amount/pers_amount).
3. Стратифицированный сплит train/test (80/20) по severity.
4. Обучает CatBoost с auto_class_weights='Balanced'.
5. Считает все метрики (accuracy, F1-macro/weighted, per-class P/R/F1,
   ROC-AUC, PR-AUC, confusion matrix, feature importance).
6. Сохраняет модель + метрики + датасет для последующего тюнинга.

Запуск:
    python -m scripts.train_severity_classifier

Артефакты:
    models/catboost_severity.cbm
    data/processed/catboost_baseline_metrics.json
    data/processed/catboost_features.pkl   ← для последующего тюнинга и SHAP
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.session import get_session
from src.ml.severity_classifier import (
    LEAKAGE_COLUMNS,
    SEVERITY_CLASSES,
    build_feature_matrix,
    evaluate,
    get_categorical_features,
    save_dataframe_pickle,
    save_metrics,
    save_model,
    stratified_split,
    train_catboost,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _setup_logging()
    logger = logging.getLogger("train_severity")

    out_model = PROJECT_ROOT / "models" / "catboost_severity.cbm"
    out_metrics = PROJECT_ROOT / "data" / "processed" / "catboost_baseline_metrics.json"
    out_features = PROJECT_ROOT / "data" / "processed" / "catboost_features.pkl"

    # =================================================================
    # 1. Загрузка feature matrix
    # =================================================================
    with get_session() as session:
        X, y = build_feature_matrix(session)

    # Sanity: количество ДТП vs распределение
    logger.info("=" * 60)
    logger.info("FEATURE MATRIX SANITY CHECK")
    logger.info("=" * 60)
    logger.info("Total accidents: %d", len(X))
    assert len(X) == 29413, f"Ожидаем 29413, получили {len(X)}"

    severity_dist = y.value_counts(dropna=False)
    logger.info("Severity distribution:")
    for cls in SEVERITY_CLASSES:
        n = int(severity_dist.get(cls, 0))
        share = n / len(y) * 100
        logger.info("  %-18s %6d  %5.2f%%", cls, n, share)

    logger.info("Features: %d columns", len(X.columns))
    for col in X.columns:
        nan_rate = X[col].isna().mean() * 100
        nun = X[col].nunique(dropna=True)
        logger.info(
            "  %-25s dtype=%-10s NaN=%5.1f%%  nuniq=%5d",
            col,
            str(X[col].dtype),
            nan_rate,
            nun,
        )

    # Leakage-defence (двойная проверка)
    for bad in LEAKAGE_COLUMNS:
        assert bad not in X.columns, f"LEAKAGE: {bad!r} в X.columns"
    logger.info("Leakage-check: lost_amount/suffer_amount/pers_amount ✓ отсутствуют")

    # =================================================================
    # 2. Train/test split
    # =================================================================
    logger.info("=" * 60)
    logger.info("STRATIFIED 80/20 SPLIT")
    logger.info("=" * 60)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    cat_features = get_categorical_features(X_train)
    logger.info("Categorical features (%d): %s", len(cat_features), cat_features)

    # =================================================================
    # 3. Обучение
    # =================================================================
    logger.info("=" * 60)
    logger.info("CATBOOST TRAINING")
    logger.info("=" * 60)
    model = train_catboost(
        X_train=X_train,
        y_train=y_train,
        cat_features=cat_features,
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        eval_set=(X_test, y_test),
        verbose=100,
    )

    # =================================================================
    # 4. Оценка
    # =================================================================
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    metrics = evaluate(model, X_test, y_test)
    metrics["meta"] = {
        "n_accidents_total": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(X.columns),
        "n_categorical_features": len(cat_features),
        "iterations_used": int(getattr(model, "best_iteration_", -1) or -1),
        "iterations_max": 1500,
        "test_size": 0.2,
        "random_state": 42,
        "auto_class_weights": "Balanced",
    }

    logger.info("Accuracy:    %.4f", metrics["accuracy"])
    logger.info("F1-macro:    %.4f", metrics["f1_macro"])
    logger.info("F1-weighted: %.4f", metrics["f1_weighted"])
    logger.info("Per-class:")
    for c in SEVERITY_CLASSES:
        pc = metrics["per_class"][c]
        logger.info(
            "  %-18s P=%.3f R=%.3f F1=%.3f n=%d",
            c,
            pc["precision"],
            pc["recall"],
            pc["f1"],
            pc["support"],
        )
    logger.info("ROC-AUC per class:")
    for c, auc in metrics["roc_auc_per_class"].items():
        logger.info("  %-18s ROC-AUC=%.3f  PR-AUC=%.3f", c, auc, metrics["pr_auc_per_class"][c])

    cm = np.array(metrics["confusion_matrix"])
    logger.info("Confusion matrix (rows=true, cols=predicted):")
    header = "                " + "".join(f"{c[:8]:>10s}" for c in SEVERITY_CLASSES)
    logger.info(header)
    for i, c in enumerate(SEVERITY_CLASSES):
        row_str = f"  {c[:14]:<14s}" + "".join(
            f"{cm[i, j]:>10d}" for j in range(len(SEVERITY_CLASSES))
        )
        logger.info(row_str)

    logger.info("Top-15 features by importance:")
    for fi in metrics["feature_importance"][:15]:
        logger.info("  %-25s %.3f", fi["feature"], fi["importance"])

    # =================================================================
    # 5. Сохранение
    # =================================================================
    logger.info("=" * 60)
    logger.info("SAVING ARTIFACTS")
    logger.info("=" * 60)
    save_model(model, out_model)
    save_metrics(metrics, out_metrics)
    # Сохраним feature matrix целиком — пригодится для SHAP/тюнинга,
    # чтобы не пересчитывать тяжёлый SQL заново
    full_dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cat_features": cat_features,
    }
    save_dataframe_pickle(pd.Series(full_dataset), out_features)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
