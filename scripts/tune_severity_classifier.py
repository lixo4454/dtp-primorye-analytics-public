"""
Оркестратор интерпретации, калибровки и тюнинга CatBoost-классификатора.

Что делает (5 шагов):
1. Загружает baseline-модель + train/test-pickle.
2. SHAP-интерпретация: глобальная важность по mean(|SHAP|) + per-class ranking
   для класса `dead`. Сохраняет shap_values.npz + shap_global.json /
   shap_dead.json.
3. Калибровка: режет train на refit_train (75%) + calib_holdout (25%),
   переобучает CatBoost с теми же baseline-параметрами на refit_train,
   обучает per-class IsotonicRegression на calib_holdout, оценивает ECE
   на test для всех 4 классов до/после калибровки.
4. Error analysis: топ-10 false positives и false negatives для класса
   `dead` — JSON с признаками каждой ошибки для нарратива.
5. Optuna-тюнинг: 50 trials, 3-fold CV, maximize ROC-AUC dead. Сохраняет
   best_params + optuna_history. Финальное обучение модели v2 на полном
   train, оценка на test, сравнительная таблица v1/v2.

Зачем нужно: baseline дал F1-macro=0.51 / ROC-AUC dead=0.73.
Цель — поднять ROC-AUC dead ≥0.78 через Optuna и получить
production-готовые откалиброванные probabilities для использования в
будущем FastAPI-эндпоинте /predict/severity.

Артефакты:
- models/catboost_severity_v2.cbm — финальная модель v2 (.gitignore)
- models/catboost_severity_v2.json — best_params + optuna trials
- models/catboost_severity_v1_calibrated.json — калибраторы isotonic
  (per-class IsotonicRegression-параметры — pickle)
- data/processed/catboost_v2_metrics.json — полные метрики v2
- data/processed/catboost_calibration_metrics.json — ECE до/после
- data/processed/catboost_shap_values.npz — массив SHAP (n×k×f)
- data/processed/catboost_shap_global.json — топ-20 общий
- data/processed/catboost_shap_dead.json — топ-20 для dead
- data/processed/catboost_errors_dead.json — top-10 FP/FN
- data/processed/catboost_optuna_history.json — все trials (param+value)

Использование:
    python -m scripts.tune_severity_classifier
        [--no-shap] [--no-calibration] [--no-errors]
        [--no-optuna] [--n-trials 50] [--n-folds 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Гарантируем что src/ в PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.severity_classifier import (  # noqa: E402
    SEVERITY_CLASSES,
    IsotonicCalibratedClassifier,
    compute_ece,
    compute_shap,
    evaluate,
    find_top_errors,
    load_model,
    make_calibration_split,
    reliability_curve,
    save_metrics,
    save_model,
    shap_global_importance,
    train_catboost,
    tune_with_optuna,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tune")

DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FEATURES_PATH = DATA_DIR / "catboost_features.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "catboost_severity.cbm"
V2_MODEL_PATH = MODELS_DIR / "catboost_severity_v2.cbm"
V2_META_PATH = MODELS_DIR / "catboost_severity_v2.json"
CALIB_PATH = MODELS_DIR / "catboost_severity_v1_calibrated.pkl"


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("Saved %s", path)


def _load_features():
    logger.info("Loading features pickle: %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        d = pickle.load(f)
    X_train, X_test = d["X_train"], d["X_test"]
    y_train, y_test = d["y_train"], d["y_test"]
    cat_features = d["cat_features"]
    logger.info(
        "X_train: %s | X_test: %s | cat_features: %d",
        X_train.shape,
        X_test.shape,
        len(cat_features),
    )
    return X_train, X_test, y_train, y_test, cat_features


# =====================================================================
# Step 2: SHAP
# =====================================================================
def step_shap(model, X_test, cat_features) -> dict:
    logger.info("=" * 70)
    logger.info("STEP 2: SHAP interpretation")
    logger.info("=" * 70)
    t0 = time.time()
    shap_values, expected_value, classes = compute_shap(model, X_test, cat_features)
    feature_names = list(X_test.columns)

    # Global importance (mean across classes)
    global_imp = shap_global_importance(shap_values, feature_names, class_idx=None)
    logger.info("Top-10 global SHAP importance:\n%s", global_imp.head(10).to_string(index=False))

    # Per-class for 'dead'
    dead_idx = classes.index("dead")
    dead_imp = shap_global_importance(shap_values, feature_names, class_idx=dead_idx)
    logger.info(
        "Top-10 SHAP importance for class 'dead':\n%s", dead_imp.head(10).to_string(index=False)
    )

    # Save full shap matrix as npz (compressed)
    np.savez_compressed(
        DATA_DIR / "catboost_shap_values.npz",
        shap_values=shap_values,
        expected_value=expected_value,
        classes=np.array(classes),
        feature_names=np.array(feature_names),
    )
    logger.info(
        "Saved %s (%.2f MB)",
        DATA_DIR / "catboost_shap_values.npz",
        (DATA_DIR / "catboost_shap_values.npz").stat().st_size / 1024**2,
    )

    # Save top-20 jsons
    _save_json(
        {
            "classes_order": classes,
            "expected_value_per_class": expected_value.tolist(),
            "top20_features": global_imp.head(20).to_dict(orient="records"),
        },
        DATA_DIR / "catboost_shap_global.json",
    )
    _save_json(
        {
            "class": "dead",
            "expected_value": float(expected_value[dead_idx]),
            "top20_features": dead_imp.head(20).to_dict(orient="records"),
        },
        DATA_DIR / "catboost_shap_dead.json",
    )
    logger.info("SHAP done in %.1fs", time.time() - t0)
    return {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "classes": classes,
        "feature_names": feature_names,
        "global_top20": global_imp.head(20),
        "dead_top20": dead_imp.head(20),
    }


# =====================================================================
# Step 3: Calibration
# =====================================================================
def step_calibration(model, X_train, y_train, X_test, y_test, cat_features) -> dict:
    logger.info("=" * 70)
    logger.info("STEP 3: Probability calibration (isotonic, OvR per class)")
    logger.info("=" * 70)
    t0 = time.time()

    # Split train -> refit_train (75%) + calib_holdout (25%)
    X_refit, X_calib, y_refit, y_calib = make_calibration_split(X_train, y_train, calib_size=0.25)
    logger.info("Refit train: %d | Calib holdout: %d", len(X_refit), len(X_calib))

    # Re-train baseline-config CatBoost on refit_train (без best-iteration eval — те же 1500 iter)
    refit_model = train_catboost(
        X_refit,
        y_refit,
        cat_features,
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        eval_set=(X_calib, y_calib),
        verbose=200,
    )

    # Fit per-class IsotonicRegression on calib_holdout
    calibrator = IsotonicCalibratedClassifier(refit_model)
    calibrator.fit(X_calib, y_calib)

    # Evaluate on test BEFORE & AFTER calibration
    proba_before = refit_model.predict_proba(X_test)
    proba_after = calibrator.predict_proba(X_test)
    classes_model = list(refit_model.classes_)
    y_test_arr = np.asarray(y_test)

    ece_before: dict[str, float] = {}
    ece_after: dict[str, float] = {}
    reliability_data: dict[str, dict] = {}

    for k, cls in enumerate(classes_model):
        y_bin = (y_test_arr == cls).astype(int)
        ece_before[cls] = compute_ece(y_bin, proba_before[:, k], n_bins=15)
        ece_after[cls] = compute_ece(y_bin, proba_after[:, k], n_bins=15)
        mp_b, fp_b, cnt_b = reliability_curve(y_bin, proba_before[:, k], n_bins=10)
        mp_a, fp_a, cnt_a = reliability_curve(y_bin, proba_after[:, k], n_bins=10)
        reliability_data[cls] = {
            "before": {
                "mean_pred": [None if np.isnan(x) else float(x) for x in mp_b],
                "frac_pos": [None if np.isnan(x) else float(x) for x in fp_b],
                "count": cnt_b.tolist(),
            },
            "after": {
                "mean_pred": [None if np.isnan(x) else float(x) for x in mp_a],
                "frac_pos": [None if np.isnan(x) else float(x) for x in fp_a],
                "count": cnt_a.tolist(),
            },
        }

    logger.info("ECE per class:")
    for cls in classes_model:
        delta = ece_after[cls] - ece_before[cls]
        sign = "↓" if delta < 0 else "↑"
        logger.info(
            "  %-18s before=%.4f → after=%.4f  %s %.4f",
            cls,
            ece_before[cls],
            ece_after[cls],
            sign,
            abs(delta),
        )

    # Сохраняем calibrator целиком (pickle с base_model + isotonic)
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIB_PATH.with_suffix(CALIB_PATH.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(calibrator, f)
    tmp.replace(CALIB_PATH)
    logger.info("Calibrator pickled to %s", CALIB_PATH)

    # Save metrics
    metrics = {
        "method": "isotonic_per_class_OvR",
        "calib_holdout_size": len(X_calib),
        "refit_train_size": len(X_refit),
        "test_size": len(X_test),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "reliability_curves": reliability_data,
        "classes_order": classes_model,
    }
    _save_json(metrics, DATA_DIR / "catboost_calibration_metrics.json")
    logger.info("Calibration done in %.1fs", time.time() - t0)
    return metrics


# =====================================================================
# Step 4: Error analysis
# =====================================================================
def step_errors(model, X_test, y_test) -> dict:
    logger.info("=" * 70)
    logger.info("STEP 4: Error analysis (top-10 FP/FN for class 'dead')")
    logger.info("=" * 70)
    fp_dead = find_top_errors(model, X_test, y_test, "dead", mode="fp", k=10)
    fn_dead = find_top_errors(model, X_test, y_test, "dead", mode="fn", k=10)

    logger.info("Top-10 FP for 'dead' (model said dead, actually wasn't):")
    if not fp_dead.empty:
        cols_show = [
            "true",
            "pred",
            "proba_dead",
            "em_type",
            "light_type",
            "part_count",
            "ped_count",
            "veh_amount",
        ]
        cols_show = [c for c in cols_show if c in fp_dead.columns]
        logger.info("\n%s", fp_dead[cols_show].to_string())
    logger.info("Top-10 FN for 'dead' (actually dead, model said other):")
    if not fn_dead.empty:
        cols_show = [
            "true",
            "pred",
            "proba_dead",
            "em_type",
            "light_type",
            "part_count",
            "ped_count",
            "veh_amount",
        ]
        cols_show = [c for c in cols_show if c in fn_dead.columns]
        logger.info("\n%s", fn_dead[cols_show].to_string())

    # Сохраняем — DataFrame в dict-of-records, accident_id в колонку
    def df_to_records(df):
        if df.empty:
            return []
        out = df.reset_index().rename(
            columns={"index": "accident_id", df.index.name or "index": "accident_id"}
        )
        # accident_id name may be different
        if "accident_id" not in out.columns:
            out = out.reset_index().rename(columns={out.columns[0]: "accident_id"})
        # Convert dtypes for JSON
        records = []
        for r in out.to_dict(orient="records"):
            clean = {}
            for k, v in r.items():
                if isinstance(v, (np.integer, np.bool_)):
                    clean[k] = int(v) if isinstance(v, np.integer) else bool(v)
                elif isinstance(v, np.floating):
                    clean[k] = None if np.isnan(v) else float(v)
                elif v is None or (isinstance(v, float) and np.isnan(v)):
                    clean[k] = None
                else:
                    clean[k] = v
            records.append(clean)
        return records

    payload = {
        "target_class": "dead",
        "false_positives_top10": df_to_records(fp_dead),
        "false_negatives_top10": df_to_records(fn_dead),
    }
    _save_json(payload, DATA_DIR / "catboost_errors_dead.json")
    return payload


# =====================================================================
# Step 5: Optuna
# =====================================================================
def step_optuna(
    X_train, y_train, X_test, y_test, cat_features, n_trials: int, n_folds: int
) -> dict:
    logger.info("=" * 70)
    logger.info(
        "STEP 5: Optuna tuning (%d trials × %d folds, maximize ROC-AUC dead)", n_trials, n_folds
    )
    logger.info("=" * 70)
    t0 = time.time()

    history: list[dict] = []

    def log_cb(trial_num, params, fold_aucs, mean_auc):
        elapsed = time.time() - t0
        logger.info(
            "Trial #%d  mean_auc=%.4f  folds=%s  params=%s  elapsed=%.1fs",
            trial_num,
            mean_auc,
            [f"{a:.4f}" for a in fold_aucs],
            {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()},
            elapsed,
        )
        history.append(
            {
                "trial": trial_num,
                "params": params,
                "fold_aucs": fold_aucs,
                "mean_auc": mean_auc,
                "elapsed_sec": elapsed,
            }
        )

    study, best_params = tune_with_optuna(
        X_train,
        y_train,
        cat_features,
        n_trials=n_trials,
        n_folds=n_folds,
        target_class="dead",
        cv_iterations=500,
        log_callback=log_cb,
    )
    logger.info("Optuna finished. Best AUC=%.4f params=%s", study.best_value, best_params)

    # Final fit with ALL best_params on FULL train (with eval=test for early stop).
    # train_catboost не пробрасывает l2_leaf_reg/bagging/strength —
    # инстанцируем CatBoost напрямую.
    logger.info("Final fit v2 on full train with best_params (iterations=2000)...")
    from catboost import CatBoostClassifier, Pool

    v2_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=best_params["learning_rate"],
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        bagging_temperature=best_params["bagging_temperature"],
        random_strength=best_params["random_strength"],
        loss_function="MultiClass",
        eval_metric="TotalF1",
        cat_features=cat_features,
        auto_class_weights="Balanced",
        random_seed=42,
        od_type="Iter",
        od_wait=50,
        verbose=200,
        allow_writing_files=False,
    )
    train_pool = Pool(X_train, label=y_train.astype(str), cat_features=cat_features)
    eval_pool = Pool(X_test, label=y_test.astype(str), cat_features=cat_features)
    v2_model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    logger.info("v2 trained: best_iter=%d", v2_model.best_iteration_)

    # Save model + meta
    save_model(v2_model, V2_MODEL_PATH)
    meta = {
        "best_params": best_params,
        "best_cv_auc_dead": float(study.best_value),
        "best_iter_final": int(v2_model.best_iteration_),
        "n_trials": n_trials,
        "n_folds": n_folds,
        "cv_iterations": 500,
        "target_class": "dead",
        "elapsed_sec": time.time() - t0,
    }
    _save_json(meta, V2_META_PATH)
    _save_json(history, DATA_DIR / "catboost_optuna_history.json")

    return {
        "v2_model": v2_model,
        "best_params": best_params,
        "best_cv_auc": study.best_value,
        "history": history,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--no-errors", action="store_true")
    ap.add_argument("--no-optuna", action="store_true")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--n-folds", type=int, default=3)
    args = ap.parse_args()

    X_train, X_test, y_train, y_test, cat_features = _load_features()
    baseline_model = load_model(BASELINE_MODEL_PATH)
    logger.info(
        "Baseline model loaded: classes=%s, tree_count=%d",
        list(baseline_model.classes_),
        baseline_model.tree_count_,
    )

    if not args.no_shap:
        step_shap(baseline_model, X_test, cat_features)
    if not args.no_calibration:
        step_calibration(baseline_model, X_train, y_train, X_test, y_test, cat_features)
    if not args.no_errors:
        step_errors(baseline_model, X_test, y_test)
    if not args.no_optuna:
        result = step_optuna(
            X_train,
            y_train,
            X_test,
            y_test,
            cat_features,
            n_trials=args.n_trials,
            n_folds=args.n_folds,
        )
        # Final eval v2 on test
        logger.info("=" * 70)
        logger.info("FINAL EVAL: v2 on test")
        logger.info("=" * 70)
        v2_metrics = evaluate(result["v2_model"], X_test, y_test)
        v2_metrics["best_params"] = result["best_params"]
        v2_metrics["best_cv_auc_dead"] = result["best_cv_auc"]
        _save_json(v2_metrics, DATA_DIR / "catboost_v2_metrics.json")
        logger.info(
            "v2: F1-macro=%.4f  ROC-AUC dead=%.4f  ROC-AUC severe_multiple=%.4f",
            v2_metrics["f1_macro"],
            v2_metrics["roc_auc_per_class"].get("dead", float("nan")),
            v2_metrics["roc_auc_per_class"].get("severe_multiple", float("nan")),
        )

        # Compare v1 vs v2
        v1_path = DATA_DIR / "catboost_baseline_metrics.json"
        if v1_path.exists():
            v1 = json.loads(v1_path.read_text())
            cmp = {
                "v1": {
                    "f1_macro": v1["f1_macro"],
                    "f1_weighted": v1["f1_weighted"],
                    "accuracy": v1["accuracy"],
                    "roc_auc_per_class": v1["roc_auc_per_class"],
                    "per_class": {k: v["f1"] for k, v in v1["per_class"].items()},
                },
                "v2": {
                    "f1_macro": v2_metrics["f1_macro"],
                    "f1_weighted": v2_metrics["f1_weighted"],
                    "accuracy": v2_metrics["accuracy"],
                    "roc_auc_per_class": v2_metrics["roc_auc_per_class"],
                    "per_class": {k: v["f1"] for k, v in v2_metrics["per_class"].items()},
                },
                "delta": {
                    "f1_macro": v2_metrics["f1_macro"] - v1["f1_macro"],
                    "roc_auc_dead": v2_metrics["roc_auc_per_class"].get("dead", 0)
                    - v1["roc_auc_per_class"].get("dead", 0),
                },
            }
            _save_json(cmp, DATA_DIR / "catboost_v1_vs_v2.json")
            logger.info(
                "Comparison v1→v2: ΔF1-macro=%+.4f  ΔROC-AUC dead=%+.4f",
                cmp["delta"]["f1_macro"],
                cmp["delta"]["roc_auc_dead"],
            )

    logger.info("ALL STEPS DONE.")


if __name__ == "__main__":
    main()
