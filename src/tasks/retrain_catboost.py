"""
Ежемесячный retrain CatBoost-severity v2 + isotonic калибровка.

Стратегия:
- НЕ запускаем Optuna каждый месяц (30+ минут, overkill для +1k ДТП).
- Берём best_params из `models/catboost_severity_v2.json` (найдены в1
  через 50+ trials), переобучаем на свежем train, recalibrate isotonic
  на calib-holdout.
- Sanity-gate: F1-macro ≥ 0.50 на test, ECE_dead < 0.06 на calib —
  иначе НЕ переключаем is_current. Параметры из плана7.

Артефакты:
- `models/catboost_severity_v2_<UTC_TS>.cbm` — base model snapshot
- `models/catboost_severity_v1_calibrated_<UTC_TS>.pkl` — full calibrated
  pickle (base_model + isotonic dict). Грузится в API как готовый
  калиброванный классификатор.
- alias-копии: `catboost_severity_v2.cbm` и `catboost_severity_v1_calibrated.pkl`
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import f1_score

from src.database.session import get_session
from src.ml.severity_classifier import (
    LEAKAGE_COLUMNS,
    IsotonicCalibratedClassifier,
    build_feature_matrix,
    compute_ece,
    get_categorical_features,
    make_calibration_split,
    save_model,
    stratified_split,
)
from src.tasks.model_registry import (
    MODELS_DIR,
    atomic_swap_alias,
    register_version,
    timestamped_filename,
    try_reload_api,
)
from src.tasks.runner import logged_task

BASE_MODEL_NAME = "catboost_severity_v2"
CALIBRATED_NAME = "catboost_severity_v1_calibrated"
BASE_ALIAS = MODELS_DIR / f"{BASE_MODEL_NAME}.cbm"
CALIB_ALIAS = MODELS_DIR / f"{CALIBRATED_NAME}.pkl"
BEST_PARAMS_PATH = MODELS_DIR / f"{BASE_MODEL_NAME}.json"

# Sanity-gates из плана7
F1_MACRO_THRESHOLD = 0.50
ECE_DEAD_THRESHOLD = 0.06


def _load_best_params() -> dict[str, Any]:
    """Достаёт best_params из артефакта1."""
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_PARAMS_PATH} не найден — запусти Optuna-tune хотя бы один раз"
        )
    data = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    if "best_params" in data:
        return data["best_params"]
    return data  # legacy: прямой dict


def _retrain() -> dict[str, Any]:
    logger.info("[retrain_catboost] загружаю feature matrix...")
    with get_session() as s:
        X, y = build_feature_matrix(s)

    logger.info(f"[retrain_catboost] X={X.shape}, severity dist={y.value_counts().to_dict()}")

    # Защита от leakage (если кто-то добавил новую leakage-колонку и забыл фильтр)
    overlap = set(X.columns) & set(LEAKAGE_COLUMNS)
    if overlap:
        raise RuntimeError(f"Leakage-колонки в X: {overlap}")

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.20)
    X_refit, X_calib, y_refit, y_calib = make_calibration_split(X_train, y_train, calib_size=0.25)

    cat_features = get_categorical_features(X_refit)
    best_params = _load_best_params()

    base_kwargs = dict(
        iterations=2000,  # с use_best_model=True — early stopping
        cat_features=cat_features,
        random_seed=42,
        thread_count=-1,
        verbose=False,
        eval_metric="TotalF1",
        loss_function="MultiClass",
        auto_class_weights="Balanced",
    )
    base_kwargs.update(best_params)
    # use_best_model=True требует eval_set; clipping iterations выставляется учётом
    base_kwargs["use_best_model"] = True

    logger.info(f"[retrain_catboost] CatBoost params: {base_kwargs}")
    model = CatBoostClassifier(**base_kwargs)

    model.fit(
        X_refit,
        y_refit,
        eval_set=(X_calib, y_calib),
        cat_features=cat_features,
        verbose=False,
    )
    logger.info(f"[retrain_catboost] base trained, best_iter={model.best_iteration_}")

    # Метрики на test
    y_pred = model.predict(X_test).flatten()
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))

    # Calibrate isotonic
    calibrator = IsotonicCalibratedClassifier(model)
    calibrator.fit(X_calib, y_calib)

    # ECE_dead на calib-holdout
    proba_calib = calibrator.predict_proba(X_calib)
    classes = list(model.classes_)
    if "dead" in classes:
        k_dead = classes.index("dead")
        y_dead_bin = (np.asarray(y_calib) == "dead").astype(int)
        ece_dead = compute_ece(y_dead_bin, proba_calib[:, k_dead], n_bins=15)
    else:
        ece_dead = float("nan")
        logger.warning("'dead' нет в classes_, ECE_dead=NaN")

    summary = {
        "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
        "n_train": int(len(X_refit)),
        "n_calib": int(len(X_calib)),
        "n_test": int(len(X_test)),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "ece_dead_calib": round(float(ece_dead), 4),
        "best_iter": int(model.best_iteration_),
        "params_used": best_params,
        "severity_classes": classes,
    }
    logger.info(f"[retrain_catboost] metrics: {summary}")
    return {
        "model": model,
        "calibrator": calibrator,
        "summary": summary,
        "train_size": int(len(X_refit)),
    }


def _save_calibrated(calibrator: IsotonicCalibratedClassifier, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)


@logged_task(name="src.tasks.retrain_catboost.retrain_catboost")
def retrain_catboost() -> dict[str, Any]:
    out = _retrain()
    summary = out["summary"]

    # ---- Sanity-gate ----
    failures: list[str] = []
    if summary["f1_macro"] < F1_MACRO_THRESHOLD:
        failures.append(f"f1_macro {summary['f1_macro']:.3f} < {F1_MACRO_THRESHOLD}")
    if not np.isnan(summary["ece_dead_calib"]) and summary["ece_dead_calib"] >= ECE_DEAD_THRESHOLD:
        failures.append(f"ece_dead {summary['ece_dead_calib']:.4f} ≥ {ECE_DEAD_THRESHOLD}")

    base_path = timestamped_filename(BASE_MODEL_NAME, "cbm")
    calib_path = timestamped_filename(CALIBRATED_NAME, "pkl")
    save_model(out["model"], base_path)
    _save_calibrated(out["calibrator"], calib_path)

    if failures:
        # Сохраняем обе версии, регистрируем как НЕ-current, не двигаем alias
        register_version(
            BASE_MODEL_NAME,
            base_path,
            metadata={**summary, "blocked": failures},
            train_size=out["train_size"],
            make_current=False,
        )
        register_version(
            CALIBRATED_NAME,
            calib_path,
            metadata={**summary, "blocked": failures},
            train_size=out["train_size"],
            make_current=False,
        )
        raise RuntimeError(
            f"CatBoost sanity-gate failed: {failures}. " "Версии сохранены, alias НЕ переключен."
        )

    # ---- OK: registers + atomic swap aliases ----
    register_version(
        BASE_MODEL_NAME,
        base_path,
        metadata=summary,
        train_size=out["train_size"],
        make_current=True,
    )
    atomic_swap_alias(base_path, BASE_ALIAS)

    register_version(
        CALIBRATED_NAME,
        calib_path,
        metadata=summary,
        train_size=out["train_size"],
        make_current=True,
    )
    atomic_swap_alias(calib_path, CALIB_ALIAS)

    try_reload_api(label="retrain_catboost")

    return {
        "base_path": str(base_path.relative_to(MODELS_DIR.parent)).replace("\\", "/"),
        "calib_path": str(calib_path.relative_to(MODELS_DIR.parent)).replace("\\", "/"),
        "f1_macro": summary["f1_macro"],
        "ece_dead_calib": summary["ece_dead_calib"],
        "train_size": out["train_size"],
    }
