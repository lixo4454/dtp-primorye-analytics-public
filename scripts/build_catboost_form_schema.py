"""Регенерация ``data/processed/catboost_form_schema.json``.

Извлекает baseline-дефолты для Streamlit-формы /predict/severity:
- median для числовых признаков
- mode для категориальных
- top-30 значений для каждой категориальной колонки (выпадающие списки)
- min/max/p05/p95 для числовых (sliders с разумными границами)
- class priors из y_train (для отображения base rate в UI)

Источник — ``data/processed/catboost_features.pkl`` (результат
базового train/test split).

Зачем как отдельный JSON, а не загрузка pickle в Streamlit:
- Pickle тащит joblib + pandas + numpy в Streamlit-контейнер
- pkl ~5 МБ vs JSON ~9 КБ
- JSON безопаснее (нет произвольного кода в десериализации)

Запуск:
    python scripts/build_catboost_form_schema.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

PKL_PATH = Path("data/processed/catboost_features.pkl")
OUT_PATH = Path("data/processed/catboost_form_schema.json")


def main() -> None:
    data = joblib.load(PKL_PATH)
    X = data["X_train"]
    y = data["y_train"]
    cat_features = list(data["cat_features"])
    bool_cols = [c for c in X.columns if X[c].dtype == bool]
    numeric_cols = [c for c in X.columns if c not in cat_features and c not in bool_cols]

    defaults: dict[str, object] = {}
    cat_choices: dict[str, list[str]] = {}
    numeric_ranges: dict[str, dict[str, float]] = {}

    for col in X.columns:
        if col in cat_features:
            vc = X[col].value_counts()
            defaults[col] = str(vc.index[0])
            cat_choices[col] = [str(v) for v in vc.head(30).index.tolist()]
        elif col in bool_cols:
            defaults[col] = bool(X[col].mode()[0])
        else:
            s = X[col].dropna()
            defaults[col] = float(X[col].median())
            numeric_ranges[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "p05": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
            }

    out = {
        "feature_columns": list(X.columns),
        "cat_features": cat_features,
        "bool_features": bool_cols,
        "numeric_features": numeric_cols,
        "defaults": defaults,
        "cat_choices": cat_choices,
        "numeric_ranges": numeric_ranges,
        "class_priors": {str(k): float(v) for k, v in y.value_counts(normalize=True).items()},
        "n_train": int(len(X)),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
    print(f"  feature_columns: {len(out['feature_columns'])}")
    print(f"  cat: {len(cat_features)} | bool: {len(bool_cols)} | num: {len(numeric_cols)}")
    print(f"  class_priors: {out['class_priors']}")


if __name__ == "__main__":
    main()
