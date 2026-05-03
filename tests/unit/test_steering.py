"""Unit-тесты CSV-loader'а справочника правый/левый руль.

Покрывает функцию ``load_brand_reference`` из ``src.analysis.steering``.
БД-зависимые функции (``apply_to_vehicles``, ``print_final_stats``)
тестируются как integration в реальной dev-БД.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis import steering


def test_load_brand_reference_returns_dataframe():
    """Реальный справочник из data/raw грузится в pandas-DataFrame."""
    df = steering.load_brand_reference()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_brand_reference_has_required_columns():
    df = steering.load_brand_reference()
    for col in ("brand", "is_right_hand_drive", "confidence"):
        assert col in df.columns


def test_load_brand_reference_bool_parsed():
    """Колонка is_right_hand_drive должна быть bool после парсинга."""
    df = steering.load_brand_reference()
    # NaN исключаем — берём только не-нулевые
    non_null = df["is_right_hand_drive"].dropna()
    assert non_null.map(lambda v: isinstance(v, bool)).all()


def test_load_brand_reference_no_missing_brand():
    """После загрузки в DataFrame не должно остаться строк с пустым brand."""
    df = steering.load_brand_reference()
    assert df["brand"].notna().all()
    assert df["confidence"].notna().all()


def test_load_brand_reference_contains_known_brands():
    """Проверяем что японские RHD-марки присутствуют — ядро справочника."""
    df = steering.load_brand_reference()
    brands_upper = {b.upper() for b in df["brand"].dropna()}
    for brand in ("TOYOTA", "NISSAN", "HONDA", "MAZDA", "SUBARU"):
        assert brand in brands_upper


def test_load_brand_reference_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Если CSV нет — ясное FileNotFoundError, а не молчаливое падение."""
    missing = tmp_path / "no_such.csv"
    monkeypatch.setattr(steering, "REFERENCE_CSV", missing)
    with pytest.raises(FileNotFoundError):
        steering.load_brand_reference()


def test_load_brand_reference_custom_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Парсер корректно читает кастомный CSV с минимальной схемой."""
    csv_path = tmp_path / "brands.csv"
    csv_path.write_text(
        "brand,is_right_hand_drive,confidence,reasoning\n"
        "FOO,true,high,test\n"
        "BAR,false,medium,test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(steering, "REFERENCE_CSV", csv_path)
    df = steering.load_brand_reference()
    assert len(df) == 2
    assert set(df["brand"]) == {"FOO", "BAR"}
    indexed = df.set_index("brand")["is_right_hand_drive"]
    assert bool(indexed["FOO"]) is True
    assert bool(indexed["BAR"]) is False
