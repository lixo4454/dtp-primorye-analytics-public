"""add_geography_gist_index_for_st_dwithin

Revision ID: b9a4f157c333
Revises: bc06c4561065
Create Date: 2026-05-01 23:07:56.833712

эндпоинт ``/recommendations/point`` использует
``ST_DWithin(point::geography, ...)`` для агрегации ДТП в радиусе
30..1000 м вокруг произвольной точки. Существующий GIST-индекс
``idx_accidents_point`` построен на ``geometry``-типе и
не помогает запросу с ``::geography``-кастом — EXPLAIN показывает
``Parallel Seq Scan`` (~70 мс на 28k точек).

Решение: добавить функциональный GIST-индекс на выражение
``(point::geography)``. После этого PostGIS использует Index Scan,
время запроса падает до ~10-30 мс (соответствует methodology
раздел 5.2).
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'b9a4f157c333'
down_revision: Union[str, None] = 'bc06c4561065'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_accidents_point_geog "
        "ON accidents USING gist ((point::geography));"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_accidents_point_geog;")
