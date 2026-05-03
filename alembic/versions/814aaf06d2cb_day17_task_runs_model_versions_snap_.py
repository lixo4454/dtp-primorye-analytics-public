"""day17_task_runs_model_versions_snap_point_osm_roads

Revision ID: 814aaf06d2cb
Revises: b9a4f157c333
Create Date: 2026-05-02 10:26:01.247206

инфраструктура для автообновления.

1. ``task_runs`` — журнал запусков Celery-задач: status (running/success/error),
   started_at/finished_at, duration_ms, error_message, payload JSONB
   (для воспроизводимости — параметры запуска).

2. ``model_versions`` — версионирование ML-моделей. Один model_name может
   иметь много версий, ровно одна ``is_current=TRUE``. Partial unique index
   гарантирует это constraint'ом БД, а не приложением.

3. Колонки ``accidents.point_snapped`` / ``snap_distance_m`` / ``snap_method``
   / ``snap_road_id`` — результаты привязки к OSM road graph.
   - ``snap_method``: 'osm_road' / 'unchanged' / 'failed' / NULL (ещё не обработано)
   - GIST-индекс на ``point_snapped`` для будущих spatial-запросов с снапнутыми

4. ``osm_roads`` — выгрузка OSM highway-ребёр для Приморья.
   - geom: LineString WGS84
   - highway_class: motorway / trunk / primary / ... / service
   - GIST-индекс на ``geom::geography`` для ST_ClosestPoint в snap-to-road
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = '814aaf06d2cb'
down_revision: Union[str, None] = 'b9a4f157c333'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ===== task_runs =====
    op.create_table(
        "task_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_name", sa.String(100), nullable=False, index=True,
                  comment="Полное имя Celery-задачи (src.tasks.parse_dtp_stat и т.д.)"),
        sa.Column("celery_task_id", sa.String(100), nullable=True,
                  comment="UUID, выданный Celery — для трассировки в логах worker'а"),
        sa.Column("status", sa.String(20), nullable=False,
                  comment="running / success / error / skipped"),
        sa.Column("started_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("payload", JSONB(), nullable=True,
                  comment="Параметры запуска + summary результата (JSON)"),
    )
    op.create_index("idx_task_runs_name_started", "task_runs", ["task_name", "started_at"])

    # ===== model_versions =====
    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("model_name", sa.String(50), nullable=False, index=True,
                  comment="prophet_dtp / catboost_severity_v2 / bertopic_dtp"),
        sa.Column("version_path", sa.Text(), nullable=False,
                  comment="Относительный путь к файлу-снапшоту (models/prophet_dtp_<ts>.pkl)"),
        sa.Column("trained_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("train_size", sa.Integer(), nullable=True,
                  comment="Кол-во train-записей"),
        sa.Column("metadata_json", JSONB(), nullable=True,
                  comment="MAPE / F1 / ECE / hyperparams — для аудита и отката"),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
    )
    # Partial unique: одна "is_current" на model_name
    op.execute(
        "CREATE UNIQUE INDEX uq_model_versions_current "
        "ON model_versions (model_name) WHERE is_current = TRUE;"
    )

    # ===== accidents: snap-to-road colonies =====
    op.add_column("accidents", sa.Column("snap_method", sa.String(20), nullable=True,
                  comment="osm_road / unchanged / failed / NULL=не обработано"))
    op.add_column("accidents", sa.Column("snap_distance_m", sa.Float(), nullable=True,
                  comment="Расстояние от raw point до snapped (метры, geography)"))
    op.add_column("accidents", sa.Column("snap_road_id", sa.BigInteger(), nullable=True,
                  comment="osm_roads.id выбранного ребра (NULL если unchanged/failed)"))
    op.execute(
        "ALTER TABLE accidents ADD COLUMN point_snapped geometry(Point, 4326);"
    )
    op.execute(
        "CREATE INDEX idx_accidents_point_snapped "
        "ON accidents USING gist (point_snapped);"
    )
    op.execute(
        "CREATE INDEX idx_accidents_snap_method ON accidents (snap_method);"
    )

    # ===== osm_roads =====
    op.execute("""
        CREATE TABLE osm_roads (
            id BIGSERIAL PRIMARY KEY,
            osm_way_id BIGINT NOT NULL,
            highway_class VARCHAR(30) NOT NULL,
            name TEXT,
            oneway BOOLEAN DEFAULT FALSE,
            maxspeed INTEGER,
            geom geometry(LineString, 4326) NOT NULL,
            length_m DOUBLE PRECISION,
            tags JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)
    op.execute("CREATE INDEX idx_osm_roads_geom ON osm_roads USING gist (geom);")
    op.execute("CREATE INDEX idx_osm_roads_geog ON osm_roads USING gist ((geom::geography));")
    op.execute("CREATE INDEX idx_osm_roads_highway ON osm_roads (highway_class);")
    op.execute("CREATE INDEX idx_osm_roads_way ON osm_roads (osm_way_id);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS osm_roads;")

    op.execute("DROP INDEX IF EXISTS idx_accidents_point_snapped;")
    op.execute("DROP INDEX IF EXISTS idx_accidents_snap_method;")
    op.execute("ALTER TABLE accidents DROP COLUMN IF EXISTS point_snapped;")
    op.drop_column("accidents", "snap_road_id")
    op.drop_column("accidents", "snap_distance_m")
    op.drop_column("accidents", "snap_method")

    op.execute("DROP INDEX IF EXISTS uq_model_versions_current;")
    op.drop_table("model_versions")

    op.drop_index("idx_task_runs_name_started", table_name="task_runs")
    op.drop_table("task_runs")
