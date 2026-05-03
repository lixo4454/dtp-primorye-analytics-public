"""add telegram age enrichment columns

Revision ID: bc06c4561065
Revises: ec580a6e805d
Create Date: 2026-04-29 21:04:01.350552

добавляем колонки для обогащения возрастов из Telegram NLP-pipeline.

В источнике dtp-stat.ru возрастов нет, но Telegram-сводки УМВД содержат
2 879 индивидуальных возрастных меток (NER-извлечения «48-летний водитель»,
«пенсионерка», «13-летний школьник»). Через 482 gold-пары пост↔ДТП
 обогащаем структурную БД.

Колонки одинаковые в participants и accident_pedestrians:
- age_from_telegram     INT — собственно возраст
- age_source            VARCHAR(50) — 'telegram_gold' / 'telegram_high_precision' (для будущего)
- age_match_context     VARCHAR(50) — оригинальный NER-контекст ("водитель", "пенсионерка") для аудита
- age_match_post_id     INT — tg_id поста для обратной трассировки

Идемпотентно: повторный UPDATE через скрипт enrichment не дублирует — обнуляет
все ранее назначенные значения с тем же age_source перед перезаписью.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bc06c4561065'
down_revision: Union[str, None] = 'ec580a6e805d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === participants ===
    op.add_column(
        'participants',
        sa.Column('age_from_telegram', sa.Integer(), nullable=True,
                  comment='Возраст из Telegram NLP (NER ages NLP-pipeline). NULL если не сматчили.'),
    )
    op.add_column(
        'participants',
        sa.Column('age_source', sa.String(length=50), nullable=True,
                  comment="Источник возраста: 'telegram_gold' / 'telegram_high_precision' / NULL"),
    )
    op.add_column(
        'participants',
        sa.Column('age_match_context', sa.String(length=50), nullable=True,
                  comment='NER-контекст возраста ("водитель", "пенсионерка") — для аудита matchа'),
    )
    op.add_column(
        'participants',
        sa.Column('age_match_post_id', sa.Integer(), nullable=True,
                  comment='tg_id Telegram-поста, из которого взят возраст — для обратной трассировки'),
    )
    op.create_index(
        'ix_participants_age_source',
        'participants',
        ['age_source'],
    )

    # === accident_pedestrians ===
    op.add_column(
        'accident_pedestrians',
        sa.Column('age_from_telegram', sa.Integer(), nullable=True,
                  comment='Возраст из Telegram NLP (NER ages NLP-pipeline). NULL если не сматчили.'),
    )
    op.add_column(
        'accident_pedestrians',
        sa.Column('age_source', sa.String(length=50), nullable=True,
                  comment="Источник возраста: 'telegram_gold' / 'telegram_high_precision' / NULL"),
    )
    op.add_column(
        'accident_pedestrians',
        sa.Column('age_match_context', sa.String(length=50), nullable=True,
                  comment='NER-контекст возраста ("пешеход", "мальчик") — для аудита matchа'),
    )
    op.add_column(
        'accident_pedestrians',
        sa.Column('age_match_post_id', sa.Integer(), nullable=True,
                  comment='tg_id Telegram-поста, из которого взят возраст — для обратной трассировки'),
    )
    op.create_index(
        'ix_accident_pedestrians_age_source',
        'accident_pedestrians',
        ['age_source'],
    )


def downgrade() -> None:
    op.drop_index('ix_accident_pedestrians_age_source', table_name='accident_pedestrians')
    op.drop_column('accident_pedestrians', 'age_match_post_id')
    op.drop_column('accident_pedestrians', 'age_match_context')
    op.drop_column('accident_pedestrians', 'age_source')
    op.drop_column('accident_pedestrians', 'age_from_telegram')

    op.drop_index('ix_participants_age_source', table_name='participants')
    op.drop_column('participants', 'age_match_post_id')
    op.drop_column('participants', 'age_match_context')
    op.drop_column('participants', 'age_source')
    op.drop_column('participants', 'age_from_telegram')
