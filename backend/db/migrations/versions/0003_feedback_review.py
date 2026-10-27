"""structured operator review on feedback (Part 6)

Adds nullable review columns to `feedback`: decision_correct, defect_correct, stage_correct, rca_useful,
action_useful, the AI values they refer to (ai_defect_types, ai_stage) and ai_snapshot (what the AI showed at
review time). Existing rows keep NULL (= not reviewed) and an empty snapshot. inspection_results is untouched.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-27
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

JSON_T = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
REVIEW_COLUMNS = ('decision_correct', 'defect_correct', 'stage_correct', 'rca_useful', 'action_useful')


def upgrade() -> None:
    with op.batch_alter_table('feedback', schema=None) as batch_op:
        for name in REVIEW_COLUMNS:
            batch_op.add_column(sa.Column(name, sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('ai_defect_types', JSON_T, nullable=True))
        batch_op.add_column(sa.Column('ai_stage', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('ai_snapshot', JSON_T, nullable=False, server_default='{}'))
        batch_op.create_index(batch_op.f('ix_feedback_ai_stage'), ['ai_stage'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('feedback', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_feedback_ai_stage'))
        batch_op.drop_column('ai_snapshot')
        batch_op.drop_column('ai_stage')
        batch_op.drop_column('ai_defect_types')
        for name in reversed(REVIEW_COLUMNS):
            batch_op.drop_column(name)
