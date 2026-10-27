"""inspection item_id for fulfillment traces

Adds a nullable, indexed `inspections.item_id`: the physical unit / handling-unit id that links inspections of
the same item across fulfillment stages (RECEIVING and STORAGE have no order id yet). Existing rows keep NULL.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-26
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('inspections', schema=None) as batch_op:
        batch_op.add_column(sa.Column('item_id', sa.String(length=64), nullable=True))
        batch_op.create_index(batch_op.f('ix_inspections_item_id'), ['item_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('inspections', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_inspections_item_id'))
        batch_op.drop_column('item_id')
