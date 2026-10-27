"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-10-26
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

# JSONB on PostgreSQL, JSON elsewhere. Enum-like columns are plain VARCHAR(32) on purpose:
# new fulfillment stages / states are added in code without a schema migration.
JSON_T = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table('skus',
    sa.Column('sku_id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('category', sa.String(length=100), nullable=True),
    sa.Column('packaging_type', sa.String(length=50), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('sku_id', name=op.f('pk_skus'))
    )
    op.create_table('stores',
    sa.Column('store_id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('city', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('store_id', name=op.f('pk_stores'))
    )
    op.create_table('inspections',
    sa.Column('inspection_id', sa.String(length=32), nullable=False),
    sa.Column('store_id', sa.String(length=64), nullable=False),
    sa.Column('sku_id', sa.String(length=64), nullable=False),
    sa.Column('order_id', sa.String(length=64), nullable=True),
    sa.Column('stage', sa.String(length=32), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('operator_id', sa.String(length=64), nullable=True),
    sa.Column('decision', sa.String(length=32), nullable=True),
    sa.Column('metadata', JSON_T, nullable=False),
    sa.Column('error_code', sa.String(length=64), nullable=True),
    sa.Column('error_message', sa.String(length=500), nullable=True),
    sa.Column('error_stage', sa.String(length=32), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['sku_id'], ['skus.sku_id'], name=op.f('fk_inspections_sku_id_skus')),
    sa.ForeignKeyConstraint(['store_id'], ['stores.store_id'], name=op.f('fk_inspections_store_id_stores')),
    sa.PrimaryKeyConstraint('inspection_id', name=op.f('pk_inspections'))
    )
    with op.batch_alter_table('inspections', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_inspections_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_inspections_order_id'), ['order_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_inspections_sku_id'), ['sku_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_inspections_stage'), ['stage'], unique=False)
        batch_op.create_index(batch_op.f('ix_inspections_status'), ['status'], unique=False)
        batch_op.create_index(batch_op.f('ix_inspections_store_id'), ['store_id'], unique=False)

    op.create_table('inspection_images',
    sa.Column('image_id', sa.String(length=32), nullable=False),
    sa.Column('inspection_id', sa.String(length=32), nullable=False),
    sa.Column('file_name', sa.String(length=255), nullable=False),
    sa.Column('mime_type', sa.String(length=50), nullable=False),
    sa.Column('size_bytes', sa.Integer(), nullable=False),
    sa.Column('width', sa.Integer(), nullable=False),
    sa.Column('height', sa.Integer(), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('storage_backend', sa.String(length=32), nullable=False),
    sa.Column('storage_reference', sa.String(length=500), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['inspection_id'], ['inspections.inspection_id'], name=op.f('fk_inspection_images_inspection_id_inspections'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('image_id', name=op.f('pk_inspection_images'))
    )
    with op.batch_alter_table('inspection_images', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_inspection_images_inspection_id'), ['inspection_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_inspection_images_sha256'), ['sha256'], unique=False)

    op.create_table('inspection_results',
    sa.Column('result_id', sa.String(length=32), nullable=False),
    sa.Column('inspection_id', sa.String(length=32), nullable=False),
    sa.Column('quality', JSON_T, nullable=True),
    sa.Column('cv', JSON_T, nullable=True),
    sa.Column('decision', JSON_T, nullable=True),
    sa.Column('pipeline_analysis', JSON_T, nullable=True),
    sa.Column('rca', JSON_T, nullable=True),
    sa.Column('action', JSON_T, nullable=True),
    sa.Column('cv_model_version', sa.String(length=100), nullable=True),
    sa.Column('model_versions', JSON_T, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['inspection_id'], ['inspections.inspection_id'], name=op.f('fk_inspection_results_inspection_id_inspections'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('result_id', name=op.f('pk_inspection_results')),
    sa.UniqueConstraint('inspection_id', name=op.f('uq_inspection_results_inspection_id'))
    )
    with op.batch_alter_table('inspection_results', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_inspection_results_cv_model_version'), ['cv_model_version'], unique=False)

    op.create_table('pipeline_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('inspection_id', sa.String(length=32), nullable=False),
    sa.Column('event_type', sa.String(length=32), nullable=False),
    sa.Column('from_state', sa.String(length=32), nullable=True),
    sa.Column('to_state', sa.String(length=32), nullable=True),
    sa.Column('module', sa.String(length=64), nullable=True),
    sa.Column('module_status', sa.String(length=32), nullable=True),
    sa.Column('duration_ms', sa.Float(), nullable=True),
    sa.Column('error_code', sa.String(length=64), nullable=True),
    sa.Column('message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['inspection_id'], ['inspections.inspection_id'], name=op.f('fk_pipeline_events_inspection_id_inspections'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_pipeline_events'))
    )
    with op.batch_alter_table('pipeline_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pipeline_events_inspection_id'), ['inspection_id'], unique=False)

    op.create_table('feedback',
    sa.Column('feedback_id', sa.String(length=32), nullable=False),
    sa.Column('inspection_id', sa.String(length=32), nullable=False),
    sa.Column('result_id', sa.String(length=32), nullable=True),
    sa.Column('operator_id', sa.String(length=64), nullable=False),
    sa.Column('outcome', sa.String(length=32), nullable=False),
    sa.Column('defect_types', JSON_T, nullable=False),
    sa.Column('corrected_detections', JSON_T, nullable=False),
    sa.Column('suspected_stage', sa.String(length=32), nullable=True),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.Column('system_outcome', sa.String(length=32), nullable=True),
    sa.Column('is_correction', sa.Boolean(), nullable=False),
    sa.Column('model_versions', JSON_T, nullable=False),
    sa.Column('learning_status', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['inspection_id'], ['inspections.inspection_id'], name=op.f('fk_feedback_inspection_id_inspections'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['result_id'], ['inspection_results.result_id'], name=op.f('fk_feedback_result_id_inspection_results'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('feedback_id', name=op.f('pk_feedback'))
    )
    with op.batch_alter_table('feedback', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_feedback_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_feedback_inspection_id'), ['inspection_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_feedback_learning_status'), ['learning_status'], unique=False)
        batch_op.create_index(batch_op.f('ix_feedback_operator_id'), ['operator_id'], unique=False)



def downgrade() -> None:
    with op.batch_alter_table('feedback', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_feedback_operator_id'))
        batch_op.drop_index(batch_op.f('ix_feedback_learning_status'))
        batch_op.drop_index(batch_op.f('ix_feedback_inspection_id'))
        batch_op.drop_index(batch_op.f('ix_feedback_created_at'))

    op.drop_table('feedback')
    with op.batch_alter_table('pipeline_events', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pipeline_events_inspection_id'))

    op.drop_table('pipeline_events')
    with op.batch_alter_table('inspection_results', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_inspection_results_cv_model_version'))

    op.drop_table('inspection_results')
    with op.batch_alter_table('inspection_images', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_inspection_images_sha256'))
        batch_op.drop_index(batch_op.f('ix_inspection_images_inspection_id'))

    op.drop_table('inspection_images')
    with op.batch_alter_table('inspections', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_inspections_store_id'))
        batch_op.drop_index(batch_op.f('ix_inspections_status'))
        batch_op.drop_index(batch_op.f('ix_inspections_stage'))
        batch_op.drop_index(batch_op.f('ix_inspections_sku_id'))
        batch_op.drop_index(batch_op.f('ix_inspections_order_id'))
        batch_op.drop_index(batch_op.f('ix_inspections_created_at'))

    op.drop_table('inspections')
    op.drop_table('stores')
    op.drop_table('skus')
