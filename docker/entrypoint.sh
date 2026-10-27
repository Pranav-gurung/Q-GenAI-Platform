#!/bin/sh
# Apply migrations, seed simulated reference data (idempotent), build the SOP index when RAG is enabled,
# then start the given command.
set -e
alembic upgrade head
if [ "${SEED_SAMPLE_DATA:-true}" = "true" ]; then
  python -m scripts.seed_sample_data
fi
if [ "${RAG_ENABLED:-false}" = "true" ]; then
  python -m backend.rag.ingest   # validates the knowledge base; fails startup on invalid documents
fi
exec "$@"
