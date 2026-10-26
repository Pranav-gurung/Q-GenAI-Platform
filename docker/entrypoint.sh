#!/bin/sh
# Apply migrations, seed simulated reference data (idempotent), then start the given command.
set -e
alembic upgrade head
if [ "${SEED_SAMPLE_DATA:-true}" = "true" ]; then
  python -m scripts.seed_sample_data
fi
exec "$@"
