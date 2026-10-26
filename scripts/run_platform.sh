#!/usr/bin/env bash
# =============================================================================
# Q-GenAI Platform — Multi-Service Local Orchestration Script
#
# Launches:
#   1. Module B/D (Inspection, Quality Policy, RBAC, Audit Ledger) on port 8000
#      using the .venv-bd environment.
#   2. Module C (GenAI Root-Cause Analysis using RAG) on port 8001
#      using the .venv environment.
#
# Handles graceful shutdown on SIGINT / SIGTERM.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure virtual environments exist
if [ ! -d "${REPO_ROOT}/.venv" ]; then
    echo "ERROR: Module A/C virtual environment '${REPO_ROOT}/.venv' not found." >&2
    exit 1
fi

if [ ! -d "${REPO_ROOT}/.venv-bd" ]; then
    echo "ERROR: Module B/D virtual environment '${REPO_ROOT}/.venv-bd' not found." >&2
    exit 1
fi

HOST="${HOST:-127.0.0.1}"
BD_PORT="${BD_PORT:-8000}"
C_PORT="${C_PORT:-8001}"

# Safe local development environment variables
export QGENAI_ENV="${QGENAI_ENV:-development}"
export QGENAI_DATABASE_URL="${QGENAI_DATABASE_URL:-sqlite:///./data/qgenai.db}"
export QGENAI_STORAGE_DIR="${QGENAI_STORAGE_DIR:-./data/storage}"
export QGENAI_MODELS_DIR="${QGENAI_MODELS_DIR:-./models}"
export QGENAI_MOCK_DETECTOR="${QGENAI_MOCK_DETECTOR:-true}"
export QGENAI_JWT_SECRET_KEY="${QGENAI_JWT_SECRET_KEY:-local-dev-jwt-secret-key-32-chars-min-secure}"
export QGENAI_AUDIT_CHECKPOINT_KEY="${QGENAI_AUDIT_CHECKPOINT_KEY:-local-dev-audit-checkpoint-hmac-32-chars-min}"

export RCA_APPROVED_DOCS_DIR="${RCA_APPROVED_DOCS_DIR:-documents/approved}"
export RCA_INDEX_DIR="${RCA_INDEX_DIR:-data/rag_index}"
export RCA_EMBEDDING_BACKEND="${RCA_EMBEDDING_BACKEND:-hashing}"
export RCA_SUPPORTED_CLASSES="${RCA_SUPPORTED_CLASSES:-crack,scratch,dent,corrosion}"
export RCA_AUDIT_SINK="${RCA_AUDIT_SINK:-jsonl}"
export RCA_AUDIT_PATH="${RCA_AUDIT_PATH:-data/audit/rca_audit.jsonl}"
export RCA_AUDIT_FAIL_MODE="${RCA_AUDIT_FAIL_MODE:-fail_closed}"

# Create required local data directories
mkdir -p "${REPO_ROOT}/module_b_d/data/storage"
mkdir -p "${REPO_ROOT}/data/rag_index"
mkdir -p "${REPO_ROOT}/data/audit"
mkdir -p "${REPO_ROOT}/documents/approved"

PID_BD=""
PID_C=""

cleanup() {
    echo ""
    echo "=== Shutting down Q-GenAI Platform services ==="
    if [ -n "${PID_BD}" ] && kill -0 "${PID_BD}" 2>/dev/null; then
        echo "Stopping Module B/D (PID: ${PID_BD})..."
        kill -TERM "${PID_BD}" 2>/dev/null || true
    fi
    if [ -n "${PID_C}" ] && kill -0 "${PID_C}" 2>/dev/null; then
        echo "Stopping Module C (PID: ${PID_C})..."
        kill -TERM "${PID_C}" 2>/dev/null || true
    fi
    wait "${PID_BD}" 2>/dev/null || true
    wait "${PID_C}" 2>/dev/null || true
    echo "Services stopped cleanly."
}

trap cleanup INT TERM EXIT

echo "================================================================="
echo " Starting Q-GenAI Platform (Multi-Service Local Orchestration)   "
echo "================================================================="
echo " Host:             ${HOST}"
echo " Module B/D Port:  ${BD_PORT} (.venv-bd)"
echo " Module C Port:    ${C_PORT} (.venv)"
echo " Mock Detector:    ${QGENAI_MOCK_DETECTOR}"
echo " Audit Fail Mode:  ${RCA_AUDIT_FAIL_MODE}"
echo "================================================================="

# Start Module B/D
echo "Starting Module B/D service on port ${BD_PORT}..."
(
    cd "${REPO_ROOT}/module_b_d"
    exec "${REPO_ROOT}/.venv-bd/bin/uvicorn" app.main:app \
        --host "${HOST}" \
        --port "${BD_PORT}" \
        --log-level info
) &
PID_BD=$!

# Start Module C
echo "Starting Module C service on port ${C_PORT}..."
(
    cd "${REPO_ROOT}"
    exec "${REPO_ROOT}/.venv/bin/uvicorn" module_c.api:create_app \
        --factory \
        --host "${HOST}" \
        --port "${C_PORT}" \
        --log-level info
) &
PID_C=$!

echo ""
echo "Services launching in background:"
echo " - Module B/D: http://${HOST}:${BD_PORT} (Docs: http://${HOST}:${BD_PORT}/docs)"
echo " - Module C:   http://${HOST}:${C_PORT} (Docs: http://${HOST}:${C_PORT}/docs)"
echo ""
echo "Press Ctrl+C to terminate both services."

# Wait for both background processes
wait "${PID_BD}" "${PID_C}"
