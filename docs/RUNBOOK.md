# Q-GenAI Platform — Operations & Verification Runbook

This runbook documents the operational lifecycle of the Q-GenAI Platform: environment setup, service startup/shutdown, configuration variables, testing procedures, and troubleshooting guides.

---

## 1. System Overview & Architecture

The Q-GenAI Platform comprises four integrated modules:

| Module | Role | Service / Path | Runtime Env | Port |
|---|---|---|---|---|
| **Module A** | Synthetic Data Engine (generation, splitting, augmentation, YOLO/COCO/Anomaly export, feedback) | `module_a/` | `.venv` | - |
| **Module B** | Vision Defect Inspection (image ingestion, defect detection, bounding box extraction) | `module_b_d/app/` | `.venv-bd` | `8000` |
| **Module C** | GenAI Root-Cause Analysis (approved SOP ingestion, vector retrieval, grounded LLM RCA generation) | `module_c/` | `.venv` | `8001` |
| **Module D** | Security & Governance (JWT authentication, RBAC, versioned quality policies, tamper-evident SHA-256 audit ledger) | `module_b_d/app/services/` | `.venv-bd` | `8000` |

---

## 2. Environment Setup & Isolation

The platform maintains two dedicated Python 3.13 virtual environments to avoid package conflicts:

### A. `.venv` (Module A & Module C)
- **Primary Dependencies:** PyTorch 2.14, Sentence-Transformers, ChromaDB, PyMuPDF, PyPDF, FastAPI, Uvicorn, Pydantic, pytest.
- **Activation / Execution:** `.venv/bin/python` or `.venv/bin/pytest`.

### B. `.venv-bd` (Module B & Module D)
- **Primary Dependencies:** PyTorch 2.6.0, Ultralytics 8.3, Torchvision, SQLAlchemy, Argon2id, PyJWT, FastAPI, Uvicorn, Streamlit, pytest.
- **Activation / Execution:** `.venv-bd/bin/python` or `.venv-bd/bin/pytest`.

> [!IMPORTANT]
> Always execute commands with the specific virtual environment binary (e.g. `.venv/bin/python` vs `.venv-bd/bin/python`) or activate the appropriate virtual environment before running commands.

---

## 3. Configuration Reference

All settings are configured via environment variables. The table below lists the essential variables:

| Variable | Default (Local Dev) | Description |
|---|---|---|
| `QGENAI_ENV` | `development` | Environment mode (`development`, `testing`, `production`). |
| `QGENAI_DATABASE_URL` | `sqlite:///./data/qgenai.db` | SQLAlchemy connection string for Module B/D SQLite database. |
| `QGENAI_STORAGE_DIR` | `./data/storage` | Local directory for uploaded inspection images. |
| `QGENAI_MODELS_DIR` | `./models` | Directory holding detection model weights. |
| `QGENAI_MOCK_DETECTOR` | `true` | Enables deterministic mock detector when weights are absent. |
| `QGENAI_JWT_SECRET_KEY` | *(32+ char secret)* | Secret key for signing Argon2id/JWT access tokens. |
| `QGENAI_AUDIT_CHECKPOINT_KEY` | *(32+ char secret)* | Secret HMAC key for periodic signed audit checkpoints. |
| `RCA_APPROVED_DOCS_DIR` | `documents/approved` | Directory containing approved SOP documents and sidecar metadata. |
| `RCA_INDEX_DIR` | `data/rag_index` | Directory storing persisted vector embeddings and chunks. |
| `RCA_EMBEDDING_BACKEND` | `hashing` | Embedding backend (`hashing` or `sentence_transformers`). |
| `RCA_SUPPORTED_CLASSES` | `crack,scratch,dent,corrosion` | Comma-delimited list of accepted defect classes. |
| `RCA_AUDIT_SINK` | `jsonl` | Standalone audit sink (`jsonl`, `none`, or `module_d`). |
| `RCA_AUDIT_FAIL_MODE` | `fail_closed` | Audit failure policy: `fail_closed` (HTTP 500) or `fail_open` (warning). |

---

## 4. Starting and Stopping the Platform

### Method A: Automated Multi-Service Orchestrator (Recommended)

Run the local orchestration script from the repository root:
```bash
./scripts/run_platform.sh
```

**What it does:**
1. Validates that both `.venv` and `.venv-bd` exist.
2. Initializes required data directories (`data/storage`, `data/rag_index`, `data/audit`, `documents/approved`).
3. Exports safe local development configuration variables.
4. Starts Module B/D on `http://127.0.0.1:8000` via `.venv-bd`.
5. Starts Module C on `http://127.0.0.1:8001` via `.venv`.
6. Traps `SIGINT` (Ctrl+C), `SIGTERM`, and `EXIT` to ensure both child processes terminate cleanly.

**To stop:** Press `Ctrl+C`. The script will gracefully terminate both services.

### Method B: Manual Service Execution

**1. Start Module B/D (Port 8000):**
```bash
cd module_b_d
export QGENAI_MOCK_DETECTOR=true
export QGENAI_JWT_SECRET_KEY="local-dev-jwt-secret-key-32-chars-min-secure"
export QGENAI_AUDIT_CHECKPOINT_KEY="local-dev-audit-checkpoint-hmac-32-chars-min"
../.venv-bd/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
```

**2. Start Module C (Port 8001):**
```bash
cd /path/to/defect-inspection
export RCA_AUDIT_SINK=jsonl
export RCA_APPROVED_DOCS_DIR=documents/approved
export RCA_INDEX_DIR=data/rag_index
.venv/bin/uvicorn module_c.api:create_app --factory --host 127.0.0.1 --port 8001 --log-level info
```

---

## 5. Service URLs and API Documentation

Once services are running, the interactive documentation is available at:

- **Module B/D (Inspection & Governance):**
  - Interactive OpenAPI UI: `http://127.0.0.1:8000/docs`
  - ReDoc UI: `http://127.0.0.1:8000/redoc`
  - Reconciled RCA Request Endpoint: `GET http://127.0.0.1:8000/inspections/{inspection_id}/rca-request`
- **Module C (GenAI RCA):**
  - Interactive OpenAPI UI: `http://127.0.0.1:8001/docs`
  - Health Endpoint: `GET http://127.0.0.1:8001/health`
  - RCA Analysis Endpoint: `POST http://127.0.0.1:8001/rca`

---

## 6. Verification and Testing Runbook

### 1. Standalone Four-Module End-to-End Test
Executes the full pipeline from Module A synthetic generation to Module D cryptographic chain verification in isolated mock mode:
```bash
.venv-bd/bin/python scripts/e2e_platform_test.py
```
*Expected Result:* All 8 steps PASS.

### 2. Existing Smoke Test (Module A + Module C)
```bash
PYTHONPATH=. .venv/bin/python scripts/end_to_end_test.py
```
*Expected Result:* `ALL END-TO-END CHECKS COMPLETED SUCCESSFULLY!`.

### 3. Full Module A & C Unit & Contract Test Suite
```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```
*Expected Result:* `96 passed` (0 failures).

### 4. Full Module B & D Test Suite
```bash
cd module_b_d
../.venv-bd/bin/pytest tests/ -v
```
*Expected Result:* `195 passed, 1 skipped` (`test_real_model.py` skipped due to absent `.pt` weights).

---

## 7. Troubleshooting Guide

### Issue 1: Port Collision (`Address already in use`)
- **Symptom:** `uvicorn` fails with `[Errno 48] Address already in use`.
- **Resolution:** Check which process is bound to port 8000 or 8001 and terminate it:
  ```bash
  lsof -i :8000
  lsof -i :8001
  kill -9 <PID>
  ```

### Issue 2: `ModuleNotFoundError: No module named 'sqlalchemy'` in `.venv`
- **Symptom:** Pytest fails when searching for `sqlalchemy` while using `.venv`.
- **Cause:** `.venv` is dedicated to Module A and Module C and does not have SQLAlchemy installed.
- **Resolution:**
  - For root tests (Module A and C), run: `PYTHONPATH=. .venv/bin/pytest tests/ -v` (do not run against `module_b_d/tests/`).
  - For Module B and D tests, run from inside `module_b_d`: `cd module_b_d && ../.venv-bd/bin/pytest tests/ -v`.

### Issue 3: Cryptographic Audit Chain Failure (`verify_chain().valid is False`)
- **Symptom:** `verify_chain(db)` returns `valid=False` with `failure_reason="Content altered..."` or `"Broken link..."`.
- **Cause:** An audit record was modified, deleted, or inserted out of order without updating the SHA-256 hash chain.
- **Resolution:**
  - Inspect `AuditEvent` records around `v_res.first_failure_seq`.
  - In development, reset the SQLite database (`rm data/qgenai.db`) and re-initialize with `init_db()`.
  - In production, investigate security events immediately as this indicates unauthorized database modification.

### Issue 4: `422 Unprocessable Entity` on `GET /rca-request`
- **Symptom:** Client receives HTTP 422 with `unsupported_classes` or `exceeds image size`.
- **Cause:**
  - The inspection detected a defect class not supported by Module C (e.g. `unknown_defect` vs `scratch`, `crack`, `dent`, `corrosion`).
  - Bounding box coordinates exceed the dimensions of the ingested image.
- **Resolution:** Ensure detector classes match Module C's configured `RCA_SUPPORTED_CLASSES`, and that bounding boxes are in original-image pixel coordinates within `[0, image_width]` and `[0, image_height]`.

### Issue 5: RCA Returns `status: "insufficient_evidence"` (HTTP 200)
- **Symptom:** `POST /rca` returns HTTP 200 with `status: "insufficient_evidence"` and empty `possible_causes`.
- **Cause:** This is expected behavior when no approved document in the knowledge base covers the requested defect class with relevance $\ge \text{min\_score}$.
- **Resolution:**
  - Check whether an approved SOP for that defect class exists in `documents/approved/`.
  - Verify that the document metadata sidecar `<filename>.meta.json` is present and valid.
  - Re-run document ingestion: `python -m module_c.retrieval`.
