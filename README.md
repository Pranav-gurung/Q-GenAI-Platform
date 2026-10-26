# Q-GenAI Platform — Industrial Defect Inspection & Root-Cause Analysis

The **Q-GenAI Platform** is an integrated system for automated industrial defect inspection, quality policy governance, and document-grounded Root-Cause Analysis (RCA).

---

## 1. System Architecture & Modules

The platform unites four specialized modules:

```
┌─────────────────────────────────┐
│  Module A: Synthetic Data       │
│  - Generation & Ingestion       │
│  - Validation & Group Splitting │
│  - Train-Only Augmentation      │
│  - YOLO / COCO / Anomaly Export │
└────────────────┬────────────────┘
                 │
                 ▼  (Handover Gate: verify_handover())
┌─────────────────────────────────┐        ┌──────────────────────────────────┐
│  Module B: Vision Inspection    │        │  Module D: Governance & Ledger   │
│  - Image Ingestion & Hashing    │◄──────►│  - JWT Authentication & RBAC     │
│  - YOLO Defect Detection (Mock) │        │  - Versioned Quality Policies    │
│  - Bounding Box Normalization   │        │  - SHA-256 Hash-Chained Ledger   │
└────────────────┬────────────────┘        └──────────────────▲───────────────┘
                 │                                            │
                 ▼  (Contract Adapter: build_module_c_rca_request())
┌─────────────────────────────────┐                           │
│  Module C: GenAI RCA (RAG)      │                           │
│  - Approved SOP Extraction      │                           │
│  - Vector Retrieval & Filtering │                           │
│  - Grounded LLM RCA Generation  │                           │
│  - Citations & Unknowns Output  │                           │
└────────────────┬────────────────┘                           │
                 │                                            │
                 └──────── ModuleDAuditAdapter ───────────────┘
```

| Module | Name | Directory | Primary Role |
|---|---|---|---|
| **Module A** | Synthetic Data Engine | `module_a/` | Procedural fixture synthesis, validation, group/part-aware splitting, train-only bounding-box augmentation, format exporters (YOLO, COCO, Anomaly), manifests, and 4-eyes feedback workflow. |
| **Module B** | Vision Defect Inspection | `module_b_d/app/` | Image upload (`POST /predict`), deterministic defect detection, bounding-box spatial parsing, and quality policy decision evaluation. |
| **Module C** | GenAI Root-Cause Analysis | `module_c/` | Approved manufacturing SOP ingestion (PDF/TXT), vector chunking, semantic retrieval, and document-grounded RCA suggestions with citation tracking. |
| **Module D** | Security & Governance | `module_b_d/app/services/` | Argon2id/JWT authentication, role-based access control (Operator, Quality Manager, Admin), versioned quality rules, and a tamper-evident SHA-256 hash-chained audit ledger. |

---

## 2. Environment Setup & Prerequisites

Due to distinct package requirements (e.g. PyTorch 2.14 vs 2.6.0; presence vs absence of SQLAlchemy), the repository operates with two dedicated virtual environments:

### Virtual Environment 1: `.venv` (Modules A & C)
Used for synthetic dataset engineering and the Module C RAG service.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Virtual Environment 2: `.venv-bd` (Modules B & D)
Used for the vision inspection service, quality policy engine, SQLite database, and audit ledger.
```bash
python3 -m venv .venv-bd
source .venv-bd/bin/activate
pip install -r module_b_d/requirements.txt
```

---

## 3. Running the Platform

### Option A: Automated Multi-Service Orchestrator (Recommended)

To launch both Module B/D and Module C locally with managed shutdown, run:
```bash
./scripts/run_platform.sh
```
- **Module B/D** starts on `http://127.0.0.1:8000` (Docs: `http://127.0.0.1:8000/docs`)
- **Module C** starts on `http://127.0.0.1:8001` (Docs: `http://127.0.0.1:8001/docs`)
- Press `Ctrl+C` to cleanly terminate both services via POSIX process traps.

### Option B: Starting Services Manually

**1. Launch Module B/D (Port 8000):**
```bash
cd module_b_d
export QGENAI_MOCK_DETECTOR=true
export QGENAI_JWT_SECRET_KEY="local-dev-jwt-secret-key-32-chars-min-secure"
export QGENAI_AUDIT_CHECKPOINT_KEY="local-dev-audit-checkpoint-hmac-32-chars-min"
../.venv-bd/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**2. Launch Module C (Port 8001):**
```bash
export RCA_AUDIT_SINK=jsonl
export RCA_APPROVED_DOCS_DIR=documents/approved
export RCA_INDEX_DIR=data/rag_index
.venv/bin/uvicorn module_c.api:create_app --factory --host 127.0.0.1 --port 8001
```

---

## 4. End-to-End Data Flow

1. **Dataset Synthesis & Export (Module A)**:
   Module A generates synthetic fixture images and annotations, performs group-aware splitting to prevent split leakage, applies bounding-box transformations only to training data, and exports a YOLO dataset with a verifiable SHA-256 manifest.
2. **Handover Validation Gate (A $\rightarrow$ B)**:
   [`training/handover.py`](module_b_d/training/handover.py) executes `verify_handover(yolo_dir, manifest_path)` to ensure class IDs, label bounds, normalized coordinates, and image hashes match across splits with zero image or group leakage.
3. **Inspection & Quality Decision (Module B & D)**:
   An operator uploads a part image via `POST /predict`. The image is SHA-256 hashed and stored; the detector produces bounding boxes and confidences; Module D evaluates the active quality policy (`PASS`, `FAIL`, or `REVIEW`) and appends the initial event to the audit ledger.
4. **Contract Reconciliation (B $\rightarrow$ C)**:
   The operator queries `GET /inspections/{id}/rca-request`. The service reconciles Module B's database record into Module C's canonical schema:
   - Sets `image_id` to `insp.image_sha256` (content-addressable).
   - Maps bounding box coordinates `{x1, y1, x2, y2}` to `[x_min, y_min, x_max, y_max]` in original-image pixel space with bounds validation.
   - Converts inspection outcome to `defect_detected` / `nominal` / `review_required`.
   - Packages operational metadata (`site`, `policy_version`, `model_sha256`) inside `inspection_context` without violating strict schema key restrictions.
   - Rejects unsupported defect classes cleanly with HTTP 422.
5. **Knowledge Retrieval & Grounded RCA (Module C)**:
   Module C ingests approved manufacturing SOPs (`documents/approved/`), extracts chunks, and performs vector search against the defect query. If no evidence meets the threshold, it returns `status: "insufficient_evidence"` without an LLM call. If evidence exists, it prompts the LLM to return strict JSON containing only valid evidence chunk IDs.
6. **Cryptographic Audit Ledger (Module D)**:
   [`ModuleDAuditAdapter`](module_b_d/app/services/audit_adapter.py) intercepts Module C's `AuditEvent` and persists an `rca.completed` or `rca.failed` record into Module D's SQLite ledger. Every record links monotonically to the previous record's SHA-256 hash. `verify_chain()` cryptographically audits the ledger to verify that zero tampering has occurred.

---

## 5. Verification & Testing

### 1. Four-Module End-to-End Test (CLI Runner)
Runs the complete cross-module integration test across all 4 modules:
```bash
.venv-bd/bin/python scripts/e2e_platform_test.py
```

### 2. Existing Smoke Test
```bash
PYTHONPATH=. .venv/bin/python scripts/end_to_end_test.py
```

### 3. Module A & C Test Suite (`.venv`)
```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

### 4. Module B & D Test Suite (`.venv-bd`)
```bash
cd module_b_d
../.venv-bd/bin/pytest tests/ -v
```

---

## 6. Documented Limitations & Operational Boundaries

To maintain technical integrity, the following architectural boundaries are explicitly documented:

1. **Mock Detector Mode**: End-to-end integration and inspection tests operate in deterministic mock mode (`FakeDetector`). No actual model weights (`yolov8n.pt`) are present in the repository, and no real-model inference or industrial precision/recall has been evaluated.
2. **Grounded Mock LLM**: Automated integration tests employ a deterministic `GroundedMockProvider`. No external LLM providers (Anthropic Claude, OpenAI, etc.) or paid API calls were used during Phase 6 verification.
3. **Synthetic Validation**: The synthetic datasets produced by Module A demonstrate procedural pipeline correctness and zero data leakage. They do not constitute a benchmark for industrial production performance.
4. **Environment Separation**:
   - In **standalone microservice mode**, Module C operates in `.venv` with its native JSONL audit sink, while Module B/D operates in `.venv-bd`.
   - In **consolidated test mode**, `ModuleDAuditAdapter` bridges Module C directly to Module D's SQLite session factory inside `.venv-bd`.
5. **Operations Runbook**: For troubleshooting, port conflict resolution, and detailed instructions, refer to [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
