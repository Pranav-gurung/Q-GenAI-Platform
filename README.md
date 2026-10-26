# Q-GenAI

**Computer-vision-first quality intelligence for quick-commerce dark stores.**

> **Status: Part 1 of 7, Foundation + Pipeline Architecture.**
> The inspection pipeline, API, database, image handling and operator UI are implemented.
> **No AI is implemented yet.** Every AI module is a placeholder that reports `NOT_IMPLEMENTED`
> and produces no detections, scores or decisions. Inspections are routed to human review.

---

## Problem

In a quick-commerce dark store (Blinkit / Zepto / Swiggy Instamart-style operations), every product
passes through a fast, high-volume fulfillment flow:

```
Supplier → RECEIVING → STORAGE → PICKING → PACKING → DISPATCH → DELIVERY → Customer
```

Products get damaged or mishandled along the way: crushed or torn packets, leaking pouches,
broken seals, dented cans, damaged boxes, wrong or unlabelled items. Today this is
caught (if at all) by eye, inconsistently, and with no record of **where** it happened.

Q-GenAI aims to answer, for every inspection:

1. **What** happened? (defect type, from computer vision)
2. **Where** in the fulfillment flow did it happen?
3. **How severe** is it? (PASS / FAIL / REVIEW)
4. **What** should the operator do? (corrective action / SOP)
5. Can the system **learn** from the operator's correction?

> Q-GenAI is a generic platform. It has no access to, or integration with, any proprietary
> Blinkit/Zepto/Instamart systems or data. Development uses **simulated sample data**
> (`data/samples/`).

## Solution: a pipeline, not a model

The central object is the **InspectionEvent**. Each one flows through a state machine driven by
a **PipelineOrchestrator**. Each AI capability is a plug-in module behind a typed interface.

```
                   USER / OPERATOR
                         │
                         ▼
                ┌─────────────────┐
                │ Inspection API  │   FastAPI        (Part 1 ✅)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ InspectionEvent │   + state machine + PipelineContext   (Part 1 ✅)
                └────────┬────────┘
                         ▼   PipelineOrchestrator  (Part 1 ✅)
                ┌─────────────────┐
                │ Image Quality   │   ImageQualityGate   → stub  (Part 3)
                │ Gate            │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Computer Vision │   CVEngine           → stub  (Part 2)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Decision Engine │   DecisionEngine     → stub  (Part 2/4)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Fulfillment     │   PipelineAnalyzer   → stub  (Part 4)
                │ Analysis        │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ RCA / GenAI     │   RCAEngine          → stub  (Part 5)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ SOP / Corrective│   ActionEngine       → stub  (Part 5)
                │ Action          │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Operator        │   POST /feedback     (Part 1 ✅: stored for learning)
                │ Feedback        │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Active Learning │   (Part 6)
                └────────┬────────┘
                         └──────────► CV MODEL
```

## What Part 1 implements

| Area | Implemented | Where |
|---|---|---|
| Configuration (env vars / `.env`, dev/test/prod) | ✅ | `backend/core/config.py`, `.env.example` |
| Structured logging (console + JSON, per-stage pipeline logs) | ✅ | `backend/core/logging.py` |
| Consistent error envelope, no stack traces to clients | ✅ | `backend/core/exceptions.py`, `backend/api/errors.py` |
| Fulfillment stages (single source of truth) | ✅ | `backend/domain/fulfillment.py` |
| Pipeline states + **state machine** (illegal transitions rejected) | ✅ | `backend/pipeline/states.py`, `transitions.py` |
| `PipelineContext` (typed accumulation of all module results) | ✅ | `backend/pipeline/context.py` |
| `PipelineOrchestrator` (stage execution, routing, timing, failure handling) | ✅ | `backend/pipeline/orchestrator.py` |
| Module interfaces + registry (swap stub → real model by env var) | ✅ | `backend/pipeline/interfaces.py`, `registry.py` |
| Placeholder modules (explicit `NOT_IMPLEMENTED`) | ✅ | `backend/pipeline/stubs.py` |
| Result contracts (CV, decision, RCA, action, …) | ✅ | `backend/schemas/results.py` |
| Database (7 tables) + Alembic migrations, PostgreSQL / SQLite | ✅ | `backend/db/`, `alembic.ini` |
| Image validation (type, size, integrity, pixel-bomb guard) | ✅ | `backend/services/image_validation.py` |
| Image storage abstraction (local; S3/GCS/Azure-ready) | ✅ | `backend/services/storage/` |
| REST API + OpenAPI docs | ✅ | `backend/api/` → `/api/docs` |
| Operator feedback (stored for future active learning) | ✅ | `backend/services/feedback_service.py` |
| Minimal operator UI | ✅ | `frontend/` |
| Docker Compose (Postgres + API) | ✅ | `docker-compose.yml`, `docker/` |
| Tests (unit, API, integration; SQLite + optional PostgreSQL) | ✅ | `tests/` |

**Not implemented (by design):** CV model / YOLO, image-quality algorithms, decision rules,
fulfillment analysis, RCA, RAG / vector DB, LLMs, SOP generation, active-learning training,
analytics dashboards, user accounts / RBAC.

### Honest placeholders: "no result" is never a fake result

* Every module result carries `status`: `COMPLETED | NOT_IMPLEMENTED | SKIPPED | FAILED`.
* The result schemas **reject** payload values (confidence, detections, outcome, …) on anything that
  is not `COMPLETED`, so a stub *cannot* emit a fake "94% confidence".
* An inspection without a **real** decision never auto-passes or auto-fails. It ends in
  `REQUIRES_REVIEW` (`NO_AUTOMATED_DECISION`) and an operator resolves it through feedback.
* `GET /api/health` reports for each module whether it is `implemented`.

## Architecture

```
q-genai/
├── backend/
│   ├── main.py               app factory (wiring only)
│   ├── api/                  HTTP layer: routes/, dependencies/ (db, services, api-key), errors.py
│   ├── core/                 config, structured logging, exceptions
│   ├── domain/               fulfillment stages, result enums (DefectType, Severity, ...)
│   ├── pipeline/             states, transitions (state machine), context, interfaces,
│   │                         stubs, registry, orchestrator   ← the backbone
│   ├── schemas/              Pydantic contracts: InspectionEvent, results, feedback, errors
│   ├── services/             use cases: inspections, feedback, reference data,
│   │                         image validation, storage/
│   └── db/                   SQLAlchemy models, session, Alembic migrations
├── ml/                       future model implementations (preprocessing/, detection/, inference/)
├── frontend/                 static operator UI (no build step), served at /
├── data/                     raw/ processed/ feedback/ (git-ignored), samples/ (simulated data)
├── documents/                future SOP corpus for RAG (Part 5)
├── scripts/                  seed_sample_data.py, smoke_test.sh
├── docker/                   Dockerfile, entrypoint (migrate → seed → serve)
└── tests/                    unit/ api/ integration/
```

**Dependency rules**

```
api  →  services  →  pipeline (orchestrator → interfaces)  ←  ml (future implementations)
              ↘  db, storage
```

* The **API** layer only handles HTTP (validation, status codes). No business logic.
* **Services** are the only place that combines the database, storage and the pipeline.
* The **pipeline** is pure domain code. It has no FastAPI or SQLAlchemy imports and is testable in isolation.
* **ML modules** implement `backend.pipeline.interfaces` and are selected in `backend/pipeline/registry.py`.
  The backend never imports a model directly. YOLO, ONNX Runtime, TensorRT, OpenCV, local or
  API LLMs all fit behind the same contracts.

### Pipeline state machine

```
CREATED → IMAGE_RECEIVED → QUALITY_CHECK → CV_PROCESSING → DECISION → PIPELINE_ANALYSIS
        → RCA_PROCESSING → ACTION_GENERATED → COMPLETED
```

| From | Allowed to |
|---|---|
| CREATED | IMAGE_RECEIVED, FAILED |
| IMAGE_RECEIVED | QUALITY_CHECK, FAILED |
| QUALITY_CHECK | CV_PROCESSING, REQUIRES_REVIEW (image rejected), FAILED |
| CV_PROCESSING | DECISION, REQUIRES_REVIEW, FAILED |
| DECISION | PIPELINE_ANALYSIS, **COMPLETED** (confident PASS: no RCA needed), REQUIRES_REVIEW, FAILED |
| PIPELINE_ANALYSIS | RCA_PROCESSING, REQUIRES_REVIEW, FAILED |
| RCA_PROCESSING | ACTION_GENERATED, REQUIRES_REVIEW, FAILED |
| ACTION_GENERATED | COMPLETED, REQUIRES_REVIEW, FAILED |
| REQUIRES_REVIEW | COMPLETED (operator feedback), IMAGE_RECEIVED (retake), FAILED |
| COMPLETED / FAILED | (terminal) |

Anything else, e.g. `CREATED → COMPLETED`, raises `INVALID_STATE_TRANSITION` (HTTP 409 at the API).
The processing states mean "this stage is running". For example, `ACTION_GENERATED` is the state in
which the action engine runs.

**Orchestrator routing (real results only):** quality gate rejects → `REQUIRES_REVIEW`;
decision PASS → `COMPLETED`; decision REVIEW → `REQUIRES_REVIEW`; decision FAIL with an action →
`COMPLETED`; no real decision → `REQUIRES_REVIEW`; module raises or returns the wrong type → `FAILED`
(`MODULE_ERROR` / `MODULE_CONTRACT_VIOLATION`); module raises `NotImplementedError` → recorded as
`NOT_IMPLEMENTED`.

### Module contracts

| Interface | Contract | Planned |
|---|---|---|
| `ImageQualityGate.assess` | `ImageInput → ImageQualityResult` | Part 3 |
| `CVEngine.detect` | `ImageInput → CVResult` (detections: bbox, defect type, confidence, severity; model version; time) | Part 2 |
| `DecisionEngine.decide` | `CVResult + InspectionInfo → QualityDecision` (PASS/FAIL/REVIEW, reason, confidence, severity, source) | Part 2/4 |
| `PipelineAnalyzer.analyze` | `PipelineContext → PipelineAnalysis` (suspected stage, likelihoods, evidence) | Part 4 |
| `RCAEngine.analyze` | `PipelineContext → RCAResult` (possible causes, evidence, confidence, investigation) | Part 5 |
| `ActionEngine.recommend` | `RCAResult + context → ActionRecommendation` (action, priority, responsible stage, instructions) | Part 5 |
| Feedback | `InspectionResult + operator feedback → FeedbackRecord` (`learning_status=PENDING`) | Part 1 store / Part 6 learn |

Plugging in a real CV model in Part 2 looks like this:

```python
# ml/detection/yolo_engine.py
class YoloCVEngine(CVEngine):
    name = "yolov8"
    def detect(self, image: ImageInput) -> CVResult: ...

# backend/pipeline/registry.py
CV_ENGINES["yolov8"] = lambda s: YoloCVEngine(weights=s.model_path / "best.onnx")
```

Then set `CV_ENGINE=yolov8`. No API, database or orchestrator changes are needed.

### Data model

| Table | Answers |
|---|---|
| `stores`, `skus` | Which store? Which SKU? |
| `inspections` | What was inspected, at which fulfillment stage, when, current pipeline status, final decision |
| `inspection_images` | Which image (file name, MIME type, size, dimensions, hash, storage reference) |
| `pipeline_events` | Append-only history: every transition, module outcome (+ duration), error |
| `inspection_results` | Each module's result (JSON, including its status) + model versions that produced it |
| `feedback` | What the operator said / corrected, vs. what the system decided, model versions, learning status |

Enum-like columns (stage, status, …) are stored as `VARCHAR`, so adding a fulfillment stage needs no
schema migration.

## Local setup

Requirements: Python 3.11+, and either PostgreSQL 14+ or nothing (SQLite fallback). Docker is optional.

```bash
git clone https://github.com/Pranav-gurung/q-genai-platform.git
cd q-genai-platform
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # then edit DATABASE_URL (PostgreSQL or SQLite)
```

### Option A: SQLite (zero setup)

```bash
# in .env:  DATABASE_URL=sqlite:///./data/qgenai-dev.db
alembic upgrade head
python -m scripts.seed_sample_data
uvicorn backend.main:create_app --factory --reload
```

### Option B: PostgreSQL

```bash
createuser qgenai -P && createdb qgenai -O qgenai     # or use any existing server
# in .env:  DATABASE_URL=postgresql+psycopg://qgenai:<password>@localhost:5432/qgenai
alembic upgrade head
python -m scripts.seed_sample_data
uvicorn backend.main:create_app --factory --reload
```

### Option C: Docker Compose (PostgreSQL + API)

```bash
docker compose up --build         # migrates, seeds sample data, serves on :8000
```

Then open:

* UI: <http://localhost:8000/>
* API docs (Swagger): <http://localhost:8000/api/docs>
* Health: <http://localhost:8000/api/health>

End-to-end smoke test against a running server: `scripts/smoke_test.sh http://localhost:8000`

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | `development` / `test` / `production` (production forbids SQLite and `CORS_ORIGINS=*`) |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `console` | `console` (human) or `json` (log shippers) |
| `API_HOST` / `API_PORT` / `API_PREFIX` | `0.0.0.0` / `8000` / `/api` | Server binding and URL prefix |
| `CORS_ORIGINS` | `http://localhost:8000,…` | Comma-separated allowed origins |
| `SERVE_FRONTEND` | `true` | Serve `frontend/` at `/` |
| `API_KEY` | *(empty)* | If set, POST endpoints require header `X-API-Key` (stop-gap, not user auth) |
| `DATABASE_URL` | `sqlite:///./data/qgenai-dev.db` | SQLAlchemy URL (`postgresql+psycopg://…` recommended) |
| `SEED_SAMPLE_DATA` | `false` | Seed simulated stores/SKUs on startup when the DB is empty |
| `STORAGE_BACKEND` / `STORAGE_PATH` | `local` / `data/raw/images` | Image storage |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `ALLOWED_IMAGE_TYPES` | `jpeg,png,webp` | Accepted formats (checked against decoded content) |
| `MAX_IMAGE_PIXELS` | `40000000` | Decompression-bomb guard |
| `QUALITY_GATE_ENGINE`, `CV_ENGINE`, `DECISION_ENGINE`, `PIPELINE_ANALYSIS_ENGINE`, `RCA_ENGINE`, `ACTION_ENGINE` | `stub` | Module implementation selection |
| `MODEL_PATH` | `models/` | Model weights location (used from Part 2) |

Secrets live only in `.env` or the deployment environment. `.env` is git-ignored and docker-ignored.

## API overview

All endpoints are under `/api`. Interactive docs with request/response/error schemas are at `/api/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | DB/storage status, version, which pipeline modules are implemented |
| GET | `/api/reference-data` | Stores, SKUs, fulfillment stages, pipeline states (for UIs) |
| POST | `/api/inspections` | Create inspection → `CREATED` |
| GET | `/api/inspections` | List (filters: `store_id`, `sku_id`, `stage`, `status`; `limit`, `offset`) |
| GET | `/api/inspections/{id}` | Inspection + module results + pipeline history |
| POST | `/api/inspections/{id}/image` | Upload image (multipart `file`) → validate, store, run pipeline |
| POST | `/api/inspections/{id}/feedback` | Operator verdict/corrections (resolves `REQUIRES_REVIEW`) |

```bash
curl -X POST localhost:8000/api/inspections -H 'content-type: application/json' \
  -d '{"store_id":"DS-BLR-001","sku_id":"SKU-MILK-500ML","stage":"PACKING","order_id":"ORD-100245"}'
curl -X POST localhost:8000/api/inspections/INS-…/image -F "file=@pouch.jpg"
curl -X POST localhost:8000/api/inspections/INS-…/feedback -H 'content-type: application/json' \
  -d '{"operator_id":"OP-17","outcome":"FAIL","defect_types":["LEAKING_PACKAGE"],"suspected_stage":"PICKING"}'
```

**Errors** always look like this (stack traces are only logged, never returned):

```json
{"success": false,
 "error": {"code": "UNSUPPORTED_IMAGE_TYPE", "message": "Unsupported file extension '.gif'",
           "details": {"allowed_extensions": [".jpeg", ".jpg", ".png", ".webp"]}, "request_id": "9bc8f3…"}}
```

Codes: `VALIDATION_ERROR`, `UNKNOWN_STORE`, `UNKNOWN_SKU`, `INSPECTION_NOT_FOUND`,
`INVALID_STATE_TRANSITION`, `FEEDBACK_NOT_ALLOWED`, `UNSUPPORTED_IMAGE_TYPE`, `IMAGE_TOO_LARGE`,
`CORRUPT_IMAGE`, `EMPTY_FILE`, `UNAUTHORIZED`, `INTERNAL_ERROR`.

## Logging

Pipeline logs carry `inspection_id`, `pipeline_stage`, `status`, `duration_ms`, `error_code`, and a `request_id` per HTTP request:

```
2026-10-26T20:25:35.215+00:00 INFO  INS-8300E6F9D5DF4683 | CV_PROCESSING | STARTED | pipeline_module=stub
2026-10-26T20:25:35.215+00:00 INFO  INS-8300E6F9D5DF4683 | CV_PROCESSING | NOT_IMPLEMENTED | pipeline_module=stub duration_ms=0.02
2026-10-26T20:25:35.216+00:00 INFO  INS-8300E6F9D5DF4683 | PIPELINE | REQUIRES_REVIEW | duration_ms=0.56 error_code=NO_AUTOMATED_DECISION
```

Set `LOG_FORMAT=json` for one JSON object per line. Image bytes and secrets are never logged.

## Tests

```bash
pytest                                   # unit + API + integration on SQLite
TEST_DATABASE_URL=postgresql+psycopg://qgenai:<pw>@localhost:5432/qgenai_test pytest tests/integration
                                         # also runs migrations + persistence on a disposable PostgreSQL DB
```

The tests cover domain validation, every legal and illegal state transition, the orchestrator (stub flow
plus routing, failure and contract-violation cases using test-only doubles), image validation (valid,
wrong extension, mismatched content, oversized, pixel bomb, corrupt, empty), storage path-traversal
protection, configuration guards, all API endpoints, the error envelope, optional API-key auth,
Alembic migrations, and a full persisted lifecycle.

## Roadmap

| Part | Scope | Status |
|---|---|---|
| **1** | Foundation + pipeline architecture | ✅ this commit |
| 2 | Computer Vision (dataset, YOLO training, ONNX inference as `CVEngine`, decision rules) | planned |
| 3 | Image Quality Gate (blur, exposure, framing, retake loop) | planned |
| 4 | Fulfillment Intelligence (where in the flow damage happened, stage analytics) | planned |
| 5 | GenAI RCA + RAG (SOP retrieval, grounded root cause, corrective actions) | planned |
| 6 | Active Learning (feedback review → labelled data → retraining → model versions) | planned |
| 7 | Minimal security + production hardening (operator auth/roles, async workers, object storage) | planned |

## Known limitations (Part 1)

* No AI: every inspection ends in `REQUIRES_REVIEW` until an operator submits feedback.
* The pipeline runs synchronously inside the upload request. That's fine for stubs, but real models will
  need a background worker or queue (Part 7). The orchestrator is already HTTP-independent.
* Pipeline history is persisted after the run, not incrementally. A process crash mid-run would
  leave the inspection in `IMAGE_RECEIVED`.
* `API_KEY` is a single shared key: a placeholder, not user authentication.
* Only local image storage is implemented. The interface is ready for S3, GCS or Azure Blob.
* Stores and SKUs are simulated sample data, and there is no API for managing them.
