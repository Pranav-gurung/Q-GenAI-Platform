# Q-GenAI

**Computer-vision-first quality intelligence for quick-commerce dark stores.**

> **Status: Part 5 of 7, RAG / SOP Intelligence** (on top of Parts 1-4: foundation + pipeline, computer
> vision, fulfillment pipeline intelligence, GenAI RCA). Implemented: inspection pipeline, API, database, operator
> UI, an image quality gate, a **real YOLOv8 CV engine**, deterministic decision rules, dataset / training /
> evaluation tooling, **deterministic first-defect stage attribution**, an **evidence-grounded GenAI RCA** that
> only adds labelled hypotheses, actions and checks, and **SOP-grounded corrective actions** retrieved from a
> DEMO / SYNTHETIC knowledge base with citations.
> **The only trained model is a demo model trained on SYNTHETIC data.** It proves the engineering end to end.
> It is **not** evidence of accuracy on real product photos (see [Computer Vision](#computer-vision-part-2)).
> The SOP knowledge base is demo content, not any company's real SOPs. Active learning is not implemented yet.

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
                │ Image Quality   │   ImageQualityGate   → basic (Part 2 ✅; advanced: Part 3)
                │ Gate            │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Computer Vision │   CVEngine           → yolov8 (Part 2 ✅)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Decision Engine │   DecisionEngine     → rules (Part 2 ✅)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Fulfillment     │   PipelineAnalyzer   → deterministic (Part 3 ✅)
                │ Analysis        │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ RCA / GenAI     │   RCAEngine          → genai (Part 4 ✅)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ SOP / Corrective│   ActionEngine       → rca + RAG SOP grounding (Part 5 ✅)
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

## What is implemented

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
| **Part 2:** YOLOv8 CV engine (`CV_ENGINE=yolov8`), model versioning, CPU/GPU | ✅ | `ml/cv/yolo_engine.py`, `model.py` |
| **Part 2:** Image quality gate (`QUALITY_GATE_ENGINE=basic`) | ✅ | `ml/cv/quality.py` |
| **Part 2:** Deterministic decision rules (`DECISION_ENGINE=rules`) | ✅ | `backend/pipeline/decision_rules.py` |
| **Part 2:** Defect taxonomy, severity mapping, postprocessing | ✅ | `ml/cv/taxonomy.yaml`, `severity.py`, `postprocess.py` |
| **Part 2:** Dataset validator, synthetic demo data, training, evaluation, predict, visualize, benchmark, ONNX export | ✅ | `ml/cv/*.py` |
| **Part 3:** Fulfillment stage history + deterministic first-defect attribution (`PIPELINE_ANALYSIS_ENGINE=deterministic`), `GET /api/inspections/{id}/pipeline`, UI timeline | ✅ | `backend/fulfillment/`, `backend/services/fulfillment_service.py` |
| **Part 4:** Evidence-grounded GenAI RCA (`RCA_ENGINE=genai`, `ACTION_ENGINE=rca`), LLM provider abstraction (Anthropic + mock), guardrails, `GET /api/inspections/{id}/rca`, UI panel | ✅ | `backend/rca/`, `backend/services/rca_service.py` |
| **Part 5:** RAG / SOP intelligence (`RAG_ENABLED`): demo SOP knowledge base, ingestion + validation, section chunking, embeddings, local vector index, retrieval, cited SOP-grounded actions, citation guardrails, health, UI | ✅ | `knowledge/`, `backend/rag/`, `backend/schemas/rag.py` |

**Not implemented yet (by design):** a vector database (a local numpy index is enough here), real approved SOPs,
active-learning training, analytics dashboards, user accounts / RBAC. Also not implemented: any model trained
on real product photos.

### Honest placeholders: "no result" is never a fake result

* Every module result carries `status`: `COMPLETED | NOT_IMPLEMENTED | SKIPPED | UNAVAILABLE | FAILED`.
  `UNAVAILABLE` means a real module is configured but cannot run (for example, the CV model file is missing).
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
├── ml/cv/                    computer vision: taxonomy, quality gate, YOLO engine, dataset tools, training
├── frontend/                 static operator UI (no build step), served at /
├── data/                     raw/ processed/ feedback/ cv/ (git-ignored), samples/ (simulated data)
├── artifacts/cv/             trained model versions (git-ignored)
├── knowledge/                DEMO / SYNTHETIC SOP knowledge base for RAG (Part 5)
├── scripts/                  seed_sample_data.py, smoke_test.sh
├── docker/                   Dockerfile, entrypoint (migrate → seed → serve)
└── tests/                    unit/ api/ integration/ cv/
```

**Dependency rules**

```
api  →  services  →  pipeline (orchestrator → interfaces)  ←  ml/cv (implementations)
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
decision PASS → `COMPLETED`; decision REVIEW → `PIPELINE_ANALYSIS` (stage history for the reviewer) → `REQUIRES_REVIEW`; decision FAIL with an action →
`COMPLETED`; no real decision → `REQUIRES_REVIEW`; module raises or returns the wrong type → `FAILED`
(`MODULE_ERROR` / `MODULE_CONTRACT_VIOLATION`); module raises `NotImplementedError` → recorded as
`NOT_IMPLEMENTED`.

### Module contracts

| Interface | Contract | Implementations |
|---|---|---|
| `ImageQualityGate.assess` | `ImageInput → ImageQualityResult` | `stub`, **`basic`** (Part 2); advanced checks in Part 3 |
| `CVEngine.detect` | `ImageInput → CVResult` (detections: bbox, defect type, confidence, severity; inference metadata) | `stub`, **`yolov8`** (Part 2) |
| `DecisionEngine.decide` | `CVResult + InspectionInfo → QualityDecision` (PASS/FAIL/REVIEW, reason, confidence, severity, source) | `stub`, **`rules`** (Part 2) |
| `PipelineAnalyzer.analyze` | `PipelineContext → PipelineAnalysis` (structured `trace_analysis`, suspected stage, evidence codes) | `stub`, **`deterministic`** (Part 3) |
| `RCAEngine.analyze` | `PipelineContext → RCAResult` (structured `root_cause_analysis`; hypotheses, no invented confidence) | `stub`, **`genai`** (Part 4) |
| `ActionEngine.recommend` | `RCAResult + context → ActionRecommendation` (action, priority, optional responsible stage, instructions) | `stub`, **`rca`** (Part 4) |
| Feedback | `InspectionResult + operator feedback → FeedbackRecord` (`learning_status=PENDING`) | stored (Part 1); learning in Part 6 |

Modules are registered in `backend/pipeline/registry.py` and selected by env var. Part 2 plugged in the CV engine this way:

```python
# backend/pipeline/registry.py
def _yolov8_engine(settings):
    from ml.cv.yolo_engine import YoloCVEngine   # torch imported only when selected
    return YoloCVEngine(settings)

CV_ENGINES = {"stub": ..., "yolov8": _yolov8_engine}
```

No API, database or orchestrator changes were needed. A module that depends on an artifact implements `health()`, and
if it can't run it returns `UNAVAILABLE` rather than a result.

### Data model

| Table | Answers |
|---|---|
| `stores`, `skus` | Which store? Which SKU? |
| `inspections` | What was inspected (store, SKU, order, **item**), at which fulfillment stage, when, current pipeline status, final decision |
| `inspection_images` | Which image (file name, MIME type, size, dimensions, hash, storage reference) |
| `pipeline_events` | Append-only history: every transition, module outcome (+ duration), error |
| `inspection_results` | Each module's result (JSON, including its status) + model versions that produced it |
| `feedback` | What the operator said / corrected, vs. what the system decided, model versions, learning status |

Enum-like columns (stage, status, …) are stored as `VARCHAR`, so adding a fulfillment stage needs no
schema migration.

## Computer Vision (Part 2)

> **Key principle: computer vision detects visual evidence; deterministic application logic makes the
> operational decision.** The CV engine reports *what* it sees, *where*, *how confident* the model is and
> *how severe* each finding is. It never decides PASS / FAIL. `DECISION_ENGINE=rules` does that, using
> explicit, configurable thresholds.

### Architecture

```
POST /api/inspections/{id}/image                      (unchanged Part 1 API)
   │  upload validation: type, size, corruption       backend/services/image_validation.py
   ▼
storage (ImageStorage)  ──►  PipelineOrchestrator
   ▼
QUALITY_CHECK   BasicImageQualityGate                 ml/cv/quality.py
   │  blur, resolution, brightness, aspect ratio      reject → REQUIRES_REVIEW (IMAGE_QUALITY_REJECTED)
   ▼
CV_PROCESSING   YoloCVEngine (CVEngine)               ml/cv/yolo_engine.py
   │  decode + EXIF orientation → RGB → BGR           ml/cv/preprocessing.py
   │  YOLOv8 predict (letterbox, NMS)                 Ultralytics, model cached in memory
   │  validate output, map class → taxonomy,          ml/cv/postprocess.py
   │  confidence filter, clip boxes, severity         ml/cv/severity.py
   ▼  CVResult (detections + inference metadata)
DECISION        RulesDecisionEngine                   backend/pipeline/decision_rules.py
   ▼
PIPELINE_ANALYSIS → RCA → ACTION                      (still placeholders: Parts 4–5)
```

The real engine plugs in through the **existing** `CVEngine` interface and registry. There is no separate CV API.
`CV_ENGINE=stub` still works, and the backend runs without torch installed as long as `yolov8` isn't selected.

### Model / framework

* **YOLOv8n** (Ultralytics **8.4.190**, PyTorch 2.x), starting from COCO-pretrained `yolov8n.pt` and fine-tuned.
  It's the most widely used, best documented YOLO implementation, with one dependency for train / val /
  predict / export, CPU and CUDA support, and a 3.2M-parameter model that runs on CPU.
* **Licensing:** Ultralytics and its pretrained weights are **AGPL-3.0**. That's fine for this open repository.
  A closed-source / SaaS deployment needs an Ultralytics Enterprise licence, or a swap to a permissively
  licensed detector behind the same `CVEngine` interface (the postprocessing is framework-independent).

### Defect taxonomy

Defined once in `ml/cv/taxonomy.yaml`. Class ids/names, platform `DefectType`, base severity, safety-critical
flag and data status all come from it. Model class names are mapped **by name**, and a model emitting an
unknown class is rejected at load time.

| id | class | DefectType | base severity | safety-critical | data status |
|---|---|---|---|---|---|
| 0 | crushed_packaging | CRUSHED_PACKAGE | MEDIUM | – | placeholder |
| 1 | torn_packaging | TORN_PACKAGE | MEDIUM | – | **synthetic_demo** |
| 2 | leakage | LEAKING_PACKAGE | HIGH | ✔ | **synthetic_demo** |
| 3 | broken_seal | BROKEN_SEAL | HIGH | ✔ | placeholder |
| 4 | opened_package | OPENED_PACKAGE | HIGH | ✔ | placeholder |
| 5 | dented_container | DENTED_CONTAINER | MEDIUM | – | **synthetic_demo** |
| 6 | damaged_box | DAMAGED_BOX | LOW | – | **synthetic_demo** |
| 7 | missing_label | MISSING_LABEL | MEDIUM | – | **synthetic_demo** |
| 8 | wrong_item | INCORRECT_ITEM | HIGH | – | placeholder |
| 9 | visual_anomaly | VISUAL_ANOMALY | LOW | – | placeholder |

* **synthetic_demo** (5 classes): trainable and detected by the demo model, but **only on synthetic images**.
* **placeholder** (5 classes): no data exists. The demo model has never seen them, so it **cannot** detect
  them, and no metric is claimed. `wrong_item` needs SKU reference images (a matching problem, not just detection).
* **real**: none yet. No real-world dataset has been collected.

### Severity (deterministic)

```
severity = base_severity(class)
if box_area / image_area >= CV_SEVERITY_LARGE_AREA_RATIO (0.20):  severity += 1 level
if class.safety_critical:                                         severity  = max(severity, HIGH)
(capped at CRITICAL)
```

Confidence does **not** change severity. Severity describes impact; confidence describes certainty.
The decision rules combine the two. No randomness, no ML, no LLM.

### Decision rules (`DECISION_ENGINE=rules`)

| CV result | Decision |
|---|---|
| CV did not run (stub / `UNAVAILABLE` / failed) | **no decision** (`SKIPPED`): the inspection goes to `REQUIRES_REVIEW` and is never auto-passed |
| any detection with severity ≥ `DECISION_FAIL_MIN_SEVERITY` (HIGH) **and** confidence ≥ `DECISION_FAIL_MIN_CONFIDENCE` (0.5) | **FAIL** |
| other detections | **REVIEW** |
| no detections | **REVIEW** by default; **PASS** only if `DECISION_AUTO_PASS=true` |

Auto-pass is off by default. The model covers 5 of 10 classes, and only on synthetic data, so
"nothing detected" is not evidence that the product is fine.

### Detection output and coordinates

```json
{"defect_type": "LEAKING_PACKAGE", "model_class": "leakage", "confidence": 0.91, "severity": "HIGH",
 "bbox": {"x_min": 212.4, "y_min": 301.0, "x_max": 330.8, "y_max": 362.5}, "area_ratio": 0.042}
```

`bbox` is in **pixels** of the inspected image *after EXIF orientation*. The origin is top-left, x points
right and y points down, and the box is (x_min, y_min) to (x_max, y_max). Normalised 0–1 coordinates appear
only in YOLO training label files. Each CV result also carries `inference`: model name, version, framework,
artifact (`best.pt@sha256:…`), device actually used, image size, thresholds, pre/inference/post-process ms and
the raw box count. The version is also stored in `inspection_results.cv_model_version` and in `model_versions.cv`.

### Confidence threshold (`CV_CONFIDENCE_THRESHOLD`, default 0.35)

* **Lower** catches more real defects (fewer false negatives), but gives more false alarms and therefore more reviews.
* **Higher** gives fewer false alarms, but misses more defects. Missed defects aren't reviewed at all when auto-pass is on.
* Per-class overrides: add `min_confidence:` to a class in `taxonomy.yaml`.
* The `image_level` block in `metrics.json` shows the flag rate at the deployment threshold. Tune the threshold
  on **real** validation data before trusting it.

### Image quality gate (`QUALITY_GATE_ENGINE=basic`)

This gate runs after upload validation (which already checks type, size and corruption) and judges only whether
the content can be used. It's conservative, so imperfect but usable photos pass.

| Check | Default | Meaning |
|---|---|---|
| `CV_MIN_IMAGE_WIDTH` / `HEIGHT` | 128 px | smaller → `LOW_RESOLUTION` |
| `CV_MAX_ASPECT_RATIO` | 6 | long/short side above this → `EXTREME_ASPECT` |
| `CV_BLUR_THRESHOLD` | 15 | variance of Laplacian on grayscale resized to a 512 px long side; below → `BLURRY` (sharp photos are typically ≫100) |
| `CV_MIN_BRIGHTNESS` / `CV_MAX_BRIGHTNESS` | 20 / 240 | mean luminance 0–255 → `TOO_DARK` / `TOO_BRIGHT` |

A rejected image moves the inspection to `REQUIRES_REVIEW` (`IMAGE_QUALITY_REJECTED`). The operator retakes the
photo with a new upload, and CV doesn't run on the rejected image.

### Preprocessing

* **Inference (deterministic):** decode → EXIF orientation → RGB → BGR. The YOLO predictor then letterboxes to
  the image size (training size by default, aspect ratio preserved), scales to [0, 1] and maps boxes back to
  original pixels. There's no sharpening, denoising or colour correction that could hide a defect.
* **Training (stochastic, `train.py` only):** moderate HSV jitter, translate, scale, horizontal flip, mosaic
  (`TRAINING_AUGMENTATION` in `ml/cv/preprocessing.py`).

### Dataset, validation, training, evaluation

```bash
pip install -e ".[cv]"                       # ultralytics + torch (CPU wheel: see pyproject.toml comment)

python -m ml.cv.synthetic --out data/cv/synthetic-v1 --images 800 --seed 42     # SYNTHETIC / DEMO data
python -m ml.cv.validate_dataset --data data/cv/synthetic-v1/dataset.yaml       # exit 1 if invalid
python -m ml.cv.train --data data/cv/synthetic-v1/dataset.yaml \
    --version qgenai-yolov8n-synthetic-v1 --base yolov8n.pt --epochs 30 --imgsz 416 --seed 42 --device auto
python -m ml.cv.evaluate --model artifacts/cv/qgenai-yolov8n-synthetic-v1/model/best.pt \
    --data data/cv/synthetic-v1/dataset.yaml --split test --out /tmp/metrics.json
```

The dataset format, split rules and how to add real data are in [`data/cv/README.md`](data/cv/README.md).
`train.py` validates the dataset first and aborts on errors. It then trains with a fixed seed and deterministic
settings, and evaluates on val and test.

**Model artifacts** (git-ignored; weights are never committed):

```
artifacts/cv/<model-version>/
  model/best.pt     weights                       metrics.json   val + test metrics (machine-readable)
  metadata.json     version, framework, base checkpoint, classes (+ train instances, trained yes/no),
                    training params, duration, data provenance ("synthetic_demo")
  runs/             raw Ultralytics training logs (results.csv)
```

A fresh clone has **no model**. With `CV_ENGINE=yolov8` and no artifact, the app still starts: `/api/health`
reports `status: degraded` and `cv_engine.available: false, error: "MODEL_NOT_FOUND: …"`. Every inspection
records CV as `UNAVAILABLE`, gets no decision, and goes to review. Run the commands above (≈25 min on 4 CPU
cores) to produce the demo model.

### Demo model results: SYNTHETIC / DEMO validation only

Model `qgenai-yolov8n-synthetic-v1`: YOLOv8n fine-tuned from COCO `yolov8n.pt` (sha256 `f59b3d833e2f…`).
Trained for 30 epochs at 416 px, batch 16, seed 42, on CPU (4 vCPU), in **19.2 min**.
The data was 800 synthetic images: 560 train / 120 val / 120 test.

| Split | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| val (synthetic) | 0.994 | 1.000 | 0.995 | 0.953 |
| test (synthetic, held out) | 0.994 | 1.000 | 0.995 | 0.956 |

Per class, test split:

| Class | Instances | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| crushed_packaging | 0 | – | – | – | – (no data: not evaluated, cannot be detected) |
| torn_packaging | 24 | 0.997 | 1.000 | 0.995 | 0.903 |
| leakage | 54 | 0.997 | 1.000 | 0.995 | 0.985 |
| broken_seal | 0 | – | – | – | – (no data: not evaluated, cannot be detected) |
| opened_package | 0 | – | – | – | – (no data: not evaluated, cannot be detected) |
| dented_container | 18 | 0.992 | 1.000 | 0.995 | 0.958 |
| damaged_box | 12 | 0.988 | 1.000 | 0.995 | 0.942 |
| missing_label | 40 | 0.996 | 1.000 | 0.995 | 0.993 |
| wrong_item | 0 | – | – | – | – (no data: not evaluated, cannot be detected) |
| visual_anomaly | 0 | – | – | – | – (no data: not evaluated, cannot be detected) |

At the deployment threshold (0.35) on the test split: the confusion matrix shows all 148
objects matched to the correct class, with 0 missed, 0 false positives and 0 cross-class confusions. At image level,
102 of 102 defect images were flagged and 0 of 18 clean images were flagged
(flag rate 85%). Full machine-readable results are in `artifacts/cv/<version>/metrics.json`.

The near-perfect scores mean only that **the synthetic task is easy**. The generator draws defects as simple,
high-contrast shapes.

**These numbers come from held-out synthetic images drawn by the same generator as the training data. They show
that the training, evaluation and inference pipeline works. They say nothing about accuracy on real
photos.** Real-world validation requires a real annotated dataset (see `data/cv/README.md`). No real-world
validation has been done.

### Inference, visualization, benchmark, export

```bash
python -m ml.cv.predict --image photo.jpg --save-visualization annotated.jpg   # JSON: quality + cv result
python -m ml.cv.visualize --input photo.jpg --output annotated.jpg              # boxes from real inference only
python -m ml.cv.benchmark --images data/cv/synthetic-v1/images/test --limit 100
pip install -e ".[cv-export]" && python -m ml.cv.export --check                 # optional ONNX artifact
```

Measured with `ml.cv.benchmark` on 100 held-out synthetic images × 2 runs, CPU only (4 vCPU
Intel Xeon @ 2.3 GHz, no GPU), image size 416:

| | mean | median | p95 |
|---|---|---|---|
| model forward pass (`inference_time_ms`) | 24.3 ms | 24.0 ms | 27.7 ms |
| engine end to end (decode → postprocess) | 28.8 ms | 28.6 ms | 32.4 ms |

These numbers apply to this machine only. Larger photos, a larger `CV_IMAGE_SIZE` or a bigger model will be slower.

ONNX export is a future deployment artifact. The Part 2 engine always runs the PyTorch weights.

### CPU / GPU

`CV_DEVICE=auto` uses `cuda:0` when PyTorch sees a GPU, otherwise `cpu`. `CV_DEVICE=cpu` forces CPU.
`CV_DEVICE=cuda` (or `cuda:N`) on a machine without that GPU is a **clear error**: the engine reports
`DEVICE_UNAVAILABLE` and does not silently fall back. The device actually used is recorded in every result
(`inference.inference_device`) and in `/api/health`. The tests and numbers in this README were produced
on **CPU only**; no GPU run has been tested.

### Running the API with the real engine

```bash
# .env
CV_ENGINE=yolov8
QUALITY_GATE_ENGINE=basic
DECISION_ENGINE=rules
CV_MODEL_PATH=artifacts/cv/qgenai-yolov8n-synthetic-v1/model/best.pt
```

The upload endpoint, the database and the UI are unchanged. The UI now shows the decision, the detections
(type, confidence, severity, box), the model version and inference time, with boxes drawn on the uploaded image.

## Fulfillment Pipeline Intelligence (Part 3)

> **The system identifies the first observed stage at which a defect appears based on available inspection
> evidence. It does not prove physical causality.**

Each layer answers one question, and none of them answers another layer's:

| Layer | Question | Implementation |
|---|---|---|
| Computer vision | What is visually present in this image? | `CV_ENGINE=yolov8` |
| Decision rules | Does this observation pass, fail or need review? | `DECISION_ENGINE=rules` |
| **Fulfillment analysis** | **Across the item's observations, when did each defect first appear?** | `PIPELINE_ANALYSIS_ENGINE=deterministic` |
| RCA + SOP grounding (Parts 4-5) | Why might it have happened, and what do the SOPs say to do? | `RCA_ENGINE=genai`, `RAG_ENABLED=true` |

Fulfillment analysis is pure, deterministic code (`backend/fulfillment/`). It uses no ML, no LLM and no
probabilities. The same observations always give the same result.

### Stages and observation history

```
SUPPLIER → RECEIVING(1) → STORAGE(2) → PICKING(3) → PACKING(4) → DISPATCH(5) → DELIVERY(6) → CUSTOMER
```

Stage order is defined once (`FULFILLMENT_FLOW` / `STAGE_ORDER` in `backend/domain/fulfillment.py`).

Inspections of the same physical item form a **trace**:

* **store + SKU + `item_id`**: a unit or handling-unit barcode (LPN). This is the preferred key because
  RECEIVING and STORAGE happen before any order exists.
* **store + SKU + `order_id`**: used only when there is no `item_id`.
* With neither, the inspection is analysed on its own (flag `NO_TRACE_IDENTIFIER`).

Observations of different items, orders, SKUs or stores are **never combined**.

Each inspection becomes a `PipelineObservation`. It is rebuilt from persisted results (no new inference):

1. **Operator feedback** comes first. It sets the outcome, and the operator's defect types if given.
2. Otherwise the **rules decision** sets PASS / FAIL / REVIEW, with the CV detection types.
3. With **no usable decision** (image quality rejected, CV unavailable, failed run), the observation is **REVIEW**.
   It is never treated as a PASS. Inspections still in progress are not evidence yet.

### Algorithm (per defect type: defect identity = defect type)

```
for each defect type D:
  first      = earliest inspected stage showing D
  previous   = the inspected stage just before `first`
  last good  = latest stage before `first` that explicitly PASSED
  missing    = stages between `previous` and `first` that were never inspected

  classification (first rule that matches):
    timestamps out of stage order        → UNCERTAIN              TIMESTAMP_ORDER_CONFLICT
    first == RECEIVING                   → PRE_EXISTING           DEFECT_PREEXISTING   (present on arrival)
    no previous inspection               → INSUFFICIENT_EVIDENCE  NO_PRIOR_OBSERVATION
    previous stage was REVIEW            → UNCERTAIN              PREVIOUS_STAGE_REVIEW
    another defect type vanished as D
      appeared (possible re-classification) → UNCERTAIN           AMBIGUOUS_DEFECT_IDENTITY
    uninspected stages in between        → INSUFFICIENT_EVIDENCE  MISSING_INTERMEDIATE_OBSERVATIONS
    otherwise                            → NEW_DEFECT             NEW_DEFECT_AFTER_GOOD_STAGE
                                                                  (or …_AFTER_DEFECTIVE_STAGE)
```

**Missing stages are never assumed good.** For example, with RECEIVING = PASS and PACKING = FAIL, the result is
"first observed at PACKING, last known good RECEIVING, STORAGE and PICKING not inspected" →
`INSUFFICIENT_EVIDENCE`. It is never "damage happened during packing".

**Persistence and resolution:** every later inspected stage gets a status for D:

* `PERSISTING`: still there.
* `RESOLVED`: inspected and no longer observed. This could be a repack, a replaced item or a missed detection.
* `UNCERTAIN`: a REVIEW stage, or a possible re-classification.

A defect first seen at RECEIVING is `PRE_EXISTING`. A later stage that still shows it (for example PACKING)
reports `current_stage_attribution: NOT_SUPPORTED`, so that stage is not blamed merely because the defect was
observed there.

### Evidence level (deterministic; not a probability, not model confidence)

| Classification | Evidence | Rule |
|---|---|---|
| NEW_DEFECT | **HIGH** | previous stage explicitly PASSED **and** first observation confirmed (FAIL decision or operator) |
| NEW_DEFECT | **MEDIUM** | exactly one of those two holds (e.g. previous stage FAILED for another defect, or D first seen only in a REVIEW) |
| NEW_DEFECT | **LOW** | neither holds → `REQUIRES_REVIEW` |
| PRE_EXISTING | HIGH / MEDIUM | receiving observation confirmed / unconfirmed |
| UNCERTAIN | LOW | always requires review |
| INSUFFICIENT_EVIDENCE | INSUFFICIENT | |

Every result also returns `evidence_checks`: previous stage inspected / passed / not REVIEW, no missing stages,
identity unambiguous, timestamps consistent, first observation confirmed.

**Attribution:**

* NEW_DEFECT with HIGH or MEDIUM evidence → `ATTRIBUTED` to the first observed stage.
* PRE_EXISTING → `NOT_ATTRIBUTED`, because it was present before any dark-store stage.
* INSUFFICIENT_EVIDENCE → `INSUFFICIENT_EVIDENCE`.
* UNCERTAIN, or NEW_DEFECT with LOW evidence → `REQUIRES_REVIEW`.

### Determinism, duplicates, ordering

* Observations are sorted by (stage order, timestamp, inspection id), so input order never matters.
* Duplicates are dropped and counted (`duplicates_ignored`), never double-counted. A duplicate is the same
  inspection twice, or the same photo (content hash) uploaded twice; the earliest is kept.
* Several inspections of the same stage are merged conservatively: FAIL beats REVIEW beats PASS, and the defects
  are combined.
* If an earlier stage was photographed after a later one, the result is `TIMESTAMP_ORDER_CONFLICT` and every
  attribution becomes UNCERTAIN.
* Complexity is O(N log N) for N observations, with at most 6 stages per defect type.

### Integration and API

* **In the pipeline:** the `PIPELINE_ANALYSIS` stage runs after a FAIL decision, a REVIEW decision, or when there
  is no automated decision. The inspection service loads the trace's earlier observations into
  `PipelineContext`; the analyzer itself never touches the DB. The snapshot is stored in
  `inspection_results.pipeline_analysis`:
  * `trace_analysis`: the full result.
  * `suspected_stage`: set only when ATTRIBUTED.
  * `evidence`: the explanation codes.
  * `stage_likelihoods` and `confidence` stay empty on purpose.

  A PASS decision skips the stage. The PASS observation still counts as evidence for later inspections.
* **On demand:** `GET /api/inspections/{id}/pipeline` re-analyses the trace from persisted data, so operator
  corrections apply immediately. It returns:
  * the 6-stage `timeline`, with PASS / FAIL / REVIEW / NOT_OBSERVED for each stage;
  * the ordered `observations`;
  * one `defect_analyses` entry per defect type;
  * `missing_stages`, `flags` and the disclaimer.

  The response is scoped to the inspection's own store + SKU + item/order trace.
* **New request field:** `POST /api/inspections` accepts `item_id`. `GET /api/inspections` gains `item_id` /
  `order_id` filters. Migration `0002` adds the nullable, indexed `inspections.item_id` column. No analysis
  table: results are reconstructed from existing data.
* **UI:** the inspection panel shows the stage timeline and one card per defect: first observed, last known good,
  stages not inspected, evidence, and whether the current stage is supported. Root-cause hypotheses are the
  separate GenAI RCA panel (Part 4).

```json
{"defect_type": "CRUSHED_PACKAGE", "classification": "NEW_DEFECT", "attribution_status": "ATTRIBUTED",
 "attributed_stage": "PACKING", "first_observed_stage": "PACKING", "last_known_good_stage": "PICKING",
 "previous_inspected_stage": "PICKING", "missing_stages": [], "evidence_level": "HIGH",
 "current_stage_attribution": "SUPPORTED", "explanation_code": "NEW_DEFECT_AFTER_GOOD_STAGE",
 "stage_statuses": [{"stage": "RECEIVING", "status": "NOT_PRESENT"}, "…",
                    {"stage": "DISPATCH", "status": "PERSISTING"}], "requires_review": false}
```

## GenAI Root-Cause Analysis (Part 4)

> **GenAI explains the evidence and proposes likely contributing causes. It does not decide anything and does
> not prove physical causality.** If the LLM is disabled or fails, every deterministic result still stands.

1. **Computer vision** detects the defect (type, confidence, severity, box).
2. **Rules** decide PASS / FAIL / REVIEW.
3. **Pipeline intelligence** (Part 3) determines the first observed defect stage, the last known good stage, the
   stages that were not inspected, and the evidence level.
4. **GenAI** explains that evidence. It proposes *likely contributing causes*, corrective actions and verification
   steps.
5. GenAI does **not** prove physical causality.
6. **Missing inspections reduce the evidence.** Uninspected stages are passed as `NOT_OBSERVED` and are never
   called good.
7. **REVIEW is not PASS.**
8. **An LLM failure does not invalidate the deterministic inspection results.**

### What the LLM is and isn't allowed to produce

| Produced by deterministic code (never the LLM) | Produced by the LLM (labelled AI) |
|---|---|
| defect, confidence, severity, PASS/FAIL/REVIEW | `summary` |
| first observed / last known good / previous stage, missing stages | `likely_causes[]`: category from a fixed taxonomy, hedged hypothesis, rationale, optional `related_stage` |
| classification, evidence level, `requires_review` | `corrective_actions[]` (with priority) |
| `observed_evidence[]` (fact sentences) and `limitations[]` | `verification_steps[]`, `ai_limitations[]` |

The LLM's output schema has **no** fields for decisions, detections, stage attribution or numeric confidence, so
it cannot supply them. Cause categories: `PACKAGING_MATERIAL`, `PACKAGING_PROCESS`, `SEALING_PROCESS`,
`HANDLING`, `STORAGE_CONDITION`, `PICKING_HANDLING`, `TRANSPORT_HANDLING`, `LABELING`, `PRODUCT_CONDITION`,
`UNKNOWN`. A category is never forced; `UNKNOWN` or no cause is allowed.

### Flow

```
FulfillmentTraceAnalysis (Part 3, the only evidence source)
   → RCAEvidence: timeline, observations, per-defect attribution, candidate_stages, untrusted operator notes
   → prompt: static system prompt (rca-v1) + evidence JSON in <evidence> tags (data, not instructions)
   → LLMProvider (anthropic | mock) → raw JSON text
   → schema validation (Pydantic, sizes bounded) → grounding guardrails → RootCauseAnalysis
```

**Grounding guardrails** run on every response. Any violation discards the *whole* output (`INVALID_OUTPUT`):

* exactly one finding per analysed defect type (no merging, no invented defects);
* a cause's `related_stage` must lie in that defect's `candidate_stages`: after the last known good stage, up to
  the first observed stage. A PRE_EXISTING defect only allows `RECEIVING`, so later stages can't be blamed;
* hypotheses must be hedged ("likely / possible / may / could");
* no "caused by", "root cause is", "proven" or "definitely";
* no percentages or probabilities;
* no SOP ids or "per company policy" (there is no SOP corpus until RAG);
* no blaming people.

**Prompt injection:** operator comments are passed only as `untrusted_operator_notes` inside the evidence JSON.
`<` and `>` are escaped there, so a note cannot close the evidence block. The system prompt contains no request
data and says never to follow instructions found in notes.

### Fallback behaviour (never fails an inspection)

| Situation | RCA `status` | Effect |
|---|---|---|
| No defect in the item's history | `NOT_APPLICABLE` | no LLM call |
| `LLM_ENABLED=false`, no API key, SDK missing, timeout, rate limit, 4xx/5xx, network error, refusal, truncated output | `UNAVAILABLE` (`error_code` says which) | facts kept, AI fields empty, `requires_review=true` |
| Malformed JSON, schema violation, oversized or ungrounded output | `INVALID_OUTPUT` | same as above, and the output is discarded |
| Validated output | `COMPLETED` | facts plus AI hypotheses |

In the pipeline, `RCA_PROCESSING` runs after `PIPELINE_ANALYSIS` for FAIL decisions. `ACTION_GENERATED`
(`ACTION_ENGINE=rca`) turns the AI-suggested actions into an `ActionRecommendation`, without a second LLM call.
The final state is:

* `COMPLETED` for FAIL with actions and conclusive evidence;
* `REQUIRES_REVIEW (ATTRIBUTION_REQUIRES_REVIEW)` when the stage evidence is insufficient or uncertain;
* `REQUIRES_REVIEW (NO_CORRECTIVE_ACTION)` when the LLM was unavailable. The FAIL decision and the Part 3
  analysis are kept.

### Canonical example

```
Timeline   RECEIVING PASS · STORAGE PASS · PICKING PASS · PACKING FAIL · DISPATCH NOT_OBSERVED · DELIVERY NOT_OBSERVED
CV         leakage, confidence 0.96, severity HIGH                → decision FAIL
Pipeline   first observed PACKING · last known good PICKING · NEW_DEFECT · evidence HIGH
GenAI RCA  (AI-generated, evidence-grounded)
  Observed evidence (system facts): LEAKING_PACKAGE was first observed at PACKING. The last inspection that
    explicitly passed was PICKING. CV detection at PACKING: confidence 0.96, severity HIGH.
  Likely contributing causes (AI hypotheses): possible sealing-process issue [SEALING_PROCESS];
    possible packaging-material damage [PACKAGING_MATERIAL]; possible excessive handling pressure [HANDLING]
  Corrective actions: inspect sealing equipment; verify packaging-material condition; review packing handling
    procedure; re-inspect affected inventory
  Limitation: the pipeline analysis identifies the first observed stage where a defect appears based on
    available inspection evidence. It does not prove physical causality.
```

The automated test `tests/cv/test_rca_full_path.py::test_canonical_demo_end_to_end` runs exactly this path with
real YOLO inference, using the mock LLM so the run is deterministic and offline.

### Provider and model

* **Provider: Anthropic Claude**, via the official `anthropic` Python SDK (`pip install -e ".[llm]"`). The code
  is written against `LLMProvider.generate_rca(prompt) → raw text`. Adding another vendor means one class plus
  one line in `backend/rca/providers/__init__.py`. No other provider is implemented, on purpose.
* **Default model `claude-opus-5-5`**, run with `LLM_EFFORT=low` to keep latency down for a short structured
  task. It is configurable with `LLM_MODEL`; `claude-sonnet-5-5` or `claude-haiku-5-5` are cheaper and faster.
* **Structured output:** native JSON-schema structured output (`output_config.format`), with the `defect_type`
  enum narrowed per request, then strict Pydantic validation.
* **Refusal fallback:** `LLM_REFUSAL_FALLBACK=true` adds Anthropic's server-side fallback (`fallbacks: "default"`),
  so a request declined by a safety classifier is retried on Anthropic's recommended model.
* **No temperature:** `LLM_TEMPERATURE` is accepted but not sent, because the API rejects sampling parameters on
  current models.
* **Timeout:** `LLM_TIMEOUT_SECONDS` (default 30) with one SDK retry. External latency depends on the provider and
  network; a real call was **not measured** here because no API key was available in the build environment.
  Tests drive the real provider through the SDK against a simulated HTTP transport.
* **`LLM_PROVIDER=mock`** is a deterministic template test double, labelled "mock (not AI)" in the API and UI. It
  is refused when `ENVIRONMENT=production`.

### API

* `GET /api/inspections/{id}/rca` rebuilds the Part 3 analysis from persisted inspections and operator feedback,
  builds the evidence, calls the LLM once, validates the output and returns `RootCauseAnalysis`. It is not cached
  and needs **no new table**. The pipeline-time snapshot lives in the existing `inspection_results.rca` column.
* `GET /api/health` → `llm: {enabled, provider, model, status: disabled|configured|not_configured|
  dependencies_missing, effort, timeout_seconds, prompt_version}`. This is a static check: it never calls the
  provider and never includes credentials.
* **Logs** record inspection id, stage, RCA status, provider, model, latency, token counts and error code. They
  never contain prompts, raw model output, API keys or images. The SDK's and HTTP client's debug loggers are
  pinned to WARNING because they would dump request bodies.

### UI

After the fulfillment timeline, a **"GenAI RCA · AI-generated · evidence-grounded"** panel shows:

* per defect: first observed stage, last known good stage and evidence level;
* **Observed evidence (system facts)**, then **Likely contributing causes (AI hypotheses)**, **Corrective actions
  (AI-suggested)**, **Verification steps** and **Limitations**, as visually distinct lists;
* a provider/model line ("Deterministic MOCK provider (not AI)" for the mock);
* when unavailable, the reason, and a button to regenerate via `GET /rca`.

## RAG / SOP Intelligence (Part 5)

> **DEMO / SYNTHETIC knowledge base.** The 9 documents in [`knowledge/`](knowledge/README.md) were written for
> this project. They are **not** official SOPs of Blinkit, Zepto, Swiggy Instamart, Amazon or any other company,
> and every document and every API response says so (`provenance: DEMO_SYNTHETIC`).

Part 4 suggests corrective actions from the model's general knowledge. Part 5 checks those actions against the
store's SOP knowledge base. It retrieves the relevant SOP sections and keeps only actions that a retrieved section
supports, each with a citation and a verbatim quote. It is the same RCA process with one more step, not a
second pipeline:

```
Part 3 trace analysis ─► Part 4 RCA (hypotheses + AI actions) ─► retrieval (per defect) ─► LLM grounding call
                                                                        │                       │
                                                         no source ≥ threshold          citation guardrails
                                                                        ▼                       ▼
                                                         NO_RELEVANT_SOURCE        GROUNDED / NOT_SUPPORTED
                                                         (no LLM call)             (+ SOURCES_ONLY on failure)
```

### Four kinds of content, never mixed

| Kind | Where | Produced by |
|---|---|---|
| **System evidence** | `defects[].facts`, `observed_evidence` | deterministic (CV, rules, Part 3) |
| **GenAI hypothesis** | `defects[].likely_causes`, `corrective_actions`, `verification_steps` | LLM (Part 4), labelled AI |
| **SOP-backed recommendation** | `rag.grounded_actions[]` (action + `source_document`, `source_section`, `source_version`, `supporting_quote`) | LLM wording, tied to a retrieved chunk; source metadata filled **from the index, never by the LLM** |
| **Unsupported AI action** | `rag.unsupported_actions[]` | Part 4 actions no retrieved SOP supports, explicitly "not SOP-backed" |

### Ingestion

`python -m backend.rag.ingest` (also run by the Docker entrypoint when `RAG_ENABLED=true`):

1. **Load** every `knowledge/**/*.md` (sorted, deterministic).
2. **Validate** front matter (`document_id`, `title`, `stage`, `document_type`, `version`, `defect_types`,
   `provenance`) with a strict Pydantic model. Errors: `MISSING_FRONT_MATTER`, `INVALID_METADATA`,
   `NO_SECTIONS`, `DUPLICATE_DOCUMENT_ID`, `DUPLICATE_SECTION`, `EMPTY_KNOWLEDGE_BASE`,
   `KNOWLEDGE_BASE_MISSING`; any error exits 1 and writes no index. Warnings: `DUPLICATE_CONTENT` (skipped),
   `SUSPICIOUS_INSTRUCTION_TEXT` (kept, treated as data).
3. **Chunk by section**: each `## ` section is one chunk with a stable id `DOCUMENT-ID#section-slug`, carrying all
   document metadata. Sections longer than `RAG_MAX_CHUNK_CHARS` split on line boundaries (`-1`, `-2`, …).
4. **Embed** title + section + stage + defect words + text.
5. **Store** in a local index directory: `vectors.npy` + `chunks.json` + `manifest.json` (provider, model, dim,
   corpus SHA-256, document versions, provenance, build time), written atomically.

Demo corpus: 9 documents → 66 chunks.

### Embeddings and vector store

* **Default: `hashing`.** A signed feature-hashing embedding (BLAKE2b) of normalised word stems, a small
  domain synonym map (leak/seep/wet, seal/sealer, …) and bigrams. It uses 2048 dims, sublinear TF and L2
  normalisation. It is a lexical embedding: deterministic, offline, numpy only, and with no model download. It
  matches on shared vocabulary, not meaning.
* **Optional: `sentence_transformers`** (`pip install -e '.[rag-neural]'`, e.g.
  `RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`), behind the same `EmbeddingProvider` interface.
  **Untested:** Hugging Face was unreachable from the build environment. If it cannot load, RAG reports
  `UNAVAILABLE` (`EMBEDDING_DEPENDENCIES_MISSING` / `EMBEDDING_MODEL_UNAVAILABLE`).
* The service refuses an index built with a different provider/model/dim (`INDEX_EMBEDDING_MISMATCH`). After
  changing embeddings or documents, re-run the ingest.
* **Vector store:** brute-force cosine over the in-memory matrix (`local-numpy`). That is exact and well under a
  millisecond for hundreds of chunks. No vector database is needed at this scale.

### Retrieval strategy

One query per defect type, built from the RCA context: first observed stage, defect type, severity,
classification, the Part 4 hypotheses and categories, the AI actions, the summary, and an action vocabulary
("corrective actions, immediate actions, required checks, verification, escalation"). When the LLM is
unavailable, the query falls back to the deterministic facts only. Then:

1. **Similarity gate:** drop chunks with cosine < `RAG_MIN_SIMILARITY` (raw similarity, before any boost).
2. **Skip context sections** (`Purpose`, `Applicable Stage`): they describe a document, not an action.
3. **Metadata boosts, not filters:** +0.10 when the chunk's stage is in the defect's Part 3 window (first observed
   stage, plus uninspected stages unless `PRE_EXISTING`), +0.05 for `GENERAL` documents, +0.10 when the document
   lists the defect type. A relevant document from another stage can still be retrieved; it just ranks lower.
4. Sort by score (ties by chunk id; deterministic), keep **at most 2 chunks per document**, and keep `RAG_TOP_K`.

**Threshold calibration (`RAG_MIN_SIMILARITY=0.15`)** on the demo corpus with the hashing embedding: relevant
queries' top similarity was 0.17–0.36, and unrelated queries' best match was ≤ 0.07. 0.15 sits in the gap.
Recalibrate for real documents or another embedding model.

### Grounding and citations (prompt `rag-v1`)

* The LLM sees only the retrieved chunks, as escaped JSON inside `<context>` / `<sources>` blocks (`<` and `>`
  are JSON-escaped, so no document text can close a block). The static system prompt says block content is data
  and that instructions in documents must be ignored. Operator notes are **not** part of this prompt.
* The output schema's `chunk_id` enum is **only the retrieved chunk ids**, and `defect_type` is only the
  analysed defects.
* Server-side guardrails discard the whole output on any violation:
  * `CITATION_NOT_RETRIEVED`: cited a chunk not retrieved *for that defect*;
  * `QUOTE_NOT_IN_SOURCE`: the quote is not verbatim in the cited chunk;
  * `QUOTE_ACTION_MISMATCH`: the action shares no content word with its quote;
  * `INSTRUCTION_TEXT`: prompt-injection-like text in the action or quote;
  * `BAD_RCA_ACTION_INDEX`;
  * plus Part 4's causal-certainty, invented-number, fabricated-SOP-reference and person-blame checks.
* `source_document` / `section` / `version` / `provenance` are copied from the index, so citations cannot be
  invented. There are no URLs: documents are identified by id, section and version.

### Statuses and failures (never fail an inspection)

| `rag.status` | When | Result |
|---|---|---|
| `GROUNDED` | ≥ 1 validated SOP-backed action | `requires_review=false` only if every defect is grounded |
| `NOT_SUPPORTED` | sources found, none supports an action | *"No supporting SOP guidance was found in the available knowledge base for these actions. Manual review is recommended."* |
| `NO_RELEVANT_SOURCE` | nothing ≥ threshold | no LLM call, no unrelated documents passed; *"No relevant SOP guidance was found … Manual review is recommended."* |
| `SOURCES_ONLY` | LLM disabled / unavailable / malformed or ungrounded output | sources listed for manual review; no SOP-backed actions; `generation.error_code` says why |
| `UNAVAILABLE` | KB or index missing / corrupt / mismatched, embeddings unavailable, retrieval error | RCA unaffected; `message` carries the code |
| *(field `null`)* | `RAG_ENABLED=false` | Part 4 behaviour unchanged |

Pipeline routing: with RAG enabled, a FAIL whose actions are not fully SOP-grounded ends in
`REQUIRES_REVIEW (SOP_GUIDANCE_REQUIRES_REVIEW)`. `ATTRIBUTION_REQUIRES_REVIEW` still takes precedence. The
action engine lists SOP-backed actions first (`sop_reference: "PACKING-SOP demo-1.0 §Seal Inspection"`, with the
quoted SOP text), then AI-only actions marked *"not SOP-backed"*.

### Canonical example

RECEIVING, STORAGE and PICKING PASS; PACKING FAIL with `LEAKING_PACKAGE`. Part 3 gives first observed PACKING,
last known good PICKING, HIGH evidence. Part 4 hypothesises a sealing-process issue. RAG retrieves
`PACKING-SOP#seal-inspection` among others, and returns:

```json
{"defect_type": "LEAKING_PACKAGE",
 "action": "Inspect sealing equipment and verify seal integrity before releasing affected inventory",
 "source_document": "Packing and Sealing SOP", "source_section": "Seal Inspection", "source_version": "demo-1.0",
 "source_provenance": "DEMO_SYNTHETIC",
 "supporting_quote": "Inspect sealing equipment and verify seal integrity before releasing affected inventory.",
 "supports_rca_action": "Inspect the sealing equipment and settings at PACKING"}
```

This is tested end to end with the operator-feedback path (always run) and with the **real trained YOLO model**
detecting the leak (`tests/rag/test_rag_real_cv.py`, skipped if the demo model is not trained).

### Latency (CPU, demo corpus, hashing embeddings, mock LLM)

| Step | Median | p95 |
|---|---|---|
| Ingestion (load + validate + chunk + embed, 9 docs / 66 chunks) | 21.8 ms | 28.1 ms |
| … of which embedding 66 chunks | 10.5 ms | 13.6 ms |
| One retrieval | 0.25 ms | 0.33 ms |
| RAG step (retrieval + grounding call + guardrails) | 2.1 ms | 3.4 ms |
| RCA + RAG total | 3.2 ms | 4.8 ms |

With the real Anthropic provider, the grounding call adds one more LLM round trip per `GET /rca`. That latency is
unmeasured: no API key was available.

### API, health, UI, logs

* `GET /api/inspections/{id}/rca` now includes `rag` (status, defects, `retrieved_sources[]`,
  `grounded_actions[]`, `unsupported_actions[]`, knowledge-base metadata, retrieval latency, generation). The
  pipeline result carries the same object.
* `GET /api/health` → `rag: {enabled, embedding_provider, embedding_model, vector_store, documents, chunks,
  dimensions, status: disabled|ready|unavailable, error}`, with no document text or secrets. Health is `degraded`
  when RAG is enabled but unavailable.
* UI: the RCA panel shows a RAG status badge and three sections per defect: **Observed Evidence (system)**,
  **GenAI Analysis (hypotheses, not facts)** (including the AI actions that are *not SOP-backed*) and
  **SOP-Grounded Actions** (✓ action with *Source · Section · Version* underneath). Retrieved sources are in a
  collapsible list.
* Logs: inspection id, defect, stage, retrieval count, top similarity, status, latency and counts. They never
  include document text, prompts or keys.

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
| `QUALITY_GATE_ENGINE` / `CV_ENGINE` / `DECISION_ENGINE` | `stub` | `basic` / `yolov8` / `rules` enable the Part 2 modules |
| `PIPELINE_ANALYSIS_ENGINE` | `stub` | `deterministic` enables fulfillment stage attribution (Part 3) |
| `RCA_ENGINE` / `ACTION_ENGINE` | `stub` | `genai` / `rca` enable GenAI RCA and RCA-derived actions (Part 4) |
| `LLM_ENABLED` | `false` | Turn the LLM on; everything deterministic works without it |
| `LLM_PROVIDER` / `LLM_MODEL` | `anthropic` / `claude-opus-5-5` | `mock` = deterministic test double (refused in production) |
| `LLM_API_KEY` | *(empty)* | Falls back to `ANTHROPIC_API_KEY`; never logged, never returned |
| `LLM_MAX_TOKENS` / `LLM_TIMEOUT_SECONDS` / `LLM_EFFORT` | `8000` / `30` / `low` | Output budget (incl. thinking), request timeout, effort level |
| `LLM_REFUSAL_FALLBACK` | `true` | Anthropic server-side fallback on safety-classifier refusals |
| `LLM_TEMPERATURE` | *(unset)* | Accepted but not sent (the API rejects sampling parameters) |
| `RAG_ENABLED` | `false` | Ground RCA actions in the SOP knowledge base (build the index first: `python -m backend.rag.ingest`) |
| `RAG_KNOWLEDGE_PATH` / `RAG_INDEX_PATH` | `knowledge` / `data/rag/index` | Documents / local vector index (git-ignored) |
| `RAG_EMBEDDING_PROVIDER` / `RAG_EMBEDDING_MODEL` / `RAG_EMBEDDING_DIM` | `hashing` / `hashing-lexical-v1` / `2048` | `sentence_transformers` optional (`.[rag-neural]`, untested here) |
| `RAG_TOP_K` / `RAG_MIN_SIMILARITY` / `RAG_MAX_CHUNK_CHARS` | `5` / `0.15` / `1200` | Sources per defect, cosine gate (calibrated on the demo corpus), chunk size |
| `CV_MODEL_PATH` | `artifacts/cv/qgenai-yolov8n-synthetic-v1/model/best.pt` | Model weights |
| `CV_MODEL_VERSION` | *(from metadata.json)* | Version override |
| `CV_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:N` (an unavailable CUDA device is an error, not a fallback) |
| `CV_CONFIDENCE_THRESHOLD` / `CV_IOU_THRESHOLD` | `0.35` / `0.5` | Detection threshold / NMS overlap |
| `CV_IMAGE_SIZE` | *(training size)* | Inference letterbox size |
| `CV_MAX_DETECTIONS` | `50` | Detections kept per image |
| `CV_TAXONOMY_PATH` | `ml/cv/taxonomy.yaml` | Defect taxonomy |
| `CV_SEVERITY_LARGE_AREA_RATIO` | `0.20` | Box/image area that escalates severity by one level |
| `CV_MIN_IMAGE_WIDTH` / `HEIGHT`, `CV_MAX_ASPECT_RATIO`, `CV_BLUR_THRESHOLD`, `CV_MIN_BRIGHTNESS` / `MAX` | 128 / 128, 6, 15, 20 / 240 | Quality gate thresholds |
| `DECISION_FAIL_MIN_SEVERITY` / `DECISION_FAIL_MIN_CONFIDENCE` | `HIGH` / `0.5` | Automatic FAIL criteria |
| `DECISION_AUTO_PASS` | `false` | PASS when CV finds nothing (off by default) |

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
| GET | `/api/inspections/{id}/pipeline` | Fulfillment history of the item/order: stage timeline + first-defect attribution |
| GET | `/api/inspections/{id}/rca` | GenAI RCA grounded in that evidence (facts + labelled AI hypotheses; safe fallback) + `rag`: retrieved SOP sources and cited SOP-grounded actions |
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
pytest                                   # unit + API + integration on SQLite (+ CV tests if ultralytics is installed)
TEST_DATABASE_URL=postgresql+psycopg://qgenai:<pw>@localhost:5432/qgenai_test pytest tests/integration
                                         # also runs migrations + persistence on a disposable PostgreSQL DB
```

The tests cover domain validation, every legal and illegal state transition, the orchestrator (stub flow
plus routing, failure and contract-violation cases using test-only doubles), image validation (valid,
wrong extension, mismatched content, oversized, pixel bomb, corrupt, empty), storage path-traversal
protection, configuration guards, all API endpoints, the error envelope, optional API-key auth,
Alembic migrations, and a full persisted lifecycle.

**CV tests** (`tests/cv/`, skipped automatically without `.[cv]`) cover:
* taxonomy, severity, postprocessing (class mapping, thresholds, clipping, malformed outputs), preprocessing,
  quality gate, decision rules, the dataset validator (every error class) and the synthetic generator.
* the real `YoloCVEngine`, using a **seeded, untrained YOLOv8n built at test time**. This is genuine Ultralytics
  inference whose outputs are meaningless, so the mechanics can be tested deterministically without committing
  weights. Also: missing / corrupt model, CUDA-unavailable, taxonomy mismatch, model versioning, caching.
* end-to-end API runs: upload → quality gate → real inference → decision → pipeline → persistence → health.
* one test that needs the **trained demo artifact** (skipped if it hasn't been trained): it checks that the
  model finds the labelled defect class in ≥ 80% of held-out *synthetic* defect images.

**Fulfillment tests** (`tests/fulfillment/`, plus `tests/cv/test_fulfillment_full_path.py`) cover:
* the pure analyzer: all 10 required scenarios (all PASS, new defect, pre-existing, missing stages, REVIEW
  predecessor, multiple defect types, resolved defect, out-of-order timestamps, other items, duplicates),
  evidence rules, input-order independence and a 20,000-observation scaling check.
* observation precedence: operator over decision over no-decision (which becomes REVIEW).
* the API on persisted data: traces by item / order / none, in-flight exclusion, pipeline-stage snapshot,
  health, validation.
* the full path with **real CV inference**: image → YOLO → rules → persisted observation → attribution.

**GenAI RCA tests** (`tests/rca/`, plus `tests/cv/test_rca_full_path.py`; no live API, no API key) cover:
* the real Anthropic provider via the SDK against a simulated HTTP transport: request shape (structured output,
  effort, fallback, no temperature), every failure class (timeout, network, 401/403/400/404/429/500/529, refusal,
  truncation, missing key, missing SDK), and no key or prompt in logs;
* RCA per Part 3 scenario: new defect, pre-existing, missing stages, REVIEW predecessor, multiple defects,
  insufficient observations, no defect, operator override;
* grounding: exactly the supplied observations reach the LLM, NOT_OBSERVED and REVIEW are kept as they are,
  prompt-injection delimiting, and every guardrail violation rejected with the facts kept;
* API and health (including secret safety);
* pipeline routing: COMPLETED, ATTRIBUTION_REQUIRES_REVIEW, LLM unavailable;
* the canonical demo end to end with real YOLO inference.

## Roadmap

| Part | Scope | Status |
|---|---|---|
| 1 | Foundation + pipeline architecture | ✅ |
| 2 | Computer Vision (YOLOv8 `CVEngine`, basic quality gate, decision rules, dataset/training/eval tooling) | ✅ (demo model: synthetic data only) |
| 3 | Fulfillment Pipeline Intelligence (stage history, first-defect attribution, evidence levels) | ✅ |
| 4 | GenAI Root-Cause Analysis (grounded hypotheses, corrective actions, safe fallback) | ✅ |
| **5** | RAG / SOP Intelligence (demo SOP knowledge base, retrieval, cited SOP-grounded actions) | ✅ this commit (demo SOPs only) |
| 6 | Active Learning (feedback review → labelled data → retraining → model versions) | planned |
| 7 | Minimal security + production hardening (operator auth/roles, async workers, object storage) | planned |
| later | Image Quality Gate, advanced (glare, framing/occlusion, product-in-frame, retake UX) | planned |

## Known limitations

* **No real-world model.** The only trained model learned from synthetic cartoon scenes. Expect it to perform
  poorly on real photos until it is retrained on a real annotated dataset. Its metrics are demo metrics.
* 5 of 10 taxonomy classes have no data at all (`placeholder`), and `wrong_item` needs SKU reference matching.
* RAG / SOP:
  * the knowledge base is 9 DEMO / SYNTHETIC documents, not approved SOPs; replace them before real use;
  * the default hashing embedding is lexical: it misses paraphrases with no shared words, and its threshold is
    calibrated only on this corpus. Neural embeddings are wired in but were not tested (no Hugging Face access);
  * grounding shows that an SOP section supports an action; it does not prove the action fixes the cause;
  * the quote check is verbatim matching, so a correct paraphrase without a quote is discarded.
* GenAI RCA:
  * produces hypotheses only; SOP knowledge comes from Part 5 RAG, whose documents are demo content;
  * the guardrails are pattern-based and can be over-strict (a valid but differently worded answer is discarded)
    or miss creative wording, so a human must still review AI suggestions;
  * each `GET /rca` makes a synchronous LLM call with no caching or rate limiting;
  * no live LLM call was made during development, because no API key was available. The provider is tested
    against simulated HTTP responses, and real-model output quality and latency are unmeasured.
* Stage attribution is only as good as the inspections that exist:
  * it needs an `item_id` (or `order_id`) to link stages;
  * defect identity is the defect *type*, with no visual re-identification of the same physical defect;
  * a "resolved" defect may simply be a missed detection.
  It reports the first *observed* stage, never a proven cause.
* Inference runs synchronously inside the upload request, one image at a time per process (the model is guarded
  by a lock). That's fine for a demo, but production needs a worker queue (Part 7). The orchestrator is already
  HTTP-independent.
* GPU inference is implemented but has not been tested (no GPU was available). ONNX export is an optional
  artifact; it isn't used for serving.
* The Docker image does not include the CV extra (PyTorch is large). Build a CV image with `pip install ".[cv]"`
  if you need it.
* Ultralytics is AGPL-3.0 (see [Model / framework](#model--framework)).
* Pipeline history is persisted after the run, not incrementally. A process crash mid-run would
  leave the inspection in `IMAGE_RECEIVED`.
* `API_KEY` is a single shared key: a placeholder, not user authentication.
* Only local image storage is implemented. The interface is ready for S3, GCS or Azure Blob.
* Stores and SKUs are simulated sample data, and there is no API for managing them.
