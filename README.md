# Q-GenAI

**Computer-vision-first quality intelligence for quick-commerce dark stores.**

> **Status: Part 2 of 7, Computer Vision inspection engine** (on top of Part 1: foundation + pipeline).
> Implemented: inspection pipeline, API, database, operator UI, an image quality gate, a **real YOLOv8
> CV engine**, deterministic decision rules, and dataset / training / evaluation tooling.
> **The only trained model is a demo model trained on SYNTHETIC data.** It proves the engineering end to end.
> It is **not** evidence of accuracy on real product photos (see [Computer Vision](#computer-vision-part-2)).
> Fulfillment analysis, RCA, corrective actions and active learning are still placeholders (`NOT_IMPLEMENTED`).

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

**Not implemented yet (by design):** fulfillment analysis, RCA, RAG / vector DB, LLMs, SOP generation,
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
├── documents/                future SOP corpus for RAG (Part 5)
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
decision PASS → `COMPLETED`; decision REVIEW → `REQUIRES_REVIEW`; decision FAIL with an action →
`COMPLETED`; no real decision → `REQUIRES_REVIEW`; module raises or returns the wrong type → `FAILED`
(`MODULE_ERROR` / `MODULE_CONTRACT_VIOLATION`); module raises `NotImplementedError` → recorded as
`NOT_IMPLEMENTED`.

### Module contracts

| Interface | Contract | Implementations |
|---|---|---|
| `ImageQualityGate.assess` | `ImageInput → ImageQualityResult` | `stub`, **`basic`** (Part 2); advanced checks in Part 3 |
| `CVEngine.detect` | `ImageInput → CVResult` (detections: bbox, defect type, confidence, severity; inference metadata) | `stub`, **`yolov8`** (Part 2) |
| `DecisionEngine.decide` | `CVResult + InspectionInfo → QualityDecision` (PASS/FAIL/REVIEW, reason, confidence, severity, source) | `stub`, **`rules`** (Part 2) |
| `PipelineAnalyzer.analyze` | `PipelineContext → PipelineAnalysis` (suspected stage, likelihoods, evidence) | `stub` (Part 4) |
| `RCAEngine.analyze` | `PipelineContext → RCAResult` (possible causes, evidence, confidence, investigation) | `stub` (Part 5) |
| `ActionEngine.recommend` | `RCAResult + context → ActionRecommendation` (action, priority, responsible stage, instructions) | `stub` (Part 5) |
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
| `inspections` | What was inspected, at which fulfillment stage, when, current pipeline status, final decision |
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
| `PIPELINE_ANALYSIS_ENGINE`, `RCA_ENGINE`, `ACTION_ENGINE` | `stub` | Only `stub` exists (Parts 4–5) |
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

## Roadmap

| Part | Scope | Status |
|---|---|---|
| 1 | Foundation + pipeline architecture | ✅ |
| **2** | Computer Vision (YOLOv8 `CVEngine`, basic quality gate, decision rules, dataset/training/eval tooling) | ✅ this commit (demo model: synthetic data only) |
| 3 | Image Quality Gate, advanced (glare, framing/occlusion, product-in-frame, retake UX) | planned |
| 4 | Fulfillment Intelligence (where in the flow damage happened, stage analytics) | planned |
| 5 | GenAI RCA + RAG (SOP retrieval, grounded root cause, corrective actions) | planned |
| 6 | Active Learning (feedback review → labelled data → retraining → model versions) | planned |
| 7 | Minimal security + production hardening (operator auth/roles, async workers, object storage) | planned |

## Known limitations

* **No real-world model.** The only trained model learned from synthetic cartoon scenes. Expect it to perform
  poorly on real photos until it is retrained on a real annotated dataset. Its metrics are demo metrics.
* 5 of 10 taxonomy classes have no data at all (`placeholder`), and `wrong_item` needs SKU reference matching.
* Fulfillment analysis, RCA and corrective actions are still placeholders. A FAIL decision therefore ends in
  `REQUIRES_REVIEW` (`NO_CORRECTIVE_ACTION`), and inspections without a real decision go to human review.
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
