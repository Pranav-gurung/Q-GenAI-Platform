# Q-GenAI — Architecture (Sections B + D)

## 1. Overview

```
 ┌───────────────────────────┐   HTTPS/JSON + Bearer JWT   ┌──────────────────────────────────────────┐
 │ Streamlit UI (frontend/)  │ ──────────────────────────▶ │ FastAPI backend (app/)                   │
 │  login · inspect · history│                             │                                          │
 │  review · audit · admin   │ ◀────────────────────────── │  routers/ ──▶ dependencies.py (authN)    │
 └───────────────────────────┘                             │      │        services/authorization.py  │
                                                           │      ▼                (RBAC + scope)     │
                                                           │  services/inspection_service.py          │
                                                           │      ├─▶ utils/image_utils.py  (validate)│
                                                           │      ├─▶ services/inference.py ─▶        │
                                                           │      │     services/detector.py (YOLOv8) │
                                                           │      ├─▶ services/inspection_rules.py    │
                                                           │      └─▶ services/audit_service.py       │
                                                           │             (SHA-256 hash chain)         │
                                                           │                  │                       │
                                                           │       SQLite (SQLAlchemy)  storage/      │
                                                           └──────────────────────────────────────────┘
 Section A ──(YOLO dataset)──▶ training/*.py ──▶ models/*.pt ──(admin registers/activates)──▶ detector
 Section C ◀──(GET /inspections/{id}/rca-payload, image-free JSON)── backend
```

One process for the API and one for the UI: no microservices, queues or cloud dependencies.

### Request flow (`POST /predict`)
1. `dependencies.get_auth_context` verifies the JWT (signature, expiry, issuer), checks it has not
   been revoked, re-loads the user **from the database**, and checks `is_active` and `token_version`.
2. `require_permission(INSPECTION_RUN)` lets the request through only for operators and quality managers.
3. `inspection_service.run_inspection`:
   1. commits the audit event `inspection.requested`;
   2. resolves the confidence threshold. A value outside the operator range needs the
      `THRESHOLD_OVERRIDE` permission and a reason;
   3. validates and decodes the image (extension, magic bytes, size, pixel count, integrity);
   4. runs YOLO through `inference.run_inference` and measures inference time with `perf_counter`;
   5. computes PASS/FAIL/REVIEW with `inspection_rules.evaluate` under the active policy version;
   6. writes the annotated PNG, then **in one DB transaction** inserts the `Inspection` row and the
      audit events `inspection.completed`, optional `quality.threshold_override`, and
      `inspection.result_created`.
4. The response contains the inspection ID and the IDs of the audit events that recorded it.

### Failure handling
| Failure | Behaviour |
|---|---|
| Model weights missing or corrupt | `model_status` = `unavailable`/`error`, **no detections**, decision **REVIEW** |
| Weights are generic COCO | flagged `is_generic_pretrained`, decision forced to **REVIEW** |
| Invalid image | 400 with a safe message; `inspection.rejected` audited |
| Audit write or commit fails | whole transaction rolled back, annotated file deleted, **500 "NOT saved"**; no inspection ID is returned |
| Audit write fails on login | login refused (fails closed) |
| Unhandled exception | 500 with an opaque reference ID; details go only to the server log |

## 2. Database schema

```
users ─────────────┐ 1
  id (uuid, PK)    │
  username UNIQUE  │                         quality_policies
  full_name        │                           version (PK, autoinc)  ◀── new row per change
  password_hash    │                           created_by → users.id
  role ENUM(operator, quality_manager, admin)  default_confidence_threshold
  site             │                           operator_min_threshold / operator_max_threshold
  is_active        │                           fail_confidence, fail_classes(JSON), ignored_classes(JSON)
  token_version    │                           max_allowed_defects, require_approved_model
  created_at/updated_at                        change_reason
                   │ N                                   ▲
inspections ◀──────┘                                     │ N:1
  id (uuid, PK), created_at, user_id → users.id, site     │
  image_sha256, image_width, image_height, annotated_image_file
  model_status, model_name, model_version, model_sha256
  confidence_threshold, threshold_overridden, policy_version ─┘
  ai_decision, ai_decision_reasons, defects_json, defect_count   ◀── ORIGINAL AI RESULT (never updated)
  processing_time_ms, inference_time_ms
  final_decision                                                 ◀── = latest override or ai_decision
     │ 1
     │ N
inspection_overrides
  id, inspection_id → inspections.id, actor_id → users.id, actor_role
  old_decision, new_decision, reason (CHECK length ≥ 10), created_at

model_versions
  id, version_label UNIQUE, weights_file, sha256, class_names(JSON), notes,
  approved, is_active, registered_by → users.id, created_at

audit_events                                   revoked_tokens
  seq (PK, 1..n, no gaps)                        jti (PK), expires_at
  event_id UNIQUE, timestamp (ISO-8601 text)
  actor_id, actor_role, action, target_type, target_id
  metadata_json (canonical JSON, sanitised)
  prev_hash, hash UNIQUE
```

CHECK constraints enforce valid decisions, the threshold range, a non-negative defect count, a
minimum reason length, and policy range consistency. SQLite foreign keys are switched on with
`PRAGMA foreign_keys=ON`.

## 3. Module responsibilities

| Concern | Module |
|---|---|
| Config / secrets | `app/config.py` (env vars, fails fast on a weak JWT secret) |
| Model loading (once) | `app/services/detector.py` (`DetectorManager`) |
| Inference + timing | `app/services/inference.py` |
| PASS/FAIL/REVIEW rules | `app/services/inspection_rules.py` (pure, unit-tested) |
| Orchestration + transactions | `app/services/inspection_service.py` |
| Password hashing / JWT | `app/services/authentication.py` |
| RBAC + object-level scope | `app/services/authorization.py` |
| Hash-chained audit log | `app/services/audit_service.py` + `app/utils/hashing.py` |
| Reports / Section C payload | `app/services/report_service.py` |
| Policy versions | `app/services/policy_service.py` |
| Login throttling | `app/services/rate_limiter.py` |

## 4. Assumptions
* One factory network, a few to a few dozen users, one API process (SQLite and the in-memory rate limiter assume this).
* Images come from file uploads or a browser webcam. No live industrial camera or PLC integration is implemented.
* Section A delivers a standard Ultralytics YOLO detection dataset (bounding boxes, not polygons).
* "Site" is the authorization scope for quality managers.
* Administrators do not run inspections or override decisions (separation of duties).
