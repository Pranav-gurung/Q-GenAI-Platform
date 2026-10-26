# Module C API Contract (Module B/D Integration)

**Status: RECONCILED & IMPLEMENTED.**
The schema defined below is implemented and validated by both Module C (`module_c/models.py::validate_request` and `module_c/api.py`)
and Module B/D (`module_b_d/app/services/report_service.py` and `module_b_d/app/routers/inspections.py`).
JSON Schema files are maintained at `schemas/rca_request.schema.json` and `schemas/rca_response.schema.json`.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/rca` | Analyse detector output, return document-grounded RCA suggestions |
| GET | `/health` | Liveness + index size, LLM configured?, supported classes (no secrets) |

## Request (`POST /rca`)
```json
{"inspection_id": "QC-0001", "image_id": "IMG-0001", "model_version": "v1", "status": "review_required",
 "defects": [{"class": "scratch", "confidence": 0.94, "bbox": [120, 80, 260, 190]}]}
```
Optional: `image_width`+`image_height` (sent together; enables bounds check), `part_type`, `material`,
`production_process`, `inspection_context` (<=300 chars each; treated as data in the LLM prompt), per-defect `mask_ref`.
Unknown top-level or defect fields are **rejected** (HTTP 422) so typos are caught; tell us if you need extra fields.

Validation: ids non-empty `[A-Za-z0-9._:-]{1,128}`; `confidence` finite in [0,1]; `bbox` exactly 4 finite numbers,
`[x_min,y_min,x_max,y_max]`, original-image pixels, non-negative, positive width/height (and inside the image if
dimensions are sent); `class` must be in the configured class list (`RCA_SUPPORTED_CLASSES` / `RCA_CLASS_MAP_PATH`;
**must match Module A's class map**); at most `RCA_MAX_DEFECTS` defects; **empty `defects` is valid**.
The original image is not needed.

*Module B/D compatibility:* Module B provides an adapter endpoint `GET /inspections/{inspection_id}/rca-request`
which maps inspection records directly into this schema, utilizing `insp.image_sha256` as `image_id`, converting
`{"x1", "y1", "x2", "y2"}` bounding boxes to `[x_min, y_min, x_max, y_max]`, and moving operational metadata into
`inspection_context`.

## Response and status semantics
| Situation | HTTP | `status` | Body |
|---|---|---|---|
| Analysed with evidence for every defect class | 200 | `success` | full RCA |
| No defects reported | 200 | `success` | observations only, no LLM/retrieval |
| No sufficiently relevant approved passage for >=1 class | 200 | `insufficient_evidence` | `defect_analyses[*].status`, `unknowns`, escalation action; **no LLM call for that class** |
| Invalid request (schema/bbox/confidence/unknown class/too many defects) | 422 | - | `{"detail": [{"loc":[...],"msg":"..."}], "request_id"}` |
| LLM rate-limited | 429 (+`Retry-After`) | `error` | RCA body with `error.code=llm_rate_limited` |
| LLM malformed / invalid output after retry | 502 | `error` | `error.code=llm_output_invalid` |
| LLM not configured / unavailable | 503 | `error` | `error.code=llm_not_configured` / `llm_unavailable` |
| LLM timeout | 504 | `error` | `error.code=llm_timeout` |
| Audit write failed and `RCA_AUDIT_FAIL_MODE=fail_closed` | 500 | `error` | `error.code=audit_failure`, result withheld |
| Auth hook rejects | 401/403 | - | `{"detail": ...}` |

`insufficient_evidence` is a **valid answer**, not an error: treat it as "send to a human".
Error bodies still contain `observations` (what the detector said) but never causes or citations.

## Meaning of the fields
- `observations` / `summary`: built from your request by the server. Never LLM-authored.
- `possible_causes[].status`: `supported_by_document` (cites >=1 real chunk), `hypothesis`, `requires_investigation`.
  A cause that claims document support without a valid citation is downgraded to `hypothesis` and noted in `notes`.
- `recommended_actions[].basis`: `document` (has citations) or `investigation_suggestion` (no document backing).
- `citations[]`: real indexed passages: `document_id, title, version, page, section, chunk_id, score, passage`.
  `possible_causes[].evidence` and `recommended_actions[].citations` contain `chunk_id`s that exist in `citations`.
- `unknowns`: what a human still has to establish. `limitations`: constant disclaimer to show in your UI.
- Defects of the same class in one request share one retrieval + LLM call; `defect_indices` map results back to your `defects[]` order.

**What the server verifies and what it cannot:** it guarantees that every citation is a real indexed passage and
that the model could not cite anything it was not given. It does **not** verify that the cited passage truly
entails the model's wording. Show the passage next to the claim and keep a human in the loop.

Example response (real output from the test suite: synthetic document + stubbed LLM; passage truncated):
```json
{
  "request_id": "dfaa5332f578440eb93e2eebdb84851e",
  "inspection_id": "QC-0001",
  "image_id": "IMG-0001",
  "model_version": "v1",
  "status": "success",
  "summary": "1 defect(s) of 1 class(es) reported by the detector; 1 class(es) analysed with document evidence, 0 without sufficient evidence.",
  "observations": [
    "Detector model v1 reported 'scratch' (defect #0) with confidence 0.94 at bbox [120.0, 80.0, 260.0, 190.0] (original-image pixels).",
    "Detector-reported inspection status: review_required."
  ],
  "possible_causes": [
    {
      "defect_class": "scratch",
      "defect_indices": [
        0
      ],
      "cause": "Contact abrasion during part transfer",
      "status": "supported_by_document",
      "evidence": [
        "TEST-SOP-001:p2:c0:fac77154b8"
      ]
    },
    {
      "defect_class": "scratch",
      "defect_indices": [
        0
      ],
      "cause": "Fixture wear",
      "status": "requires_investigation",
      "evidence": []
    }
  ],
  "recommended_actions": [
    {
      "defect_class": "scratch",
      "defect_indices": [
        0
      ],
      "action": "Review separator tray use per the handling procedure.",
      "citations": [
        "TEST-SOP-001:p2:c0:fac77154b8"
      ],
      "basis": "document"
    },
    {
      "defect_class": "scratch",
      "defect_indices": [
        0
      ],
      "action": "Inspect fixtures for burrs.",
      "citations": [],
      "basis": "investigation_suggestion"
    }
  ],
  "citations": [
    {
      "document_id": "TEST-SOP-001",
      "title": "Test Handling SOP",
      "version": "2.0",
      "page": 2,
      "section": "2. INSPECTION AFTER TRANSFER",
      "chunk_id": "TEST-SOP-001:p2:c0:fac77154b8",
      "score": 0.1076,
      "passage": "2. INSPECTION AFTER TRANSFER \nAfter each transfer, inspect housings for scratches and record findings. If scratches exceed \nthe test acceptance limit, quarantin..."
    }
  ],
  "defect_analyses": [
    {
      "defect_class": "scratch",
      "defect_indices": [
        0
      ],
      "n_candidates": 2,
      "n_relevant_passages": 1,
      "n_excluded_no_class_term": 0,
      "best_score": 0.1076,
      "status": "analyzed"
    }
  ],
  "unknowns": [
    "Which station produced the contact is not known."
  ],
  "notes": [],
  "limitations": "An image-based detection alone does not establish a root cause. Items labelled 'hypothesis' or 'requires_investigation' are unverified and must be confirmed by a qualified person.",
  "error": null,
  "generated_at": "2026-10-24T18:43:05+00:00"
}
```
