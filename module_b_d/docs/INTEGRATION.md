# Integration with Section A (dataset) and Section C (root-cause analysis)

## Section A → Section B: dataset and training

### Expected format (Ultralytics YOLO detection)
```
dataset/
  data.yaml
  images/train/*.jpg|png    labels/train/*.txt
  images/val/...            labels/val/...
  images/test/...           labels/test/...      (optional but recommended)
```
`data.yaml`:
```yaml
path: /absolute/or/relative/path/to/dataset   # optional; defaults to the yaml's folder
train: images/train
val: images/val
test: images/test
nc: 5
names: [scratch, crack, rust, dent, weld_defect]
```
Each label line is `class_id cx cy w h`, normalised to 0–1. One `.txt` per image with the same stem.
Images without a label file count as defect-free background images.

### Handover checklist for the Section A teammate
1. Copy or symlink the dataset to `dataset/` (or anywhere) and run:
   `python -m training.validate_dataset --data dataset/data.yaml`. It must print `RESULT: VALID`.
2. Train: `python -m training.train --data dataset/data.yaml --model yolov8n.pt --epochs 100 --output defects_v1`
   (use `--device 0` on an NVIDIA GPU).
3. Evaluate on the held-out test split:
   `python -m training.evaluate --weights models/defects_v1.pt --data dataset/data.yaml --split test`.
   Report **these measured numbers**. Do not quote metrics that were not measured.
4. Admin UI → **Policy & models** → register `defects_v1.pt` with label `defects-v1`, approve it,
   and activate it. Or set `QGENAI_MODEL_WEIGHTS=defects_v1.pt` and restart.
5. Optionally set the policy's `fail_classes` / `ignored_classes` using the new class names.

The inference API and JSON schema stay the same. Class names come from the weights themselves.

**Synthetic data caveat:** a model trained only on synthetic images must be evaluated on **real**
product photos before anyone relies on it. Keep a real-image test split.

## Section B → Section C: structured inspection results

### Option 1: REST (recommended)
`GET /inspections/{inspection_id}/rca-payload` with a bearer token for a service account. Create a
`quality_manager` user for the RCA service with the correct `site`.

```json
{
  "inspection_id": "6ff18679-52e8-4983-8780-69f006c2ce00",
  "timestamp": "2026-10-26T04:59:57.516215Z",
  "site": "plant-pune",
  "model_version": "defects-v1",
  "model_sha256": "…64 hex…",
  "policy_version": 2,
  "confidence_threshold": 0.5,
  "ai_decision": "FAIL",
  "final_decision": "FAIL",
  "defects": [
    {"class_id": 1, "class_name": "crack", "confidence": 0.87,
     "bbox": {"x1": 120.0, "y1": 80.0, "x2": 250.0, "y2": 160.0}}
  ],
  "image_width": 1280,
  "image_height": 960
}
```
The payload contains **no image data and no user identity**. Section C must not send product
images to external LLM services. If it needs images, use an on-premise model and get the
quality manager's approval.

To get a batch of failed parts for trend analysis, call `GET /inspections?ai_decision=FAIL&limit=500`
and then fetch the payload for each one.

### Option 2: in-process Python
```python
from app.database import SessionLocal
from app.models.inspection import Inspection
from app.services.report_service import build_rca_payload

with SessionLocal() as db:
    payload = build_rca_payload(db.get(Inspection, inspection_id))
```
This bypasses HTTP authorization, so use it only inside trusted backend code.

### What Section C should NOT do
* Treat `confidence` as severity. It is the model's certainty that the box is that class.
* Present any output as BIS/ISO certification.
* Write back into `inspections`. If RCA results need storing, add an `rca_reports` table that
  references `inspection_id`, and audit it with `audit_service.append_event`.
