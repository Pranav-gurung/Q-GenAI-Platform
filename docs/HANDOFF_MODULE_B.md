# Handoff to Module B (detection)

## 1. Using the dataset
After `python -m module_a.prepare_dataset ...` read `dataset_README.md` (generated; lists actual classes, split sizes, source, license, limitations).
- **Canonical** (always written): `data/annotations/<v>/canonical_{train,val,test}.json`
  ```python
  from module_a.schema import load_canonical
  records, class_map, meta = load_canonical("data/annotations/<v>/canonical_train.json")
  img_path = f"data/splits/<v>/train/images/{records[0].file_name}"
  for a in records[0].annotations: a.class_name, a.bbox   # bbox = [x_min, y_min, x_max, y_max], pixels, y down
  ```
  An image with an empty `annotations` list is defect-free.
- **COCO** (`--formats coco`): `data/processed/<v>/coco/instances_<split>.json`; `bbox=[x,y,w,h]`, category ids start at **1**.
  Image paths are relative to `data/splits/<v>/<split>/images/`.
- **YOLO** (`--formats yolo`): `data/processed/<v>/yolo/{images,labels}/<split>` + `data.yaml`; class ids start at **0**.
- **Anomaly folders** (`--formats anomaly`): `<split>/good|<class>/...`.
- `val`/`test` contain only un-augmented originals. Augmented variants (`*__augNN.png`) are train-only and keep their source's group.
- Check integrity with `module_a.version_dataset.verify_manifest(manifest, "data/splits/<v>")`. Record the dataset version id with every trained model.
- **Automated Handover Gate (Module A -> Module B)**:
  Module B provides an automated validator in `training/handover.py`:
  ```python
  from training.handover import verify_handover
  report = verify_handover(yolo_dir, manifest_path=manifest_path)
  assert report.valid is True
  assert report.leakage_free is True
  ```
  This validates `data.yaml`, image/label correspondence, bounding-box normalization, manifest alignment, and cross-split image/group leakage (SHA-256 duplicate detection).
- Tell Module A which format you need if it is not listed (segmentation is not implemented).
- Confirm the class list: Module C validates `class` against `RCA_SUPPORTED_CLASSES` / `RCA_CLASS_MAP_PATH` (point this at a canonical annotation file or JSON class map).

## 2. Calling the RCA service
See `docs/API_CONTRACT.md`. Module B/D exposes `GET /inspections/{inspection_id}/rca-request` which automatically formats
an inspection into Module C's exact schema:
- Maps `insp.image_sha256` to `image_id` (content-addressable, stable).
- Maps bounding boxes from `{"x1", "y1", "x2", "y2"}` to `[x_min, y_min, x_max, y_max]` in original-image pixels.
- Maps `insp.final_decision` to status (`PASS` -> `nominal`, `FAIL` -> `defect_detected`, `REVIEW` -> `review_required`).
- Preserves operational metadata safely in `inspection_context`.
- Enforces Module C supported defect classes (`scratch, crack, dent, corrosion`).

Minimal call from Module B to Module C:
```python
import requests

# 1. Fetch compliant payload from Module B/D
rca_req = requests.get(f"http://localhost:8000/inspections/{inspection_id}/rca-request", headers=auth_headers).json()

# 2. Dispatch to Module C RCA service
r = requests.post("http://localhost:8001/rca", json=rca_req, timeout=60)
if r.status_code == 200:
    body = r.json()   # body["status"] in {"success", "insufficient_evidence"}
elif r.status_code == 422:
    ...               # validation error; r.json()["detail"] lists the problems
else:
    ...               # error; retry later
```
Always display `limitations` and the cited passages next to any suggested cause. Send boxes in **original-image pixels** (not resized/normalised).
Questions for you: final field names, whether you can send `image_width/height`, `part_type/material/process`, mask support, auth scheme.
