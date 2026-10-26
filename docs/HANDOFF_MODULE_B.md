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
- Tell Module A which format you need if it is not listed (segmentation is not implemented).
- Confirm the class list: Module C validates `class` against `RCA_SUPPORTED_CLASSES` / `RCA_CLASS_MAP_PATH` (point this at a canonical annotation file or JSON class map).

## 2. Calling the RCA service
See `docs/API_CONTRACT.md`. Minimal call:
```python
import requests
r = requests.post("http://localhost:8001/rca", json={
    "inspection_id": "QC-0001", "image_id": "IMG-0001", "model_version": "v1", "status": "review_required",
    "defects": [{"class": "scratch", "confidence": 0.94, "bbox": [120, 80, 260, 190]}]}, timeout=60)
if r.status_code == 200:  body = r.json()   # body["status"] in {"success","insufficient_evidence"}
elif r.status_code == 422: ...              # fix the request; r.json()["detail"] lists the problems
else: ...                                   # 429/502/503/504/500: body["status"]=="error"; retry later / show detector result only
```
Always display `limitations` and the cited passages next to any suggested cause. Send boxes in **original-image pixels** (not resized/normalised).
Questions for you: final field names, whether you can send `image_width/height`, `part_type/material/process`, mask support, auth scheme.
