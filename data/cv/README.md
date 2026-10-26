# data/cv: object-detection datasets

Everything in this directory except this README is git-ignored. Datasets are large and may be licensed.

## Layout

One directory per dataset version:

```
data/cv/<dataset>/
├── dataset.yaml          Ultralytics data file: split dirs + class names (generated from ml/cv/taxonomy.yaml)
├── MANIFEST.json         provenance (generator, seed, synthetic flag, instance counts)
├── images/
│   ├── train/  *.jpg | *.jpeg | *.png | *.webp
│   ├── val/
│   └── test/
└── labels/
    ├── train/  <same stem as image>.txt
    ├── val/
    └── test/
```

## Annotation format (YOLO)

One line per defect box:

```
<class_id> <x_center> <y_center> <width> <height>
```

* All four coordinates are **normalised to 0–1** relative to the image width/height, with the origin at the top-left.
* `class_id` is the `id` from `ml/cv/taxonomy.yaml` (0 = crushed_packaging … 9 = visual_anomaly).
* An image with **no defects must still have a label file**, left empty. This makes "no defect" an explicit,
  reviewed negative instead of a forgotten annotation.
* The model's API output uses **pixel** coordinates (`x_min, y_min, x_max, y_max`) of the image after EXIF
  rotation. Normalised coordinates appear only in these training files.

## Class names

`dataset.yaml` `names` must equal the taxonomy list exactly, in order. The validator enforces this.
See the taxonomy table in the main README for which classes currently have data.

## Splits

* `train` is used for fitting, `val` for model selection / early stopping, and `test` is held out for the final report.
* Split **before** augmenting, and keep near-duplicate photos (same product, same session) in one split.
  The validator warns on byte-identical images that appear in more than one split.
* The synthetic generator uses 70 / 15 / 15.

## Adding real data

1. Collect photos you are authorised to use. Record their source and licence in `MANIFEST.json`.
2. Annotate with any tool that exports YOLO format (CVAT, Label Studio, Roboflow, …) using the taxonomy class ids.
3. Put the files in a **new** directory (e.g. `data/cv/real-v1/`), copy a `dataset.yaml` with the same `names`, and
   set `"synthetic": false` in its `MANIFEST.json`.
4. Validate:
   ```bash
   python -m ml.cv.validate_dataset --data data/cv/real-v1/dataset.yaml --report data/cv/real-v1/validation.json
   ```
   Exit code 0 means valid and 1 means errors. `--strict` also fails on warnings.
5. Train a new model version (`python -m ml.cv.train --data … --version …`). Never overwrite an existing version.
6. Once a class has real, evaluated data, change its `data_status` in `ml/cv/taxonomy.yaml` to `real`.

## What the validator checks

| Errors (dataset rejected) | Warnings |
|---|---|
| missing `dataset.yaml` / split dirs, class names ≠ taxonomy | duplicate boxes, near-duplicate same-class boxes (IoU > 0.95) |
| unsupported image format, corrupt image | boxes < 2 px |
| image without label file, label file without image | identical images across splits |
| malformed lines, invalid class ids, non-finite values | classes with zero training instances |
| boxes outside the image, zero-area boxes | |

## Synthetic demo dataset

```bash
python -m ml.cv.synthetic --out data/cv/synthetic-v1 --images 800 --seed 42
```

**SYNTHETIC / DEMO DATA.** These are procedurally drawn cartoon scenes of cartons, cans and pouches with drawn-on
defects (torn_packaging, leakage, dented_container, damaged_box, missing_label). The generator is deterministic
for a given seed. Use it to build and demonstrate the pipeline. Never use it to claim real-world accuracy.

## Model artifacts

Trained models go to `artifacts/cv/<model-version>/` (git-ignored), never to `data/`. See the main README.
