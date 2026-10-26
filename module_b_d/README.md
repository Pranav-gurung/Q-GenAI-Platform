# dataset/

Put the Section A dataset here (it is git-ignored). Expected YOLO layout:

```
dataset/
  data.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
```

Validate before training:

```bash
python -m training.validate_dataset --data dataset/data.yaml
```

No dataset is included with the project. To test the pipeline without the real dataset, generate
a tiny synthetic **smoke-test** dataset (NOT for real inspections):

```bash
python -m training.make_smoke_dataset --out dataset_smoke --n 60
```

See `docs/INTEGRATION.md` for the full handover procedure.
