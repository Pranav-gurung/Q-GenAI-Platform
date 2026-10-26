"""Section A handover: the dataset validator must accept a correct YOLO
dataset and pinpoint common mistakes (no ML libraries needed)."""
from pathlib import Path

import yaml

from training.validate_dataset import validate_dataset
from tests.conftest import image_bytes


def make_ds(root: Path, labels: dict[str, str], names=("scratch", "crack")) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for stem, content in labels.items():
            (root / "images" / split / f"{stem}.jpg").write_bytes(image_bytes("JPEG"))
            if content is not None:
                (root / "labels" / split / f"{stem}.txt").write_text(content)
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"train": "images/train", "val": "images/val",
                                    "nc": len(names), "names": list(names)}))
    return data


def test_valid_dataset(tmp_path):
    rep = validate_dataset(make_ds(tmp_path, {"a": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n", "b": None}))
    assert rep["valid"], rep["errors"]
    assert rep["splits"]["train"]["instances"] == 2
    assert rep["splits"]["train"]["background_images"] == 1
    assert rep["class_names"] == ["scratch", "crack"]


def test_bad_labels_reported(tmp_path):
    rep = validate_dataset(make_ds(tmp_path, {"a": "5 0.5 0.5 0.2 0.2\n", "b": "0 1.5 0.5 0.2 0.2\n",
                                              "c": "0 0.5 0.5\n"}))
    assert not rep["valid"]
    text = " ".join(rep["errors"])
    assert "out of range" in text and "normalised" in text and "expected 5 values" in text


def test_missing_yaml_and_nc_mismatch(tmp_path):
    assert validate_dataset(tmp_path / "nope.yaml")["valid"] is False
    data = make_ds(tmp_path, {"a": "0 0.5 0.5 0.2 0.2\n"})
    cfg = yaml.safe_load(data.read_text())
    cfg["nc"] = 3
    data.write_text(yaml.safe_dump(cfg))
    assert any("nc=3" in e for e in validate_dataset(data)["errors"])


def test_orphan_label_detected(tmp_path):
    data = make_ds(tmp_path, {"a": "0 0.5 0.5 0.2 0.2\n"})
    (tmp_path / "labels" / "train" / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert any("without an image" in e for e in validate_dataset(data)["errors"])
