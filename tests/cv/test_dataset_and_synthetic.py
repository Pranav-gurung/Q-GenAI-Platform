import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ml.cv import validate_dataset as cli
from ml.cv.dataset import validate_dataset
from ml.cv.synthetic import DISCLAIMER, generate
from ml.cv.taxonomy import load_taxonomy

TAX = load_taxonomy()


@pytest.fixture(scope="module")
def base_dataset(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synth") / "ds"
    generate(out, n_images=20, seed=3, size=160)
    return out


@pytest.fixture
def ds(base_dataset, tmp_path) -> Path:
    dst = tmp_path / "ds"
    shutil.copytree(base_dataset, dst)
    return dst


def codes(path):
    return {i.code for i in validate_dataset(path / "dataset.yaml", TAX).errors}


def first_label(ds, split="train"):
    return sorted((ds / "labels" / split).glob("*.txt"))[0]


def test_generated_dataset_is_valid_and_labelled_synthetic(ds):
    report = validate_dataset(ds / "dataset.yaml", TAX)
    assert report.ok, [i.__dict__ for i in report.errors]
    assert report.images == {"train": 14, "val": 3, "test": 3}
    assert sum(report.instances["train"].values()) > 0
    assert DISCLAIMER in (ds / "dataset.yaml").read_text()
    assert json.loads((ds / "MANIFEST.json").read_text())["synthetic"] is True
    warn = {i.code for i in report.warnings}
    assert "CLASS_WITHOUT_TRAINING_DATA" in warn  # placeholder classes are flagged
    assert cli.main(["--data", str(ds / "dataset.yaml")]) == 0


def test_relative_dataset_path(ds, monkeypatch):
    monkeypatch.chdir(ds.parent)
    assert validate_dataset(Path("ds/dataset.yaml"), TAX).ok
    assert cli.main(["--data", "ds/dataset.yaml"]) == 0


def test_generator_is_deterministic(tmp_path):
    def digest(p):
        return hashlib.sha256(b"".join(f.read_bytes() for f in sorted(p.rglob("*")) if f.is_file())).hexdigest()
    generate(tmp_path / "a", 6, seed=11, size=128)
    generate(tmp_path / "b", 6, seed=11, size=128)
    generate(tmp_path / "c", 6, seed=12, size=128)
    assert digest(tmp_path / "a") == digest(tmp_path / "b") != digest(tmp_path / "c")


def test_missing_image(ds):
    img = sorted((ds / "images" / "train").glob("*.jpg"))[0]
    img.unlink()
    assert "MISSING_IMAGE_FILE" in codes(ds)


def test_missing_label(ds):
    first_label(ds).unlink()
    assert "MISSING_LABEL_FILE" in codes(ds)


@pytest.mark.parametrize("line,code", [
    ("0 0.5 0.5 0.2", "MALFORMED_LABEL"),
    ("x 0.5 0.5 0.2 0.2", "MALFORMED_LABEL"),
    ("17 0.5 0.5 0.2 0.2", "INVALID_CLASS_ID"),
    ("1.5 0.5 0.5 0.2 0.2", "INVALID_CLASS_ID"),
    ("1 0.95 0.5 0.2 0.2", "BBOX_OUT_OF_BOUNDS"),
    ("1 0.5 0.5 1.2 0.2", "BBOX_OUT_OF_BOUNDS"),
    ("1 0.5 0.5 0 0.2", "ZERO_AREA_BBOX"),
    ("1 0.5 0.5 nan 0.2", "MALFORMED_BBOX"),
])
def test_bad_annotations(ds, line, code):
    first_label(ds).write_text(line + "\n")
    assert code in codes(ds)
    assert cli.main(["--data", str(ds / "dataset.yaml")]) == 1


def test_corrupted_and_unsupported_images(ds):
    img = sorted((ds / "images" / "train").glob("*.jpg"))[0]
    img.write_bytes(img.read_bytes()[:200])
    (ds / "images" / "train" / "notes.gif").write_bytes(b"GIF89a")
    assert {"CORRUPT_IMAGE", "UNSUPPORTED_IMAGE_FORMAT"} <= codes(ds)


def test_duplicates_are_warnings(ds):
    lbl = first_label(ds)
    lbl.write_text("1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.201 0.2\n")
    test_img = sorted((ds / "images" / "test").glob("*.jpg"))[0]
    shutil.copy(sorted((ds / "images" / "train").glob("*.jpg"))[0], test_img)
    report = validate_dataset(ds / "dataset.yaml", TAX)
    warn = {i.code for i in report.warnings}
    assert {"DUPLICATE_ANNOTATION", "SUSPICIOUS_OVERLAP", "CROSS_SPLIT_DUPLICATE"} <= warn
    assert report.ok
    assert cli.main(["--data", str(ds / "dataset.yaml"), "--strict"]) == 1


def test_class_names_must_match_taxonomy(ds):
    y = ds / "dataset.yaml"
    y.write_text(y.read_text().replace("leakage", "spill"))
    assert codes(ds) == {"CLASS_NAMES_MISMATCH"}


def test_missing_yaml(tmp_path):
    assert codes(tmp_path) == {"MISSING_DATASET_YAML"}
