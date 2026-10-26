import json
import shutil
from pathlib import Path
import pytest
import yaml

import sys
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from module_a.fixtures import FIXTURE_CLASSES, generate_fixture
from module_a.prepare_dataset import AugmentConfig, prepare
from training.handover import verify_handover


@pytest.fixture
def module_a_yolo_dataset(tmp_path):
    """Generate an authentic Module A YOLO export using Module A's prepare pipeline."""
    fx_dir = tmp_path / "raw_fixture"
    img_dir, ann_csv = generate_fixture(fx_dir, n_parts=10, views_per_part=3, seed=42)

    out_dir = tmp_path / "module_a_out"
    res = prepare(
        image_dir=img_dir,
        annotations=ann_csv,
        classes=FIXTURE_CLASSES,
        out_root=out_dir,
        source={"name": "test_fixture", "path": str(img_dir), "is_test_fixture": True},
        export_formats=("yolo",),
        seed=42,
        augment=AugmentConfig(seed=42, variants_per_image=1),
        readme_path=tmp_path / "README.md",
    )
    yolo_dir = Path(res["layout"]["processed_dir"]) / "yolo"
    manifest_path = Path(res["manifest"])
    return yolo_dir, manifest_path


def test_handover_authentic_module_a_export_succeeds(module_a_yolo_dataset):
    yolo_dir, manifest_path = module_a_yolo_dataset
    report = verify_handover(yolo_dir, manifest_path=manifest_path)

    assert report.valid is True
    assert report.leakage_free is True
    assert report.manifest_verified is True
    assert len(report.errors) == 0
    assert report.class_names == FIXTURE_CLASSES
    assert "train" in report.splits
    assert "val" in report.splits
    assert report.splits["train"]["images"] > 0
    assert report.splits["val"]["images"] > 0


def test_handover_missing_data_yaml(tmp_path):
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    report = verify_handover(empty_dir)

    assert report.valid is False
    assert any("data.yaml not found" in e for e in report.errors)


def test_handover_class_map_mismatch_with_manifest(module_a_yolo_dataset, tmp_path):
    yolo_dir, manifest_path = module_a_yolo_dataset

    # Corrupt manifest class_map
    m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    m_data["class_map"] = {"corrupted_class": 0, "another_class": 1}
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps(m_data), encoding="utf-8")

    report = verify_handover(yolo_dir, manifest_path=bad_manifest)
    assert report.valid is False
    assert any("Class names mismatch between manifest" in e for e in report.errors)


def test_handover_detects_duplicate_image_leakage(module_a_yolo_dataset):
    yolo_dir, manifest_path = module_a_yolo_dataset

    # Deliberately copy an image from train into val (identical content hash)
    train_imgs = list((yolo_dir / "images" / "train").glob("*.png"))
    val_img_dir = yolo_dir / "images" / "val"
    val_lbl_dir = yolo_dir / "labels" / "val"

    assert len(train_imgs) > 0
    src_img = train_imgs[0]
    dst_img = val_img_dir / f"leaked_{src_img.name}"
    shutil.copy2(src_img, dst_img)

    # Provide corresponding empty label in val
    (val_lbl_dir / f"leaked_{src_img.stem}.txt").write_text("")

    report = verify_handover(yolo_dir, manifest_path=manifest_path)
    assert report.valid is False
    assert report.leakage_free is False
    assert any("DATA LEAKAGE: Identical image content" in e for e in report.errors)


def test_handover_detects_group_part_leakage(module_a_yolo_dataset):
    yolo_dir, manifest_path = module_a_yolo_dataset

    # Find a train image with partXXX name and copy to val with new content but same part prefix
    train_imgs = list((yolo_dir / "images" / "train").glob("part*.png"))
    assert len(train_imgs) > 0
    target_part = train_imgs[0].name.split("_")[0]  # e.g. "part001"

    val_img_dir = yolo_dir / "images" / "val"
    val_lbl_dir = yolo_dir / "labels" / "val"

    # Create new file with different bytes but same part prefix
    leaked_file = val_img_dir / f"{target_part}_view99.png"
    leaked_file.write_bytes(b"distinct-content-so-not-hash-identical")
    (val_lbl_dir / f"{target_part}_view99.txt").write_text("")

    report = verify_handover(yolo_dir, manifest_path=manifest_path)
    assert report.valid is False
    assert report.leakage_free is False
    assert any("DATA LEAKAGE: Images from the same group/part" in e for e in report.errors)


def test_handover_malformed_bounding_boxes(module_a_yolo_dataset):
    yolo_dir, manifest_path = module_a_yolo_dataset

    # Corrupt a label file with coordinate > 1.0
    val_labels = list((yolo_dir / "labels" / "val").glob("*.txt"))
    assert len(val_labels) > 0
    # Overwrite label with out-of-bounds coordinate
    val_labels[0].write_text("0 0.5 0.5 1.5 0.5\n")

    report = verify_handover(yolo_dir, manifest_path=manifest_path)
    assert report.valid is False
    assert any("coordinates must be normalised to [0, 1]" in e for e in report.errors)


def test_handover_orphan_label_file(module_a_yolo_dataset):
    yolo_dir, manifest_path = module_a_yolo_dataset

    # Add label file without corresponding image
    orphan_lbl = yolo_dir / "labels" / "val" / "orphan_without_image.txt"
    orphan_lbl.write_text("0 0.5 0.5 0.2 0.2\n")

    report = verify_handover(yolo_dir, manifest_path=manifest_path)
    assert report.valid is False
    assert any("without an image" in e for e in report.errors)
