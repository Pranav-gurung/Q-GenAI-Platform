"""Fixtures for tests that exercise REAL Ultralytics inference.

`fixture_model` builds a YOLOv8n with the taxonomy's 10 classes and seeded random weights. It is a
genuine model (real preprocessing, forward pass, NMS) whose outputs are meaningless, which is
exactly what we need to test the engine mechanics deterministically without shipping weights.
Tests that need a model that actually detects defects use the trained demo artifact and are
skipped when it has not been trained (see README "Computer Vision").
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]
DEMO_ARTIFACT = ROOT / "artifacts/cv/qgenai-yolov8n-synthetic-v1/model/best.pt"
DEMO_DATASET = ROOT / "data/cv/synthetic-v1/dataset.yaml"
FIXTURE_VERSION = "fixture-untrained-v0"


def build_fixture_model(path: Path, names: list[str], imgsz: int = 160, seed: int = 0) -> Path:
    import torch

    from ml.cv._ultralytics import import_ultralytics
    import_ultralytics()
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(seed)
    model = DetectionModel("yolov8n.yaml", nc=len(names), verbose=False)
    model.names = dict(enumerate(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "train_args": {"imgsz": imgsz, "model": "yolov8n.yaml", "task": "detect"},
                "date": None, "version": "fixture"}, path)
    return path


@pytest.fixture(scope="session")
def fixture_model(tmp_path_factory) -> Path:
    pytest.importorskip("ultralytics")
    from ml.cv.taxonomy import load_taxonomy

    root = tmp_path_factory.mktemp("artifacts") / FIXTURE_VERSION
    weights = build_fixture_model(root / "model" / "best.pt", load_taxonomy().names)
    (root / "metadata.json").write_text(json.dumps({"model_name": "yolov8n", "model_version": FIXTURE_VERSION,
                                                    "data_provenance": "none (untrained test fixture)"}))
    return weights


@pytest.fixture
def cv_settings(fixture_model, tmp_path) -> Settings:
    return Settings(_env_file=None, cv_engine="yolov8", cv_model_path=fixture_model, cv_device="cpu",
                    quality_gate_engine="basic", decision_engine="rules", environment="test",
                    database_url=f"sqlite:///{tmp_path / 'cv.db'}", storage_path=tmp_path / "images",
                    sample_data_path=ROOT / "data/samples/dark_store_sample.json", log_level="WARNING")
