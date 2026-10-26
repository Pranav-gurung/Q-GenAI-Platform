"""Selects module implementations by name (env: QUALITY_GATE_ENGINE, CV_ENGINE, ...).

To plug in a real module, add it to the matching registry below, e.g.
    CV_ENGINES["yolov8"] = lambda settings: YoloCVEngine(weights=settings.model_path / "best.onnx")
Imports of heavy ML libraries should happen inside the factory so the API starts without them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from backend.core.config import Settings
from backend.pipeline import stubs
from backend.pipeline.interfaces import (ActionEngine, CVEngine, DecisionEngine, ImageQualityGate,
                                         PipelineAnalyzer, PipelineModule, RCAEngine)


@dataclass(frozen=True)
class PipelineModules:
    quality_gate: ImageQualityGate
    cv_engine: CVEngine
    decision_engine: DecisionEngine
    pipeline_analyzer: PipelineAnalyzer
    rca_engine: RCAEngine
    action_engine: ActionEngine

    def describe(self) -> dict[str, dict]:
        return {slot: {"name": m.name, "implemented": m.implemented}
                for slot, m in vars(self).items() if isinstance(m, PipelineModule)}


Factory = Callable[[Settings], PipelineModule]

QUALITY_GATES: dict[str, Factory] = {"stub": lambda s: stubs.StubImageQualityGate()}
CV_ENGINES: dict[str, Factory] = {"stub": lambda s: stubs.StubCVEngine()}
DECISION_ENGINES: dict[str, Factory] = {"stub": lambda s: stubs.StubDecisionEngine()}
PIPELINE_ANALYZERS: dict[str, Factory] = {"stub": lambda s: stubs.StubPipelineAnalyzer()}
RCA_ENGINES: dict[str, Factory] = {"stub": lambda s: stubs.StubRCAEngine()}
ACTION_ENGINES: dict[str, Factory] = {"stub": lambda s: stubs.StubActionEngine()}


def _pick(registry: dict[str, Factory], key: str, setting: str, settings: Settings) -> PipelineModule:
    try:
        return registry[key](settings)
    except KeyError:
        raise ValueError(f"{setting}={key!r} is not a registered implementation; "
                         f"available: {sorted(registry)}") from None


def build_modules(settings: Settings) -> PipelineModules:
    return PipelineModules(
        quality_gate=_pick(QUALITY_GATES, settings.quality_gate_engine, "QUALITY_GATE_ENGINE", settings),
        cv_engine=_pick(CV_ENGINES, settings.cv_engine, "CV_ENGINE", settings),
        decision_engine=_pick(DECISION_ENGINES, settings.decision_engine, "DECISION_ENGINE", settings),
        pipeline_analyzer=_pick(PIPELINE_ANALYZERS, settings.pipeline_analysis_engine,
                                "PIPELINE_ANALYSIS_ENGINE", settings),
        rca_engine=_pick(RCA_ENGINES, settings.rca_engine, "RCA_ENGINE", settings),
        action_engine=_pick(ACTION_ENGINES, settings.action_engine, "ACTION_ENGINE", settings),
    )
