"""Placeholder modules for Part 1.

They return an explicit NOT_IMPLEMENTED result and never invent detections, scores or
decisions (the result schemas reject payload values on non-COMPLETED results anyway).
"""
from __future__ import annotations

from backend.pipeline.context import ImageInput, InspectionInfo, PipelineContext
from backend.pipeline.interfaces import (ActionEngine, CVEngine, DecisionEngine, ImageQualityGate,
                                         PipelineAnalyzer, RCAEngine)
from backend.schemas.results import (ActionRecommendation, CVResult, ImageQualityResult, PipelineAnalysis,
                                     QualityDecision, RCAResult)

STUB = "stub"


class StubImageQualityGate(ImageQualityGate):
    name = STUB
    implemented = False

    def assess(self, image: ImageInput) -> ImageQualityResult:
        return ImageQualityResult.not_implemented(STUB, "Placeholder image quality gate (real one: QUALITY_GATE_ENGINE=basic)")


class StubCVEngine(CVEngine):
    name = STUB
    implemented = False

    def detect(self, image: ImageInput) -> CVResult:
        return CVResult.not_implemented(STUB, "Placeholder CV engine (real one: CV_ENGINE=yolov8)")


class StubDecisionEngine(DecisionEngine):
    name = STUB
    implemented = False

    def decide(self, cv_result: CVResult, inspection: InspectionInfo) -> QualityDecision:
        if not cv_result.is_real:
            return QualityDecision.skipped(STUB, "No computer vision result to decide on")
        return QualityDecision.not_implemented(STUB, "Placeholder decision engine (real one: DECISION_ENGINE=rules)")


class StubPipelineAnalyzer(PipelineAnalyzer):
    name = STUB
    implemented = False

    def analyze(self, context: PipelineContext) -> PipelineAnalysis:
        return PipelineAnalysis.not_implemented(STUB, "Placeholder fulfillment analysis (real one: "
                                                      "PIPELINE_ANALYSIS_ENGINE=deterministic)")


class StubRCAEngine(RCAEngine):
    name = STUB
    implemented = False

    def analyze(self, context: PipelineContext) -> RCAResult:
        return RCAResult.not_implemented(STUB, "Root-cause analysis not implemented yet (planned: Part 5)")


class StubActionEngine(ActionEngine):
    name = STUB
    implemented = False

    def recommend(self, rca_result: RCAResult, context: PipelineContext) -> ActionRecommendation:
        return ActionRecommendation.not_implemented(STUB, "Corrective-action engine not implemented yet "
                                                          "(planned: Part 5)")
