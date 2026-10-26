"""Module contracts. Every AI/analysis capability plugs into the pipeline through one of these.

    ImageQualityGate     ImageInput                   -> ImageQualityResult   (Part 3)
    CVEngine             ImageInput                   -> CVResult             (Part 2)
    DecisionEngine       CVResult (+ inspection info) -> QualityDecision      (Part 2/4)
    PipelineAnalyzer     PipelineContext              -> PipelineAnalysis     (Part 4)
    RCAEngine            PipelineContext              -> RCAResult            (Part 5)
    ActionEngine         RCAResult (+ context)        -> ActionRecommendation (Part 5)

Implementations must not depend on FastAPI or the database. A future YOLO engine is just

    class YoloCVEngine(CVEngine):
        name = "yolov8"
        def detect(self, image): ...

registered in backend/pipeline/registry.py and selected with CV_ENGINE=yolov8.
Raising NotImplementedError is also handled: the orchestrator records NOT_IMPLEMENTED.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from backend.pipeline.context import ImageInput, InspectionInfo, PipelineContext
from backend.schemas.results import (ActionRecommendation, CVResult, ImageQualityResult, PipelineAnalysis,
                                     QualityDecision, RCAResult)


class PipelineModule(ABC):
    name: ClassVar[str]
    #: False for placeholders; surfaced by /api/health so nobody mistakes a stub for a model.
    implemented: ClassVar[bool] = True


class ImageQualityGate(PipelineModule):
    @abstractmethod
    def assess(self, image: ImageInput) -> ImageQualityResult: ...


class CVEngine(PipelineModule):
    @abstractmethod
    def detect(self, image: ImageInput) -> CVResult: ...


class DecisionEngine(PipelineModule):
    @abstractmethod
    def decide(self, cv_result: CVResult, inspection: InspectionInfo) -> QualityDecision: ...


class PipelineAnalyzer(PipelineModule):
    @abstractmethod
    def analyze(self, context: PipelineContext) -> PipelineAnalysis: ...


class RCAEngine(PipelineModule):
    @abstractmethod
    def analyze(self, context: PipelineContext) -> RCAResult: ...


class ActionEngine(PipelineModule):
    @abstractmethod
    def recommend(self, rca_result: RCAResult, context: PipelineContext) -> ActionRecommendation: ...
