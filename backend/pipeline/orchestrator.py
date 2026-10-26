"""PipelineOrchestrator: runs the inspection pipeline stage by stage.

    IMAGE_RECEIVED -> QUALITY_CHECK -> CV_PROCESSING -> DECISION -> PIPELINE_ANALYSIS
                   -> RCA_PROCESSING -> ACTION_GENERATED -> COMPLETED | REQUIRES_REVIEW | FAILED

Routing rules (all based on REAL results only):
  * quality gate says image not acceptable        -> REQUIRES_REVIEW (IMAGE_QUALITY_REJECTED)
  * decision PASS                                 -> COMPLETED (no RCA / action needed)
  * decision REVIEW                               -> REQUIRES_REVIEW (DECISION_REVIEW)
  * decision FAIL and a corrective action exists  -> COMPLETED
  * no real decision (e.g. modules not implemented yet) -> REQUIRES_REVIEW (NO_AUTOMATED_DECISION):
    the system never auto-passes or auto-fails a product it could not actually inspect.
  * a module raises / violates its contract       -> FAILED (MODULE_ERROR / MODULE_CONTRACT_VIOLATION)

A module that raises NotImplementedError is recorded as NOT_IMPLEMENTED (not a failure).
A module that returns UNAVAILABLE (e.g. CV model artifact missing) is recorded with error code
MODULE_UNAVAILABLE and logged as a warning; it produced no result, so routing continues as above
and the inspection ends in REQUIRES_REVIEW rather than ever being auto-passed.
The orchestrator knows nothing about HTTP or the database.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from backend.core.logging import log_pipeline
from backend.domain.enums import DecisionOutcome, ModuleStatus
from backend.pipeline.context import PipelineContext
from backend.pipeline.interfaces import PipelineModule
from backend.pipeline.registry import PipelineModules
from backend.pipeline.states import PipelineState as S
from backend.schemas.results import (ActionRecommendation, CVResult, ImageQualityResult, ModuleResult,
                                     PipelineAnalysis, QualityDecision, RCAResult)

logger = logging.getLogger(__name__)
R = TypeVar("R", bound=ModuleResult)


class _StageFailed(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class PipelineOrchestrator:
    def __init__(self, modules: PipelineModules) -> None:
        self.modules = modules

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.state is not S.IMAGE_RECEIVED or ctx.image is None:
            raise ValueError(f"pipeline can only start from IMAGE_RECEIVED with an image (state={ctx.state.value})")
        m = self.modules
        started = time.perf_counter()
        log_pipeline(ctx.inspection_id, "PIPELINE", "STARTED")
        try:
            quality = self._stage(ctx, S.QUALITY_CHECK, "quality", m.quality_gate, ImageQualityResult,
                                  lambda: m.quality_gate.assess(ctx.image))
            if quality.is_real and quality.acceptable is False:
                return self._finish(ctx, S.REQUIRES_REVIEW, started, "IMAGE_QUALITY_REJECTED",
                                    "Image rejected by quality gate; retake required")

            self._stage(ctx, S.CV_PROCESSING, "cv", m.cv_engine, CVResult, lambda: m.cv_engine.detect(ctx.image))

            decision = self._stage(ctx, S.DECISION, "decision", m.decision_engine, QualityDecision,
                                   lambda: m.decision_engine.decide(ctx.cv, ctx.inspection))
            if decision.is_real and decision.outcome is DecisionOutcome.PASS:
                return self._finish(ctx, S.COMPLETED, started, message="Decision PASS")
            if decision.is_real and decision.outcome is DecisionOutcome.REVIEW:
                return self._finish(ctx, S.REQUIRES_REVIEW, started, "DECISION_REVIEW",
                                    decision.reason or "Decision engine requested human review")

            self._stage(ctx, S.PIPELINE_ANALYSIS, "pipeline_analysis", m.pipeline_analyzer, PipelineAnalysis,
                        lambda: m.pipeline_analyzer.analyze(ctx))
            self._stage(ctx, S.RCA_PROCESSING, "rca", m.rca_engine, RCAResult, lambda: m.rca_engine.analyze(ctx))
            action = self._stage(ctx, S.ACTION_GENERATED, "action", m.action_engine, ActionRecommendation,
                                 lambda: m.action_engine.recommend(ctx.rca, ctx))

            if not decision.is_real:
                return self._finish(ctx, S.REQUIRES_REVIEW, started, "NO_AUTOMATED_DECISION",
                                    "No automated decision available; operator review required")
            if not action.is_real:
                return self._finish(ctx, S.REQUIRES_REVIEW, started, "NO_CORRECTIVE_ACTION",
                                    "Defect confirmed but no corrective action generated; operator review required")
            return self._finish(ctx, S.COMPLETED, started, message="Decision FAIL with corrective action")
        except _StageFailed as exc:
            ctx.record_error(exc.code, exc.message)
            failed_stage = ctx.state
            ctx.transition_to(S.FAILED, message=exc.message, error_code=exc.code)
            log_pipeline(ctx.inspection_id, failed_stage.value, "FAILED", level=logging.ERROR,
                         duration_ms=_ms(started), error_code=exc.code, message=exc.message)
            return ctx

    # ------------------------------------------------------------------ internals
    def _stage(self, ctx: PipelineContext, state: S, slot: str, module: PipelineModule,
               expected: type[R], call: Callable[[], R]) -> R:
        ctx.transition_to(state)
        log_pipeline(ctx.inspection_id, state.value, "STARTED", module=module.name)
        t0 = time.perf_counter()
        failure: _StageFailed | None = None
        try:
            result = call()
        except NotImplementedError:
            result = expected.not_implemented(module.name, f"{type(module).__name__} is not implemented")
        except Exception:
            # Full traceback goes to the log only; the stored/API message stays generic.
            logger.exception("module %s raised during %s", module.name, state.value)
            result = expected.failed(module.name, "Module raised an error")
            failure = _StageFailed("MODULE_ERROR", f"{state.value}: module '{module.name}' raised an error")
        else:
            if not isinstance(result, expected):
                logger.error("module contract violation: %s returned %s, expected %s",
                             type(module).__name__, type(result).__name__, expected.__name__)
                result = expected.failed(module.name, "Module returned an invalid result")
                failure = _StageFailed("MODULE_CONTRACT_VIOLATION",
                                       f"{state.value}: module returned an invalid result")

        duration = _ms(t0)
        if result.processing_time_ms is None:
            result = result.model_copy(update={"processing_time_ms": duration})
        setattr(ctx, slot, result)
        unavailable = result.status is ModuleStatus.UNAVAILABLE
        code = failure.code if failure else ("MODULE_UNAVAILABLE" if unavailable else None)
        ctx.record_module(module.name, result.status, duration, result.message, code)
        log_pipeline(ctx.inspection_id, state.value, result.status.value, module=module.name,
                     duration_ms=duration, error_code=code,
                     message=(result.message or "") if unavailable else "",
                     level=logging.ERROR if failure else (logging.WARNING if unavailable else logging.INFO))
        if failure:
            raise failure
        return result

    def _finish(self, ctx: PipelineContext, target: S, started: float, code: str | None = None,
                message: str | None = None) -> PipelineContext:
        ctx.transition_to(target, message=message, error_code=code)
        log_pipeline(ctx.inspection_id, "PIPELINE", target.value, duration_ms=_ms(started),
                     error_code=code, message=message or "")
        return ctx


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
