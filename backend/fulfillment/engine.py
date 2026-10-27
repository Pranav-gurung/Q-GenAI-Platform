"""PIPELINE_ANALYSIS stage implementation (PIPELINE_ANALYSIS_ENGINE=deterministic).

Reads the current inspection's own results from the PipelineContext plus the prior observations of the same
trace (loaded from the database by the inspection service before the run). No DB access, no ML, no LLM.
"""
from __future__ import annotations

from backend.core.logging import log_pipeline
from backend.domain.enums import ModuleStatus
from backend.fulfillment.analyzer import ANALYZER_VERSION, analyze_trace
from backend.fulfillment.observations import build_observation, trace_key_for
from backend.pipeline.context import PipelineContext
from backend.pipeline.interfaces import PipelineAnalyzer
from backend.schemas.fulfillment import AttributionStatus, FulfillmentTraceAnalysis
from backend.schemas.results import PipelineAnalysis


def log_analysis(inspection_id: str, analysis: FulfillmentTraceAnalysis) -> None:
    for a in analysis.defect_analyses:
        log_pipeline(inspection_id, "PIPELINE_ANALYSIS", "DEFECT_ANALYSED",
                     message=f"defect={a.defect_type.value} first_observed={a.first_observed_stage.value} "
                             f"classification={a.classification.value} evidence={a.evidence_level.value} "
                             f"attribution={a.attribution_status.value} code={a.explanation_code.value}")
    log_pipeline(inspection_id, "PIPELINE_ANALYSIS", "ANALYSED",
                 message=f"trace={analysis.trace.key_type.value} observations={len(analysis.observations)} "
                         f"stages={[s.value for s in analysis.inspected_stages]} status={analysis.status.value} "
                         f"code={analysis.explanation_code.value}")


class DeterministicPipelineAnalyzer(PipelineAnalyzer):
    name = "deterministic"

    def health(self) -> dict:
        return {"available": True, "version": ANALYZER_VERSION}

    def analyze(self, context: PipelineContext) -> PipelineAnalysis:
        info = context.inspection
        meta = context.image.metadata if context.image else None
        current = build_observation(
            inspection_id=info.inspection_id, stage=info.stage, observed_at=meta.created_at if meta else None,
            store_id=info.store_id, sku_id=info.sku_id, item_id=info.item_id, order_id=info.order_id,
            decision=context.decision, cv=context.cv, image_id=meta.image_id if meta else None,
            image_sha256=meta.sha256 if meta else None)
        trace = trace_key_for(info.store_id, info.sku_id, info.item_id, info.order_id)
        log_pipeline(info.inspection_id, "PIPELINE_ANALYSIS", "OBSERVATIONS_LOADED",
                     message=f"trace={trace.key_type.value} prior_observations={len(context.trace_observations)}")
        analysis = analyze_trace(trace, [*context.trace_observations, current],
                                 current_inspection_id=info.inspection_id, current_stage=info.stage)
        log_analysis(info.inspection_id, analysis)

        attributed_here = {a.attributed_stage for a in analysis.defect_analyses
                           if a.attribution_status is AttributionStatus.ATTRIBUTED
                           and a.current_stage_status is not None and a.current_stage_attribution.value != "NOT_APPLICABLE"}
        return PipelineAnalysis(
            module=self.name, module_version=ANALYZER_VERSION, status=ModuleStatus.COMPLETED,
            suspected_stage=attributed_here.pop() if len(attributed_here) == 1 else None,
            evidence=[analysis.explanation_code.value, *[f.value for f in analysis.flags]],
            trace_analysis=analysis,
            message=f"{analysis.status.value}: {analysis.explanation_code.value} "
                    f"({len(analysis.defect_analyses)} defect type(s), {len(analysis.inspected_stages)} stage(s) inspected)")
