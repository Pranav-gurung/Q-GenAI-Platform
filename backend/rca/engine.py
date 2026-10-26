"""Pipeline modules for Part 4 (selected in backend/pipeline/registry.py).

RCA_ENGINE=genai      RCA_PROCESSING: explains the PIPELINE_ANALYSIS result (Part 3) with the configured LLM.
ACTION_ENGINE=rca     ACTION_GENERATED: turns the RCA's suggested corrective actions into ActionRecommendation.
                      No second LLM call; nothing is produced when the RCA did not complete.
"""
from __future__ import annotations

from backend.core.config import Settings
from backend.domain.enums import ActionPriority, ModuleStatus
from backend.pipeline.context import PipelineContext
from backend.pipeline.interfaces import ActionEngine, RCAEngine
from backend.rca.service import RCAService
from backend.schemas.rca import RCAStatus
from backend.schemas.results import ActionRecommendation, CorrectiveAction, RCAResult, RootCause


class GenAIRCAEngine(RCAEngine):
    name = "genai"

    def __init__(self, settings: Settings, service: RCAService | None = None) -> None:
        self.service = service or RCAService(settings)

    def health(self) -> dict:
        h = self.service.health()
        return {"available": h["status"] in ("configured", "disabled"), "llm": h}

    def analyze(self, context: PipelineContext) -> RCAResult:
        analysis = context.pipeline_analysis
        trace = analysis.trace_analysis if analysis is not None and analysis.is_real else None
        if trace is None:
            return RCAResult.skipped(self.name, "No fulfillment pipeline analysis to explain")
        rca = self.service.analyze(trace)
        if rca.status is RCAStatus.NOT_APPLICABLE:
            return RCAResult.skipped(self.name, "No defect in the item's inspection history")
        if rca.status is not RCAStatus.COMPLETED:
            g = rca.generation
            return RCAResult.unavailable(self.name, f"{g.error_code}: {g.error_message}"[:500])
        g = rca.generation
        return RCAResult(
            module=self.name, module_version=f"{g.provider}:{g.model}:{g.prompt_version}", status=ModuleStatus.COMPLETED,
            possible_causes=[RootCause(cause=c.hypothesis, evidence=d.observed_evidence[1:3], stage=c.related_stage)
                             for d in rca.defects for c in d.likely_causes],
            recommended_investigation=[v for d in rca.defects for v in d.verification_steps],
            root_cause_analysis=rca, processing_time_ms=g.latency_ms,
            message=f"GenAI RCA ({g.provider}/{g.model}) for {len(rca.defects)} defect type(s)"
                    + ("; requires review" if rca.requires_review else ""))


class RCAActionEngine(ActionEngine):
    name = "rca"

    def health(self) -> dict:
        return {"available": True}

    def recommend(self, rca_result: RCAResult, context: PipelineContext) -> ActionRecommendation:
        rca = rca_result.root_cause_analysis if rca_result is not None and rca_result.is_real else None
        if rca is None:
            return ActionRecommendation.skipped(self.name, "No completed root-cause analysis to take actions from")
        actions = [CorrectiveAction(action=a.action, priority=ActionPriority(a.priority.value),
                                    responsible_stage=a.responsible_stage,
                                    instructions=[f"Defect: {d.defect_type.value}", f"Addresses: {a.addresses_category.value}",
                                                  "AI-suggested action - verify before acting"])
                   for d in rca.defects for a in d.corrective_actions]
        if not actions:
            return ActionRecommendation.skipped(self.name, "The RCA suggested no corrective actions")
        return ActionRecommendation(module=self.name, module_version=rca_result.module_version,
                                    status=ModuleStatus.COMPLETED, actions=actions,
                                    message=f"{len(actions)} AI-suggested corrective action(s)")
