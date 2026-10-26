"""RCAService: evidence -> provider -> validation -> RootCauseAnalysis. Never raises for LLM problems.

  no defect in the trace          -> NOT_APPLICABLE (no LLM call)
  LLM disabled / provider error   -> UNAVAILABLE    (deterministic facts kept, AI fields empty, requires_review)
  malformed / ungrounded output   -> INVALID_OUTPUT (output discarded, same as above)
  validated output                -> COMPLETED      (facts + clearly separated AI hypotheses)
Logs: inspection id, status, provider, model, latency, token counts, error code. Never prompts, outputs or keys.
"""
from __future__ import annotations

import json
import logging
import time

from pydantic import ValidationError

from backend.core.config import Settings
from backend.core.logging import log_pipeline
from backend.rca import evidence as ev
from backend.rca.guardrails import GroundingViolation, check_output
from backend.rca.prompt import PROMPT_VERSION, build_prompt
from backend.rca.providers import LLMError, LLMProvider, build_provider
from backend.schemas.fulfillment import AttributionStatus, FulfillmentTraceAnalysis
from backend.schemas.rca import (DefectRCA, LLMRCAOutput, OperatorNote, RCAGeneration, RCAStatus,
                                 RootCauseAnalysis)

logger = logging.getLogger(__name__)
REVIEW_ATTRIBUTIONS = (AttributionStatus.REQUIRES_REVIEW, AttributionStatus.INSUFFICIENT_EVIDENCE)


class RCAService:
    def __init__(self, settings: Settings, provider: LLMProvider | None = None) -> None:
        self.settings = settings
        self.provider = provider if provider is not None else (build_provider(settings) if settings.llm_enabled else None)
        if settings.llm_temperature is not None and settings.llm_provider == "anthropic":
            logger.warning("LLM_TEMPERATURE is set but not sent: the Anthropic API rejects sampling parameters")

    def health(self) -> dict:
        s = self.settings
        if not s.llm_enabled:
            status = "disabled"
        else:
            status = self.provider.status() if self.provider else "not_configured"
        return {"enabled": s.llm_enabled, "provider": s.llm_provider, "model": s.llm_model if s.llm_provider != "mock"
                else "mock-rca-1", "status": status, "effort": s.llm_effort, "timeout_seconds": s.llm_timeout_seconds,
                "prompt_version": PROMPT_VERSION}

    # ------------------------------------------------------------------------------------------------
    def analyze(self, trace: FulfillmentTraceAnalysis, operator_notes: list[OperatorNote] | None = None) -> RootCauseAnalysis:
        evidence = ev.build_evidence(trace, operator_notes)
        defects = [DefectRCA(defect_type=a.defect_type, facts=ev.defect_facts(trace, a),
                             observed_evidence=[ev.timeline_sentence(trace), *ev.observed_evidence(trace, a)],
                             limitations=ev.limitations(trace, a))
                   for a in trace.defect_analyses]
        base = dict(inspection_id=trace.current_inspection_id, current_stage=trace.current_stage,
                    defects=defects, limitations=ev.trace_limitations(trace))
        needs_review = any(a.attribution_status in REVIEW_ATTRIBUTIONS for a in trace.defect_analyses)

        if not defects:
            return self._log(RootCauseAnalysis(status=RCAStatus.NOT_APPLICABLE, requires_review=False,
                                               generation=RCAGeneration(ai_generated=False, prompt_version=PROMPT_VERSION),
                                               **base))
        if not self.settings.llm_enabled or self.provider is None:
            return self._log(self._unavailable(base, "LLM_DISABLED", "GenAI RCA is disabled (LLM_ENABLED=false)"))

        prompt = build_prompt(evidence)
        log_pipeline(trace.current_inspection_id, "RCA_PROCESSING", "LLM_REQUEST",
                     message=f"provider={self.provider.name} model={self.provider.model} defects={len(defects)}")
        t0 = time.perf_counter()
        try:
            completion = self.provider.generate_rca(prompt)
        except LLMError as exc:
            return self._log(self._unavailable(base, exc.code, exc.message, latency=_ms(t0)))
        except Exception:  # a provider bug must never fail the inspection; details go to the log only
            logger.exception("LLM provider %s raised unexpectedly", self.provider.name)
            return self._log(self._unavailable(base, "LLM_PROVIDER_ERROR", "Unexpected provider error", latency=_ms(t0)))
        latency = _ms(t0)
        gen = dict(provider=self.provider.name, model=completion.model, prompt_version=PROMPT_VERSION,
                   latency_ms=latency, input_tokens=completion.input_tokens, output_tokens=completion.output_tokens)
        try:
            out = LLMRCAOutput.model_validate(json.loads(completion.text))
            check_output(out, evidence)
        except (json.JSONDecodeError, ValidationError) as exc:
            where = (f"{len(exc.errors())} schema error(s) at {[e['loc'] for e in exc.errors()][:3]}"
                     if isinstance(exc, ValidationError) else "malformed JSON")
            return self._log(self._invalid(base, gen, "LLM_INVALID_OUTPUT", where))
        except GroundingViolation as exc:
            return self._log(self._invalid(base, gen, f"LLM_UNGROUNDED_OUTPUT:{exc.code}", exc.detail))

        by_type = {f.defect_type: f for f in out.findings}
        for d in defects:
            f = by_type[d.defect_type]
            d.likely_causes, d.corrective_actions = f.likely_causes, f.corrective_actions
            d.verification_steps, d.ai_limitations = f.verification_steps, f.limitations
        return self._log(RootCauseAnalysis(status=RCAStatus.COMPLETED, requires_review=needs_review,
                                           summary=out.summary, generation=RCAGeneration(ai_generated=True, **gen),
                                           **base))

    def _unavailable(self, base, code, message, latency=None) -> RootCauseAnalysis:
        return RootCauseAnalysis(status=RCAStatus.UNAVAILABLE, requires_review=True, **base, generation=RCAGeneration(
            ai_generated=False, provider=self.settings.llm_provider, model=self.provider.model if self.provider else None,
            prompt_version=PROMPT_VERSION, latency_ms=latency, error_code=code, error_message=message))

    def _invalid(self, base, gen, code, message) -> RootCauseAnalysis:
        return RootCauseAnalysis(status=RCAStatus.INVALID_OUTPUT, requires_review=True, **base,
                                 generation=RCAGeneration(ai_generated=False, error_code=code,
                                                          error_message=f"LLM output discarded: {message}"[:300], **gen))

    @staticmethod
    def _log(rca: RootCauseAnalysis) -> RootCauseAnalysis:
        g = rca.generation
        log_pipeline(rca.inspection_id, "RCA_PROCESSING", rca.status.value,
                     level=logging.WARNING if rca.status in (RCAStatus.UNAVAILABLE, RCAStatus.INVALID_OUTPUT) else logging.INFO,
                     duration_ms=g.latency_ms, error_code=g.error_code,
                     message=f"provider={g.provider} model={g.model} defects={len(rca.defects)} "
                             f"requires_review={rca.requires_review} tokens_in={g.input_tokens} tokens_out={g.output_tokens}")
        return rca


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
