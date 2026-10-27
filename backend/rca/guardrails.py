"""Deterministic grounding checks on schema-valid LLM output. Any violation discards the whole output
(RCAStatus.INVALID_OUTPUT): a partially trusted RCA is worse than none, and the deterministic facts remain.

Checks
  DEFECT_SET_MISMATCH      exactly one finding per analysed defect type (no merging, no invented defects)
  STAGE_OUTSIDE_EVIDENCE   a cause's related_stage must be null or in that defect's candidate_stages
  STAGE_BLAMED_IN_TEXT     a hypothesis must not name a stage outside the candidate window (e.g. blaming PACKING
                           for a PRE_EXISTING defect)
  UNHEDGED_HYPOTHESIS      every hypothesis must be phrased as a possibility (likely / possible / may / could ...)
  CAUSAL_CERTAINTY         no "caused by", "root cause is", "proven", "definitely", ... in summary / causes / actions
  UNSUPPORTED_NUMBER       no percentages or probabilities anywhere
  FABRICATED_REFERENCE     no SOP ids or "according to policy" references (no SOP corpus exists yet)
  PERSON_BLAME             no blaming operators / staff
"""
from __future__ import annotations

import re

from backend.domain.fulfillment import FULFILLMENT_FLOW
from backend.schemas.rca import LLMRCAOutput, RCAEvidence


class GroundingViolation(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


HEDGE = re.compile(r"\b(likely|possibl[ey]|may|might|could|potential(ly)?|suspected|suggests?|consistent with)\b", re.I)
CERTAINTY = re.compile(r"\b(was|were|is|are|been) caused by\b|\bcaused (the|this)\b|\broot cause (is|was)\b|"
                       r"\b(definitely|certainly|undoubtedly|conclusively)\b|\bproven\b|\bconfirmed (that|as|to be)\b|"
                       r"\bwithout (a )?doubt\b", re.I)
NUMBER = re.compile(r"\d+(\.\d+)?\s?%|\bpercent\b|\bprobability\b|\bconfidence (of|score|level)\s*\d", re.I)
REFERENCE = re.compile(r"\bSOP[\s#:_-]*[A-Z]{0,4}[\s#:_-]*\d|\b(according to|per|as per) (the |our |company )?"
                       r"(sop|policy|policies|guideline|manual)\b|\b(company|our|store) policy\b", re.I)
PERSON = re.compile(r"\b(operator|picker|packer|staff|employee|worker|rider|associate)s?\b[^.]{0,40}\b"
                    r"(caused|is responsible|was responsible|is to blame|was careless|negligen)", re.I)
STAGE_WORDS = {s: re.compile(rf"\b{s.value}\b", re.I) for s in FULFILLMENT_FLOW}


def check_output(out: LLMRCAOutput, evidence: RCAEvidence) -> None:
    analyses = {a.defect_type: a for a in evidence.defect_analyses}
    found = [f.defect_type for f in out.findings]
    if sorted(found, key=lambda d: d.value) != sorted(analyses, key=lambda d: d.value):
        raise GroundingViolation("DEFECT_SET_MISMATCH", f"findings {[d.value for d in found]} vs analysed "
                                                        f"{sorted(d.value for d in analyses)}")
    texts: list[tuple[str, str]] = [("summary", out.summary)]
    for f in out.findings:
        window = set(analyses[f.defect_type].candidate_stages)
        for i, c in enumerate(f.likely_causes):
            where = f"{f.defect_type.value}.likely_causes[{i}]"
            if c.related_stage is not None and c.related_stage not in window:
                raise GroundingViolation("STAGE_OUTSIDE_EVIDENCE", f"{where}.related_stage={c.related_stage.value}")
            outside = [s.value for s, rx in STAGE_WORDS.items() if s not in window and rx.search(c.hypothesis)]
            if outside:
                raise GroundingViolation("STAGE_BLAMED_IN_TEXT", f"{where} names {outside}")
            if not HEDGE.search(c.hypothesis):
                raise GroundingViolation("UNHEDGED_HYPOTHESIS", where)
            texts += [(where, c.hypothesis), (where, c.rationale)]
        texts += [(f"{f.defect_type.value}.action", a.action) for a in f.corrective_actions]
        texts += [(f"{f.defect_type.value}.verification", v) for v in f.verification_steps]
        for where, text in [(f"{f.defect_type.value}.limitation", t) for t in f.limitations]:
            for code, rx in (("UNSUPPORTED_NUMBER", NUMBER), ("FABRICATED_REFERENCE", REFERENCE)):
                if rx.search(text):
                    raise GroundingViolation(code, where)
    for where, text in texts:
        for code, rx in (("CAUSAL_CERTAINTY", CERTAINTY), ("UNSUPPORTED_NUMBER", NUMBER),
                         ("FABRICATED_REFERENCE", REFERENCE), ("PERSON_BLAME", PERSON)):
            if rx.search(text):
                raise GroundingViolation(code, where)
