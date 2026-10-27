"""Citation guardrails for grounded actions. Any violation discards the whole grounding output.

  UNKNOWN_DEFECT          defect_type must be one of the analysed defects
  CITATION_NOT_RETRIEVED  chunk_id must be a chunk retrieved FOR THAT DEFECT (no invented / borrowed citations)
  QUOTE_NOT_IN_SOURCE     quote must appear verbatim (case / whitespace / bullet-insensitive) in the cited chunk
  QUOTE_ACTION_MISMATCH   action and quote must share at least one content word (the quote must support the action)
  BAD_RCA_ACTION_INDEX    rca_action_index must point at an existing Part 4 action for that defect
  INSTRUCTION_TEXT        instruction-like / prompt-injection text (from a document) can never become an action
  CAUSAL_CERTAINTY / UNSUPPORTED_NUMBER / FABRICATED_REFERENCE / PERSON_BLAME   as in the RCA guardrails
"""
from __future__ import annotations

import re

from backend.rag.documents import SUSPICIOUS
from backend.rag.embeddings import tokens
from backend.rca.guardrails import CERTAINTY, NUMBER, PERSON, REFERENCE, GroundingViolation
from backend.schemas.rag import LLMGroundingOutput

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", re.sub(r"(^|\n)\s*[-*]\s+", r"\1", text)).strip().lower().rstrip(".")


def check_grounding(out: LLMGroundingOutput, retrieved: dict, chunk_text: dict, rca_actions: dict) -> None:
    """retrieved: defect -> set(chunk_id); chunk_text: chunk_id -> text; rca_actions: defect -> list[str]."""
    for i, g in enumerate(out.grounded_actions):
        where = f"grounded_actions[{i}]"
        if g.defect_type not in retrieved:
            raise GroundingViolation("UNKNOWN_DEFECT", f"{where}.defect_type={g.defect_type.value}")
        if g.chunk_id not in retrieved[g.defect_type]:
            raise GroundingViolation("CITATION_NOT_RETRIEVED", f"{where}.chunk_id={g.chunk_id}")
        if _norm(g.quote) not in _norm(chunk_text[g.chunk_id]):
            raise GroundingViolation("QUOTE_NOT_IN_SOURCE", where)
        if not set(tokens(g.action)) & set(tokens(g.quote)):
            raise GroundingViolation("QUOTE_ACTION_MISMATCH", where)
        if g.rca_action_index is not None and g.rca_action_index >= len(rca_actions.get(g.defect_type, [])):
            raise GroundingViolation("BAD_RCA_ACTION_INDEX", where)
        if SUSPICIOUS.search(g.action) or SUSPICIOUS.search(g.quote):
            raise GroundingViolation("INSTRUCTION_TEXT", where)
        for code, rx in (("CAUSAL_CERTAINTY", CERTAINTY), ("UNSUPPORTED_NUMBER", NUMBER),
                         ("FABRICATED_REFERENCE", REFERENCE), ("PERSON_BLAME", PERSON)):
            if rx.search(g.action):
                raise GroundingViolation(code, where)
