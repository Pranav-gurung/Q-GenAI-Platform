"""Deterministic mock provider (LLM_PROVIDER=mock): a TEST DOUBLE, not AI.

It reads the same evidence the real model receives and returns template text per defect type, respecting the
grounding rules, so tests and offline demos exercise the full RCA path without a network. Outputs are labelled
provider="mock" everywhere. Not allowed in production (see Settings). `mode` simulates provider failures.
"""
from __future__ import annotations

import json
import re

from backend.rca.prompt import RCAPrompt, evidence_from_user_message
from backend.rag.prompt import PROMPT_VERSION as RAG_PROMPT_VERSION, blocks_from_user_message
from backend.rca.providers.base import LLMCompletion, LLMError

RAG_MODES = ("ok", "fake_citation", "fake_quote", "borrowed_citation", "rag_malformed", "no_support")
MODES = ("ok", "malformed_json", "schema_violation", "oversized", "unhedged", "blame_outside_window",
         "wrong_defect", "fake_sop", "timeout", "unavailable", "rate_limited", "api_error", "refused")

# defect type -> [(category, hypothesis noun phrase, action, verification)]
TEMPLATES = {
    "LEAKING_PACKAGE": [
        ("SEALING_PROCESS", "a sealing-process issue", "Inspect the sealing equipment and settings at {stage}",
         "Re-inspect a sample of units sealed at {stage} in the same period for leaks"),
        ("PACKAGING_MATERIAL", "damaged or defective packaging material", "Verify the condition of the packaging "
         "material lot in use at {stage}", "Check other units from the same packaging material lot"),
        ("HANDLING", "excessive handling pressure", "Review the handling procedure at {stage} for compression of "
         "liquid products", "Observe handling of liquid SKUs at {stage} during the next shift"),
    ],
    # appended to every defect's actions
    "_COMMON": [("PRODUCT_CONDITION", None, "Re-inspect affected inventory of this SKU", None)],
    "CRUSHED_PACKAGE": [
        ("HANDLING", "excessive handling force", "Review handling of fragile SKUs at {stage}",
         "Re-inspect recently handled units of this SKU at {stage}"),
        ("PACKAGING_PROCESS", "insufficient protective packing", "Verify packing configuration for this SKU at {stage}",
         "Check whether heavy items were packed on top of this SKU"),
        ("STORAGE_CONDITION", "stacking load", "Check stacking height and load limits for this SKU",
         "Inspect the storage location for overloaded stacks"),
    ],
    "TORN_PACKAGE": [
        ("HANDLING", "abrasion or snagging during handling", "Inspect bins, totes and surfaces at {stage} for sharp edges",
         "Re-inspect units handled through the same equipment"),
        ("PACKAGING_MATERIAL", "weak packaging material", "Verify the packaging material condition of this SKU",
         "Check other units from the same batch for tears"),
    ],
    "DENTED_CONTAINER": [
        ("HANDLING", "an impact or drop during handling", "Review drop and impact risks at {stage}",
         "Re-inspect other rigid containers handled at {stage}"),
        ("TRANSPORT_HANDLING", "an impact in transit", "Check securing of containers during movement between stages",
         "Inspect transit totes for loose rigid items"),
    ],
    "DAMAGED_BOX": [
        ("HANDLING", "rough handling of the carton", "Review carton handling at {stage}",
         "Re-inspect cartons of this SKU at {stage}"),
        ("STORAGE_CONDITION", "stacking or moisture exposure in storage", "Check stacking and humidity at the storage "
         "location", "Inspect neighbouring cartons in the same location"),
    ],
    "MISSING_LABEL": [
        ("LABELING", "label adhesion or application failure", "Verify label application and adhesive for this SKU",
         "Check other units of this SKU for loose or missing labels"),
    ],
    "BROKEN_SEAL": [("SEALING_PROCESS", "a seal failure", "Inspect seal integrity checks for this SKU",
                     "Re-inspect seals on units from the same batch")],
    "OPENED_PACKAGE": [("HANDLING", "the package being opened during handling", "Review handling and tamper checks at "
                        "{stage}", "Re-inspect units of this SKU for tamper evidence")],
    "INCORRECT_ITEM": [("PICKING_HANDLING", "a picking mix-up", "Verify bin labelling and scan compliance at picking",
                        "Re-check the picked item against the order line")],
    "VISUAL_ANOMALY": [("UNKNOWN", "an unclassified visual anomaly", "Have an operator inspect the item",
                        "Compare with reference images of the SKU")],
}
PRE_EXISTING = [("PRODUCT_CONDITION", "the product arriving damaged from the supplier or inbound transport",
                 "Quarantine affected stock from the same inbound delivery and raise it with the supplier",
                 "Inspect other units from the same inbound delivery")]


def build_output(ev: dict) -> dict:
    findings = []
    for a in ev["defect_analyses"]:
        cls, first = a["classification"], a["first_observed_stage"]
        weak = cls in ("INSUFFICIENT_EVIDENCE", "UNCERTAIN") or a["requires_review"]
        rows = PRE_EXISTING if cls == "PRE_EXISTING" else TEMPLATES.get(a["defect_type"], TEMPLATES["VISUAL_ANOMALY"])
        stage = first if cls == "NEW_DEFECT" else None
        where = stage or "the relevant stages"
        causes = [{"category": c, "hypothesis": "Possible " + re.sub(r"^(a|an|the) ", "", h) + (f" at {stage}" if stage else ""),
                   "rationale": f"{a['defect_type']} was first observed at {first} ({cls}, {a['evidence_level']} evidence).",
                   "related_stage": stage if cls != "PRE_EXISTING" else None} for c, h, _, _ in rows]
        verification = ([f"Review the missing or uncertain inspection evidence ({', '.join(a['missing_stages']) or 'previous stage'}) "
                         "before acting on any hypothesis"] if weak else [])
        verification += [v.format(stage=where) for _, _, _, v in rows]
        findings.append({
            "defect_type": a["defect_type"],
            "likely_causes": causes[:2] if weak else causes,
            "corrective_actions": [{"action": act.format(stage=where), "addresses_category": c,
                                    "responsible_stage": stage, "priority": "HIGH" if a["evidence_level"] == "HIGH" else "MEDIUM"}
                                   for c, _, act, _ in rows][:5] + [
                {"action": act, "addresses_category": c, "responsible_stage": None, "priority": "MEDIUM"}
                for c, _, act, _ in TEMPLATES["_COMMON"]],
            "verification_steps": verification[:6],
            "limitations": (["Evidence is insufficient to attribute the defect to a specific stage."] if weak else []),
        })
    first = ev["defect_analyses"][0]
    summary = (f"{first['defect_type']} was first observed at {first['first_observed_stage']} "
               f"({first['classification']}, {first['evidence_level']} evidence). The listed causes are hypotheses to "
               "verify; the evidence does not establish physical causality.")
    return {"summary": summary, "findings": findings}


_BULLET = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(.+?)\s*$")


def build_grounding(context: list[dict], sources: list[dict], max_per_defect: int = 6) -> dict:
    """rag-v1 double: copy imperative bullet lines from each defect's retrieved chunks as action + verbatim quote,
    and link them to the Part 4 action sharing the most content words (>= 3). Skips text the guardrails reject."""
    from backend.rag.documents import SUSPICIOUS
    from backend.rag.embeddings import tokens
    from backend.rca.guardrails import CERTAINTY, NUMBER, PERSON, REFERENCE
    text = {s["chunk_id"]: s["text"] for s in sources}
    out = []
    for d in context:
        rca = [(a["index"], set(tokens(a["action"]))) for a in d.get("suggested_actions", [])]
        picked = []
        for cid in d["source_chunk_ids"]:
            for line in text.get(cid, "").splitlines():
                m = _BULLET.match(line)
                if not m or len(picked) >= max_per_defect:
                    continue
                quote = m.group(1).strip()[:300]
                if len(quote) < 12 or any(rx.search(quote) for rx in (CERTAINTY, NUMBER, PERSON, REFERENCE, SUSPICIOUS)):
                    continue
                words = set(tokens(quote))
                best = max(rca, key=lambda r: (len(r[1] & words), -r[0]), default=None)
                idx = best[0] if best and len(best[1] & words) >= 3 else None
                picked.append({"defect_type": d["defect_type"], "action": quote.rstrip("."), "chunk_id": cid,
                               "quote": quote, "rca_action_index": idx,
                               "priority": "HIGH" if d.get("evidence_level") == "HIGH" else "MEDIUM"})
            if len(picked) >= max_per_defect:
                break
        out.extend(picked)
    return {"grounded_actions": out}


class MockProvider:
    name = "mock"

    def __init__(self, mode: str = "ok", model: str = "mock-rca-1", rag_mode: str = "ok") -> None:
        if mode not in MODES or rag_mode not in RAG_MODES:
            raise ValueError(f"unknown mock mode {mode!r}/{rag_mode!r}")
        self.mode, self.model, self.rag_mode = mode, model, rag_mode
        self.requests: list[RCAPrompt] = []  # what the "LLM" received (grounding tests)

    def status(self) -> str:
        return "configured"

    def generate_rca(self, prompt: RCAPrompt) -> LLMCompletion:
        self.requests.append(prompt)
        failures = {"timeout": "LLM_TIMEOUT", "unavailable": "LLM_UNAVAILABLE", "rate_limited": "LLM_RATE_LIMITED",
                    "api_error": "LLM_API_ERROR", "refused": "LLM_REFUSED"}
        if self.mode in failures:
            raise LLMError(failures[self.mode], f"simulated {self.mode}")
        if self.mode == "malformed_json":
            return LLMCompletion(text='{"summary": "truncated', model=self.model)
        if prompt.prompt_version == RAG_PROMPT_VERSION:
            return self._ground(prompt)
        out = build_output(evidence_from_user_message(prompt.user))
        f = out["findings"][0]
        if self.mode == "schema_violation":
            f["likely_causes"][0]["category"] = "OPERATOR_ERROR"
        elif self.mode == "oversized":
            out["summary"] = "x" * 5000
        elif self.mode == "unhedged":
            f["likely_causes"][0]["hypothesis"] = "The sealing machine caused the leak"
        elif self.mode == "blame_outside_window":
            f["likely_causes"][0]["related_stage"] = "DELIVERY"
        elif self.mode == "wrong_defect":
            f["defect_type"] = "VISUAL_ANOMALY" if f["defect_type"] != "VISUAL_ANOMALY" else "LEAKING_PACKAGE"
        elif self.mode == "fake_sop":
            f["corrective_actions"][0]["action"] = "Follow SOP-PK-17 section 4 for resealing"
        return LLMCompletion(text=json.dumps(out), model=self.model, input_tokens=len(prompt.user) // 4,
                             output_tokens=len(json.dumps(out)) // 4, stop_reason="end_turn")

    def _ground(self, prompt: RCAPrompt) -> LLMCompletion:
        if self.rag_mode == "rag_malformed":
            return LLMCompletion(text='{"grounded_actions": [{"action": ', model=self.model)
        context, sources = blocks_from_user_message(prompt.user)
        out = {"grounded_actions": []} if self.rag_mode == "no_support" else build_grounding(context, sources)
        acts = out["grounded_actions"]
        if acts and self.rag_mode == "fake_citation":
            acts[0]["chunk_id"] = "SOP-PK-17#resealing"
        elif acts and self.rag_mode == "fake_quote":
            acts[0]["quote"] = "Always reseal leaking pouches with tape and dispatch them."
        elif acts and self.rag_mode == "borrowed_citation":
            other = sorted({s["chunk_id"] for s in sources} - set(context[0]["source_chunk_ids"]))
            acts[0]["chunk_id"] = other[0] if other else "NOT-RETRIEVED#x"
        text = json.dumps(out)
        return LLMCompletion(text=text, model=self.model, input_tokens=len(prompt.user) // 4,
                             output_tokens=len(text) // 4, stop_reason="end_turn")
