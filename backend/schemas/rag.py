"""RAG / SOP grounding result (attached to RootCauseAnalysis.rag).

Three kinds of content stay separate end to end:
  system evidence        RootCauseAnalysis.defects[].facts / observed_evidence   (deterministic)
  GenAI hypotheses       RootCauseAnalysis.defects[].likely_causes / corrective_actions   (Part 4, AI)
  SOP-grounded actions   RAGResult.grounded_actions   (AI wording, each tied to a retrieved SOP chunk and a verbatim
                         quote; source document / section / version are filled from the index, never by the LLM)
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.domain.enums import DefectType
from backend.schemas.rca import RCAGeneration, SuggestedPriority


class RAGStatus(str, Enum):
    GROUNDED = "GROUNDED"                      # at least one SOP-backed action
    NOT_SUPPORTED = "NOT_SUPPORTED"            # relevant sources found, but they support no action
    NO_RELEVANT_SOURCE = "NO_RELEVANT_SOURCE"  # nothing above RAG_MIN_SIMILARITY (no LLM call)
    SOURCES_ONLY = "SOURCES_ONLY"              # sources retrieved; LLM disabled / failed / output rejected
    UNAVAILABLE = "UNAVAILABLE"                # RAG disabled, knowledge base / index / embeddings unavailable
    NOT_APPLICABLE = "NOT_APPLICABLE"          # no defect to ground


NO_SOURCE_MESSAGE = "No relevant SOP guidance was found in the available knowledge base. Manual review is recommended."
NOT_SUPPORTED_MESSAGE = ("No supporting SOP guidance was found in the available knowledge base for these actions. "
                         "Manual review is recommended.")


class RetrievedSource(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    section: str
    version: str
    stage: str
    document_type: str
    provenance: str
    text: str
    similarity: float = Field(description="Cosine similarity of the query and the chunk (retrieval score, not a "
                                          "probability).")
    rank_score: float
    defect_types: list[DefectType] = Field(description="Defects this source was retrieved for.")


class GroundedAction(BaseModel):
    defect_type: DefectType
    action: str
    priority: SuggestedPriority
    source_chunk_id: str
    source_document_id: str
    source_document: str
    source_section: str
    source_version: str
    source_provenance: str
    supporting_quote: str = Field(description="Verbatim text from the cited SOP section.")
    supports_rca_action: str | None = Field(None, description="The Part 4 AI action this SOP text supports, if any.")


class UnsupportedAction(BaseModel):
    defect_type: DefectType
    action: str
    reason: str = "No retrieved SOP section supports this AI-suggested action; it is not SOP-backed."


class DefectGrounding(BaseModel):
    defect_type: DefectType
    status: RAGStatus
    query: str
    source_chunk_ids: list[str]
    message: str | None = None


class RAGResult(BaseModel):
    status: RAGStatus
    requires_review: bool
    message: str | None = None
    defects: list[DefectGrounding] = Field(default_factory=list)
    retrieved_sources: list[RetrievedSource] = Field(default_factory=list)
    grounded_actions: list[GroundedAction] = Field(default_factory=list)
    unsupported_actions: list[UnsupportedAction] = Field(default_factory=list)
    knowledge_base: dict = Field(default_factory=dict, description="Index metadata: documents, chunks, embedding, "
                                                                   "provenance.")
    retrieval_ms: float | None = None
    generation: RCAGeneration | None = None
    disclaimer: str = ("SOP-grounded actions cite the demo knowledge base (DEMO / SYNTHETIC documents, not official "
                       "procedures of any company). Grounding shows which document supports an action; it does not "
                       "prove what caused the defect.")


# ------------------------------------------------------------------------------------------------ LLM output contract
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=300)]


class LLMGroundedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defect_type: DefectType
    action: Text
    chunk_id: str
    quote: Annotated[str, StringConstraints(strip_whitespace=True, min_length=12, max_length=300)]
    rca_action_index: int | None = Field(None, ge=0)
    priority: SuggestedPriority


class LLMGroundingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grounded_actions: list[LLMGroundedAction] = Field(max_length=20)
