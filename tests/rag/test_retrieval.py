"""Retrieval quality on the demo corpus: relevance, metadata boosts, threshold, per-document cap."""
import pytest

from backend.domain.enums import DefectType
from backend.rag.embeddings import HashingEmbeddingProvider
from backend.rag.retriever import CONTEXT_SECTIONS, MAX_PER_DOCUMENT, RetrievalQuery, retrieve
from backend.rag.store import VectorIndex

LEAK, CRUSH = DefectType.LEAKING_PACKAGE, DefectType.CRUSHED_PACKAGE
ACTIONS = "corrective actions immediate actions required checks verification escalation"


@pytest.fixture(scope="module")
def idx(demo_index):
    return VectorIndex.load(demo_index), HashingEmbeddingProvider()


def run(idx, defect, stages, text, top_k=5, min_similarity=0.15):
    index, emb = idx
    return retrieve(index, emb, RetrievalQuery(defect, tuple(stages), text), top_k=top_k, min_similarity=min_similarity)


def test_packing_leak_retrieves_packing_sealing_sop(idx):
    hits = run(idx, LEAK, ["PACKING"], "stage packing defect leaking package severity high new defect possible "
               "sealing-process issue at PACKING Inspect the sealing equipment and settings at PACKING " + ACTIONS)
    ids = [h.chunk.chunk_id for h in hits]
    assert hits[0].chunk.document_id in ("PACKING-SOP", "PACKAGING-MATERIAL-SOP", "LEAKAGE-CHECKLIST")
    assert "PACKING-SOP#seal-inspection" in ids[:3]
    assert {h.chunk.document_id for h in hits} <= {"PACKING-SOP", "PACKAGING-MATERIAL-SOP", "LEAKAGE-CHECKLIST"}


def test_receiving_crush_retrieves_receiving_sop(idx):
    hits = run(idx, CRUSH, ["RECEIVING"], "stage receiving defect crushed package pre existing product arriving "
               "damaged from the supplier quarantine affected stock inbound delivery " + ACTIONS)
    assert hits and hits[0].chunk.document_id == "RECEIVING-SOP"
    assert {h.chunk.document_id for h in hits[:3]} == {"RECEIVING-SOP", "DAMAGED-PRODUCT-SOP"}
    # other stages' SOPs that also cover crushed packages may follow, but always below the in-window documents
    assert all(h.chunk.stage in ("RECEIVING", "GENERAL") for h in hits[:3])


def test_irrelevant_documents_rank_lower(idx):
    hits = run(idx, LEAK, ["PACKING"], "leaking pouch seal heat sealer " + ACTIONS, top_k=20, min_similarity=0.0)
    rank = {h.chunk.document_id: i for i, h in reversed(list(enumerate(hits)))}  # best rank per document
    for unrelated in ("DELIVERY-SOP", "RECEIVING-SOP", "DISPATCH-SOP"):
        assert rank.get(unrelated, 99) > rank["PACKING-SOP"]


def test_unrelated_query_returns_nothing_above_threshold(idx):
    assert run(idx, LEAK, ["PACKING"], "quarterly marketing budget and social media campaign calendar") == []
    assert run(idx, LEAK, ["PACKING"], "leaking pouch seal", min_similarity=0.99) == []


def test_metadata_boosts_follow_the_stage_window(idx):
    text = "damaged crushed carton handling check " + ACTIONS
    receiving = run(idx, CRUSH, ["RECEIVING"], text)
    storage = run(idx, CRUSH, ["STORAGE"], text)
    assert receiving[0].chunk.stage in ("RECEIVING", "GENERAL") and storage[0].chunk.stage in ("STORAGE", "GENERAL")
    assert receiving[0].chunk.document_id != storage[0].chunk.document_id
    for h in receiving:  # score = similarity + boosts; boosts never exceed stage + general + defect
        assert 0 <= round(h.score - h.similarity, 4) <= 0.25


def test_per_document_cap_and_context_sections_excluded(idx):
    hits = run(idx, LEAK, ["PACKING"], "packing seal leak " + ACTIONS, top_k=20, min_similarity=0.0)
    per_doc = {}
    for h in hits:
        per_doc[h.chunk.document_id] = per_doc.get(h.chunk.document_id, 0) + 1
        assert h.chunk.section.lower() not in CONTEXT_SECTIONS
    assert max(per_doc.values()) <= MAX_PER_DOCUMENT
    assert hits == run(idx, LEAK, ["PACKING"], "packing seal leak " + ACTIONS, top_k=20, min_similarity=0.0)
