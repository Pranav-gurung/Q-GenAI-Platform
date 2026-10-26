import json
import shutil
import tempfile
import unittest
from pathlib import Path

from module_c.documents import build_chunks, chunk_page, DocMeta, Page
from module_c.retrieval import IngestionService, Retriever
from tests.helpers import (HANDLING_P1, HANDLING_P2, STORAGE_P1, make_settings, write_meta, write_pdf, write_scanned_pdf)

try:
    import reportlab  # noqa: F401
    HAVE_REPORTLAB = True
except ImportError:  # pragma: no cover
    HAVE_REPORTLAB = False


@unittest.skipUnless(HAVE_REPORTLAB, "reportlab is needed to generate PDF fixtures (pip install reportlab)")
class TestIngestionRetrieval(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.docs = self.tmp / "docs"
        self.docs.mkdir()
        self.s = make_settings(self.tmp)
        p = write_pdf(self.docs / "handling.pdf", [HANDLING_P1, HANDLING_P2])
        write_meta(p, "TEST-SOP-001", "Test Handling SOP", "2.0")
        q = write_pdf(self.docs / "storage.pdf", [STORAGE_P1])
        write_meta(q, "TEST-SOP-002", "Test Storage SOP", "1.1")

    def svc(self):
        return IngestionService(self.s)

    def test_extraction_page_metadata_and_chunks(self):
        results = self.svc().ingest_directory()
        self.assertEqual({r.status for r in results}, {"indexed"})
        store = self.svc().store
        chunks = list(store.chunks.values())
        h = [c for c in chunks if c.document_id == "TEST-SOP-001"]
        self.assertEqual({c.page for c in h}, {1, 2})
        for c in h:
            self.assertEqual((c.title, c.version, c.source_filename), ("Test Handling SOP", "2.0", "handling.pdf"))
            self.assertTrue(c.chunk_id.startswith(f"TEST-SOP-001:p{c.page}:"))
            self.assertEqual(len(c.content_hash), 64)
        # page-2 text only appears on page-2 chunks
        self.assertTrue(all("INSPECTION AFTER TRANSFER" not in c.text for c in h if c.page == 1))
        self.assertTrue(any("INSPECTION AFTER TRANSFER" in c.text for c in h if c.page == 2))

    def test_scanned_pdf_reported_not_indexed(self):
        write_scanned_pdf(self.docs / "scan.pdf")
        results = {r.filename: r for r in self.svc().ingest_directory()}
        self.assertEqual(results["scan.pdf"].status, "ocr_required")
        self.assertIn("OCR", results["scan.pdf"].detail)
        svc = self.svc()
        self.assertEqual(svc.registry.docs["scan"]["status"], "ocr_required")
        self.assertFalse(any(c.source_filename == "scan.pdf" for c in svc.store.chunks.values()))

    def test_empty_text_file_reported(self):
        (self.docs / "blank.txt").write_text("   \n")
        r = {x.filename: x for x in self.svc().ingest_directory()}["blank.txt"]
        self.assertIn(r.status, ("ocr_required", "empty"))
        self.assertEqual(r.chunks, 0)

    def test_readme_md_is_not_indexed_as_document(self):
        (self.docs / "README.md").write_text("# Documentation\nNot a real document.")
        results = {r.filename: r for r in self.svc().ingest_directory()}
        self.assertNotIn("README.md", results)

    def test_inferred_metadata_is_flagged(self):
        (self.docs / "notes.txt").write_text(BANNER_TEXT + " " + "word " * 40)
        r = {x.filename: x for x in self.svc().ingest_directory()}["notes.txt"]
        self.assertTrue(any("inferred" in w for w in r.warnings))

    def test_reingest_is_idempotent_and_no_duplicates(self):
        s1 = self.svc(); s1.ingest_directory()
        n = s1.store.count()
        again = self.svc().ingest_directory()
        self.assertEqual({r.status for r in again}, {"unchanged"})
        self.assertEqual(self.svc().store.count(), n)

    def test_changed_document_replaces_old_chunks(self):
        s = self.svc(); s.ingest_directory()
        old_ids = {c.chunk_id for c in s.store.chunks.values() if c.document_id == "TEST-SOP-001"}
        p = write_pdf(self.docs / "handling.pdf", [HANDLING_P1.replace("separator trays", "foam liners"), HANDLING_P2])
        write_meta(p, "TEST-SOP-001", "Test Handling SOP", "3.0")
        res = {r.filename: r for r in self.svc().ingest_directory()}
        self.assertEqual(res["handling.pdf"].status, "reindexed")
        s2 = self.svc()
        new = [c for c in s2.store.chunks.values() if c.document_id == "TEST-SOP-001"]
        self.assertTrue(all(c.version == "3.0" for c in new))
        self.assertTrue(any("foam liners" in c.text for c in new))
        self.assertFalse(any("separator trays" in c.text for c in new))
        self.assertTrue(old_ids - {c.chunk_id for c in new})  # stale ids are gone

    def test_index_persists_across_instances(self):
        self.svc().ingest_directory()
        hits, _ = Retriever(self.s).search("scratches on housings from handling abrasion")
        self.assertTrue(hits)
        self.assertEqual(hits[0].document_id, "TEST-SOP-001")

    def test_retrieval_results_carry_metadata_and_pick_right_document(self):
        self.svc().ingest_directory()
        r = Retriever(self.s)
        hits, _ = r.search("corrosion humid storage fasteners")
        self.assertEqual(hits[0].document_id, "TEST-SOP-002")
        self.assertEqual((hits[0].title, hits[0].version, hits[0].page), ("Test Storage SOP", "1.1", 1))
        self.assertGreaterEqual(hits[0].score, self.s.effective_min_score)
        hits, _ = r.search("scratch defect corrective action transfer inspection")
        self.assertEqual(hits[0].document_id, "TEST-SOP-001")

    def test_citation_traceability(self):
        s = self.svc(); s.ingest_directory()
        hits, _ = Retriever(self.s).search("scratches handling abrasion housings")
        for h in hits:
            stored = s.store.chunks[h.chunk_id]
            self.assertEqual(stored.text, h.text)
            self.assertEqual((stored.document_id, stored.page, stored.version), (h.document_id, h.page, h.version))

    def test_irrelevant_query_returns_nothing(self):
        self.svc().ingest_directory()
        hits, n_cand = Retriever(self.s).search("quarterly marketing budget for cafeteria catering")
        self.assertEqual(hits, [])
        self.assertGreater(n_cand, 0)

    def test_deactivate_remove_and_missing_file(self):
        s = self.svc(); s.ingest_directory()
        s.deactivate("TEST-SOP-002")
        hits, _ = Retriever(self.s, s.embedder, s.store).search("corrosion humid storage fasteners")
        self.assertFalse(any(h.document_id == "TEST-SOP-002" for h in hits))
        s.reactivate("TEST-SOP-002")
        self.assertTrue(Retriever(self.s, s.embedder, s.store).search("corrosion humid storage fasteners")[0])
        (self.docs / "storage.pdf").unlink()
        res = self.svc().ingest_directory()
        self.assertIn("deactivated_missing", {r.status for r in res})
        hits, _ = Retriever(self.s).search("corrosion humid storage fasteners")
        self.assertFalse(any(h.document_id == "TEST-SOP-002" for h in hits))
        s2 = self.svc()
        self.assertGreater(s2.remove("TEST-SOP-001"), 0)
        self.assertFalse(Retriever(self.s).search("scratches handling")[0])

    def test_access_level_filter(self):
        p = write_pdf(self.docs / "secret.pdf", ["SYNTHETIC. Scratch handling secret restricted procedure for housings " * 8])
        write_meta(p, "TEST-RESTRICTED", "Restricted", "1", access_level="restricted")
        self.svc().ingest_directory()
        hits, _ = Retriever(self.s).search("scratch handling secret restricted procedure housings")
        self.assertFalse(any(h.document_id == "TEST-RESTRICTED" for h in hits))
        s = make_settings(self.tmp, allowed_access_levels=("public", "internal", "restricted"))
        hits, _ = Retriever(s).search("scratch handling secret restricted procedure housings")
        self.assertTrue(any(h.document_id == "TEST-RESTRICTED" for h in hits))

    def test_embedder_swap_requires_reindex(self):
        self.svc().ingest_directory()

        class Other:
            name = "other-embedder"
            def embed(self, texts):
                import numpy as np
                return np.ones((len(texts), 4), np.float32)
        with self.assertRaises(RuntimeError):
            IngestionService(self.s, embedder=Other())

    def test_missing_directory_reported(self):
        r = self.svc().ingest_directory(self.tmp / "nope")
        self.assertEqual(r[0].status, "error")


BANNER_TEXT = "SYNTHETIC TEST NOTE."


class TestChunking(unittest.TestCase):
    def test_overlap_size_and_stable_ids(self):
        text = " ".join(f"Sentence number {i} about housings." for i in range(80))
        chunks = chunk_page(text, 300, 60)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        for a, b in zip(chunks, chunks[1:]):  # consecutive chunks share text
            self.assertTrue(set(a.split()[-4:]) & set(b.split()[:12]))
        meta = DocMeta("D1", "T", "1", "d.txt", "h" * 64)
        ids1 = [c.chunk_id for c in build_chunks(meta, [Page(1, text)], 300, 60)]
        ids2 = [c.chunk_id for c in build_chunks(meta, [Page(1, text)], 300, 60)]
        self.assertEqual(ids1, ids2)
        self.assertEqual(len(ids1), len(set(ids1)))

    def test_bad_overlap(self):
        with self.assertRaises(ValueError):
            chunk_page("x" * 1000, 100, 100)


if __name__ == "__main__":
    unittest.main()
