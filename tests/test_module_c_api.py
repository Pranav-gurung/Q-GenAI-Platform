"""HTTP-level tests. They require fastapi + httpx and are SKIPPED when those are not installed."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    from module_c.api import create_app
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

from module_c.audit import InMemoryAuditSink
from module_c.rca import RCAService
from module_c.retrieval import IngestionService, Retriever
from tests.helpers import HANDLING_P1, HANDLING_P2, StubProvider, make_settings, write_meta, write_pdf
from tests.test_module_c_rca import GOOD_REQ, good, llm_reply


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed in this environment (pip install -r requirements.txt)")
class TestAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        docs = self.tmp / "docs"; docs.mkdir()
        write_meta(write_pdf(docs / "h.pdf", [HANDLING_P1, HANDLING_P2]), "TEST-SOP-001", "Test Handling SOP", "2.0")
        self.s = make_settings(self.tmp)
        IngestionService(self.s).ingest_directory()
        self.audit = InMemoryAuditSink()
        self.prov = StubProvider([llm_reply()])
        self.client = TestClient(create_app(self.s, RCAService(self.s, Retriever(self.s), self.prov, self.audit)))

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertGreater(r.json()["indexed_chunks"], 0)
        self.assertTrue(r.json()["llm_configured"])

    def test_rca_success(self):
        r = self.client.post("/rca", json=GOOD_REQ)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "success")
        self.assertTrue(r.json()["citations"])

    def test_rca_422_cases(self):
        for body in (good(defects=[{"class": "nope", "confidence": .5, "bbox": [1, 1, 2, 2]}]),
                     good(defects=[{"class": "scratch", "confidence": 2, "bbox": [1, 1, 2, 2]}]),
                     good(defects=[{"class": "scratch", "confidence": .5, "bbox": [5, 5, 2, 2]}]),
                     {"inspection_id": "x"}, good(unexpected_field=1)):
            self.assertEqual(self.client.post("/rca", json=body).status_code, 422, body)
        self.assertEqual(self.client.post("/rca", content=b"not json", headers={"content-type": "application/json"}).status_code, 422)
        self.assertEqual(self.prov.calls, 0)

    def test_rca_422_returns_request_id(self):
        r = self.client.post("/rca", json={"unexpected_top_field": 1})
        self.assertEqual(r.status_code, 422)
        body = r.json()
        self.assertIn("request_id", body)
        self.assertIn("detail", body)

    def test_empty_defects_200(self):
        r = self.client.post("/rca", json=good(defects=[]))
        self.assertEqual((r.status_code, r.json()["status"]), (200, "success"))

    def test_insufficient_evidence_is_200(self):
        r = self.client.post("/rca", json=good(defects=[{"class": "crack", "confidence": .9, "bbox": [1, 1, 9, 9]}]))
        self.assertEqual((r.status_code, r.json()["status"]), (200, "insufficient_evidence"))

    def test_llm_failure_maps_to_http_error_with_error_body(self):
        from module_c.llm import LLMRateLimitError, LLMTimeoutError
        svc = RCAService(self.s, Retriever(self.s), StubProvider([LLMTimeoutError("t")]), self.audit)
        c = TestClient(create_app(self.s, svc))
        r = c.post("/rca", json=GOOD_REQ)
        self.assertEqual((r.status_code, r.json()["status"], r.json()["error"]["code"]), (504, "error", "llm_timeout"))
        svc = RCAService(self.s, Retriever(self.s), StubProvider([LLMRateLimitError("r", 5)]), self.audit)
        r = TestClient(create_app(self.s, svc)).post("/rca", json=GOOD_REQ)
        self.assertEqual((r.status_code, r.headers.get("retry-after")), (429, "5"))

    def test_static_api_key_auth(self):
        import os
        from unittest import mock
        s = make_settings(self.tmp, auth_mode="static_api_key")
        svc = RCAService(s, Retriever(s), self.prov, self.audit)
        with mock.patch.dict(os.environ, {"RCA_API_KEY": "secret"}):
            c = TestClient(create_app(s, svc))
            self.assertEqual(c.post("/rca", json=GOOD_REQ).status_code, 401)
            self.assertEqual(c.post("/rca", json=GOOD_REQ, headers={"X-API-Key": "wrong"}).status_code, 401)
            self.assertEqual(c.post("/rca", json=GOOD_REQ, headers={"X-API-Key": "secret"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
