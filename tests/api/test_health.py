def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok" and body["storage"] == "ok"
    assert body["pipeline_modules"]["cv_engine"] == {"name": "stub", "implemented": False}
    assert "X-Request-ID" in r.headers


def test_openapi_schema_loads(client):
    r = client.get("/api/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    for path in ["/api/health", "/api/inspections", "/api/inspections/{inspection_id}",
                 "/api/inspections/{inspection_id}/image", "/api/inspections/{inspection_id}/feedback"]:
        assert path in spec["paths"]
    for schema in ["InspectionEvent", "InspectionDetail", "CVResult", "QualityDecision", "RCAResult",
                   "ActionRecommendation", "PipelineAnalysis", "FeedbackRecord", "ErrorResponse"]:
        assert schema in spec["components"]["schemas"]
    assert client.get("/api/docs").status_code == 200


def test_reference_data(client):
    body = client.get("/api/reference-data").json()
    assert any(s["store_id"] == "DS-BLR-001" for s in body["stores"])
    assert [s["stage"] for s in body["stages"]][:2] == ["RECEIVING", "STORAGE"]
    assert "REQUIRES_REVIEW" in body["pipeline_states"]


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Q-GenAI" in r.text
