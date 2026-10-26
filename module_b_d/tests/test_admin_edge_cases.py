from fastapi.testclient import TestClient


def test_health_endpoint_is_public(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200


def test_admin_users_requires_authentication(client: TestClient):
    response = client.get("/admin/users")
    assert response.status_code in (401, 403)


def test_admin_models_requires_authentication(client: TestClient):
    response = client.get("/admin/models")
    assert response.status_code in (401, 403)


def test_admin_audit_logs_requires_authentication(client: TestClient):
    response = client.get("/admin/audit-logs")
    assert response.status_code in (401, 403)