
import hashlib

from fastapi.testclient import TestClient

from app.config import get_settings
from app.database import SessionLocal
from app.models.model_version import ModelVersion
from app.models.user import Role, User
from app.services.detector import ModelInfo, detector_manager
from app.services.authentication import hash_password


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


def test_update_nonexistent_user_returns_404(client, headers):
    response = client.patch(
        "/admin/users/nonexistent-user-id",
        headers=headers["admin"],
        json={"full_name": "Updated User"},
    )
    assert response.status_code == 404


def test_admin_cannot_demote_self(client, headers, users):
    response = client.patch(
        f"/admin/users/{users['admin'].id}",
        headers=headers["admin"],
        json={"role": "operator"},
    )
    assert response.status_code == 400


def test_admin_cannot_disable_self(client, headers, users):
    response = client.patch(
        f"/admin/users/{users['admin'].id}",
        headers=headers["admin"],
        json={"is_active": False},
    )
    assert response.status_code == 400



def test_register_model_rejects_path_traversal(client, headers):
    response = client.post(
        "/admin/models",
        headers=headers["admin"],
        json={
            "version_label": "edge-path-test",
            "weights_file": "../outside.pt",
            "notes": "Test invalid path",
            "approved": False,
        },
    )

    # Pydantic rejects the slash before the endpoint executes.
    assert response.status_code == 422



def test_register_model_rejects_missing_weights(client, headers):
    response = client.post(
        "/admin/models",
        headers=headers["admin"],
        json={
            "version_label": "edge-missing-test",
            "weights_file": "definitely_missing_test_weights.pt",
            "notes": "Test missing weights",
            "approved": False,
        },
    )
    assert response.status_code == 404


def test_patch_nonexistent_model_returns_404(client, headers):
    response = client.patch(
        "/admin/models/nonexistent-model-id",
        headers=headers["admin"],
        json={"approved": True},
    )
    assert response.status_code == 404


def test_activate_nonexistent_model_returns_404(client, headers):
    response = client.post(
        "/admin/models/nonexistent-model-id/activate",
        headers=headers["admin"],
    )
    assert response.status_code == 404


def test_register_model_rejects_unsupported_extension(
    client, headers
):
    response = client.post(
        "/admin/models",
        headers=headers["admin"],
        json={
            "version_label": "edge-extension-test",
            "weights_file": "model.txt",
            "notes": "Test unsupported extension",
            "approved": False,
        },
    )
    assert response.status_code == 400


def test_list_models_returns_expected_structure(client, headers):
    response = client.get("/admin/models", headers=headers["admin"])
    assert response.status_code == 200

    body = response.json()
    assert "registered" in body
    assert "weight_files" in body
    assert "active" in body
    assert isinstance(body["registered"], list)
    assert isinstance(body["weight_files"], list)


def test_admin_audit_logs_support_filtering(client, headers):
    response = client.get(
        "/admin/audit-logs",
        headers=headers["admin"],
        params={"limit": 10, "offset": 0},
    )
    assert response.status_code == 200

    body = response.json()
    assert "items" in body
    assert "total" in body
    assert isinstance(body["items"], list)


def test_admin_audit_logs_reject_invalid_limit(client, headers):
    response = client.get(
        "/admin/audit-logs",
        headers=headers["admin"],
        params={"limit": 0},
    )
    assert response.status_code == 422


def test_admin_audit_logs_reject_negative_offset(client, headers):
    response = client.get(
        "/admin/audit-logs",
        headers=headers["admin"],
        params={"offset": -1},
    )
    assert response.status_code == 422


def test_admin_cannot_set_extra_privilege_fields(client, headers):
    response = client.post(
        "/admin/users",
        headers=headers["admin"],
        json={
            "username": "edge_extra_field",
            "full_name": "Edge Test",
            "password": "StrongTestPass123!",
            "role": "operator",
            "site": "plant-a",
            "is_superuser": True,
        },
    )
    assert response.status_code == 422


def test_register_model_duplicate_version_label_returns_409(
    client, headers, tmp_path, monkeypatch
):
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    weights = settings.models_dir / "edge_duplicate.pt"
    weights.write_bytes(b"test model placeholder")

    try:
        first = client.post(
            "/admin/models",
            headers=headers["admin"],
            json={
                "version_label": "edge-duplicate-version",
                "weights_file": weights.name,
                "notes": "First registration",
                "approved": False,
            },
        )
        assert first.status_code == 201, first.text

        second = client.post(
            "/admin/models",
            headers=headers["admin"],
            json={
                "version_label": "edge-duplicate-version",
                "weights_file": weights.name,
                "notes": "Duplicate registration",
                "approved": False,
            },
        )
        assert second.status_code == 409, second.text
    finally:
        weights.unlink(missing_ok=True)


def test_activate_model_rejects_changed_checksum(client, headers):
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    weights = settings.models_dir / "edge_checksum.pt"
    weights.write_bytes(b"original model bytes")

    try:
        admin_id = users_admin_id(client, headers)

        with SessionLocal() as db:
            model = ModelVersion(
                version_label="edge-checksum-version",
                weights_file=weights.name,
                sha256=hashlib.sha256(b"different model bytes").hexdigest(),
                notes="Checksum mismatch test",
                approved=False,
                registered_by=admin_id,
                is_active=False,
            )
            db.add(model)
            db.commit()
            db.refresh(model)
            model_id = model.id

        response = client.post(
            f"/admin/models/{model_id}/activate",
            headers=headers["admin"],
        )
        assert response.status_code == 409, response.text
    finally:
        weights.unlink(missing_ok=True)


def users_admin_id(client, headers):
    response = client.get("/admin/users", headers=headers["admin"])
    assert response.status_code == 200, response.text
    admin = next(
        (user for user in response.json() if user["username"] == "admin1"),
        None,
    )
    assert admin is not None
    return admin["id"]
