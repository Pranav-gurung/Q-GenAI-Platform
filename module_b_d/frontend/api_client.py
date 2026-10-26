"""Thin HTTP client for the Q-GenAI API used by the Streamlit UI.

The UI never decides permissions itself - every call is authorised by the
backend; the UI only hides controls for convenience.
"""
import os

import requests

API_URL = os.environ.get("QGENAI_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.environ.get("QGENAI_API_TIMEOUT", "120"))


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _detail(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}"
    d = body.get("detail", body)
    if isinstance(d, list):  # FastAPI validation errors
        return "; ".join(f"{'.'.join(str(x) for x in e.get('loc', [])[1:])}: {e.get('msg')}" for e in d)
    return str(d)


class ApiClient:
    def __init__(self, token: str | None = None):
        self.token = token

    def _req(self, method: str, path: str, **kw):
        headers = kw.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=TIMEOUT, **kw)
        except requests.RequestException:
            raise ApiError(0, f"Cannot reach the Q-GenAI API at {API_URL}. Is the backend running?") from None
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, _detail(resp))
        return resp

    # --- auth
    def login(self, username: str, password: str) -> dict:
        return self._req("POST", "/auth/login", data={"username": username, "password": password}).json()

    def logout(self) -> None:
        self._req("POST", "/auth/logout")

    def me(self) -> dict:
        return self._req("GET", "/auth/me").json()

    # --- general
    def health(self) -> dict:
        return self._req("GET", "/health").json()

    def model_info(self) -> dict:
        return self._req("GET", "/model-info").json()

    # --- inspections
    def predict(self, filename: str, data: bytes, mime: str, threshold: float | None,
                override_reason: str | None) -> dict:
        form = {}
        if threshold is not None:
            form["confidence_threshold"] = str(threshold)
        if override_reason:
            form["override_reason"] = override_reason
        return self._req("POST", "/predict", files={"file": (filename, data, mime)}, data=form).json()

    def inspections(self, **params) -> dict:
        return self._req("GET", "/inspections", params={k: v for k, v in params.items() if v}).json()

    def inspection(self, inspection_id: str) -> dict:
        return self._req("GET", f"/inspections/{inspection_id}").json()

    def report_bytes(self, inspection_id: str) -> bytes:
        return self._req("GET", f"/inspections/{inspection_id}/report").content

    def annotated_image(self, inspection_id: str) -> bytes:
        return self._req("GET", f"/inspections/{inspection_id}/annotated-image").content

    def rca_payload(self, inspection_id: str) -> dict:
        return self._req("GET", f"/inspections/{inspection_id}/rca-payload").json()

    # --- quality
    def policy(self) -> dict:
        return self._req("GET", "/quality/policy").json()

    def policy_history(self) -> list:
        return self._req("GET", "/quality/policy/history").json()

    def update_policy(self, body: dict) -> dict:
        return self._req("PUT", "/quality/policy", json=body).json()

    def override(self, inspection_id: str, new_decision: str, reason: str) -> dict:
        return self._req("POST", f"/inspections/{inspection_id}/override",
                         json={"new_decision": new_decision, "reason": reason}).json()

    def review_queue(self) -> list:
        return self._req("GET", "/quality/review-queue").json()

    def scoped_audit_logs(self, **params) -> dict:
        return self._req("GET", "/quality/audit-logs", params={k: v for k, v in params.items() if v}).json()

    # --- admin
    def users(self) -> list:
        return self._req("GET", "/admin/users").json()

    def create_user(self, body: dict) -> dict:
        return self._req("POST", "/admin/users", json=body).json()

    def update_user(self, user_id: str, body: dict) -> dict:
        return self._req("PATCH", f"/admin/users/{user_id}", json=body).json()

    def audit_logs(self, **params) -> dict:
        return self._req("GET", "/admin/audit-logs", params={k: v for k, v in params.items() if v}).json()

    def verify_audit(self, checkpoint: dict | None = None) -> dict:
        return self._req("POST", "/admin/audit-logs/verify",
                         json={"checkpoint": checkpoint} if checkpoint else None).json()

    def create_checkpoint(self) -> dict:
        return self._req("POST", "/admin/audit-logs/checkpoint").json()

    def models(self) -> dict:
        return self._req("GET", "/admin/models").json()

    def register_model(self, body: dict) -> dict:
        return self._req("POST", "/admin/models", json=body).json()

    def patch_model(self, model_id: str, approved: bool) -> dict:
        return self._req("PATCH", f"/admin/models/{model_id}", json={"approved": approved}).json()

    def activate_model(self, model_id: str) -> dict:
        return self._req("POST", f"/admin/models/{model_id}/activate").json()
