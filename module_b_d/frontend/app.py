"""Q-GenAI Streamlit dashboard.

Run:  streamlit run frontend/app.py
(The FastAPI backend must be running; set QGENAI_API_URL if not on :8000.)
"""
from datetime import datetime, timezone

import streamlit as st

from api_client import ApiClient, ApiError
from views import history, inspection, review, security_dashboard, settings_admin, user_management
from views.common import inject_css, status_badge

st.set_page_config(page_title="Q-GenAI | Quality Control", page_icon="🛡️", layout="wide")
inject_css()


def client() -> ApiClient:
    return ApiClient(st.session_state.get("token"))


def logout(call_api: bool = True) -> None:
    if call_api and st.session_state.get("token"):
        try:
            client().logout()
        except ApiError:
            pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def login_screen() -> None:
    st.markdown("<div class='qg-hero'><h1>Q-GenAI</h1>"
                "<p>AI-powered industrial quality control · secure, traceable inspections</p></div>",
                unsafe_allow_html=True)
    col, _ = st.columns([1, 1.4])
    with col:
        with st.form("login"):
            st.subheader("Sign in")
            username = st.text_input("Username", max_chars=64)
            password = st.text_input("Password", type="password", max_chars=128)
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            if not username or not password:
                st.error("Enter your username and password.")
                return
            try:
                tok = ApiClient().login(username, password)
            except ApiError as e:
                st.error(e.detail)
                return
            st.session_state.token = tok["access_token"]
            st.rerun()
        try:
            h = ApiClient().health()
            st.caption(f"API online · model: {h['model_status']}")
        except ApiError as e:
            st.warning(e.detail)


# Pages and the permission that makes them visible (the backend enforces the
# same permissions again on every request).
PAGES = [
    ("🔍 Inspect product", "inspection:run", inspection.render),
    ("🗂️ Inspection history", None, history.render),
    ("✅ Review queue", "inspection:override", review.render),
    ("🛡️ Security & audit", ("audit:view_site", "audit:view_all"), security_dashboard.render),
    ("👥 User management", "user:manage", user_management.render),
    ("⚙️ Policy & models", "policy:view", settings_admin.render),
]


def allowed(perm, perms: list[str]) -> bool:
    if perm is None:
        return True
    if isinstance(perm, tuple):
        return any(p in perms for p in perm)
    return perm in perms


def main() -> None:
    if not st.session_state.get("token"):
        login_screen()
        return
    api = client()
    try:
        me = api.me()
    except ApiError as e:
        logout(call_api=False)
        st.warning(f"{e.detail} Please log in again." if e.status == 401 else e.detail)
        login_screen()
        return

    with st.sidebar:
        st.markdown("<div class='qg-brand'>Q-GenAI</div>", unsafe_allow_html=True)
        st.write(f"**{me['full_name'] or me['username']}**")
        st.markdown(status_badge(me["role"].replace("_", " ").upper(), "role"), unsafe_allow_html=True)
        st.caption(f"Site: {me['site']}")
        exp = datetime.fromisoformat(me["token_expires_at"].replace("Z", "+00:00"))
        mins = max(0, int((exp - datetime.now(timezone.utc)).total_seconds() // 60))
        st.caption(f"Session expires in ~{mins} min")
        try:
            mi = api.model_info()
            label = {"loaded": "Model loaded", "unavailable": "No model (manual review mode)",
                     "error": "Model error"}.get(mi["model_status"], mi["model_status"])
            st.markdown(status_badge(label, mi["model_status"]), unsafe_allow_html=True)
            st.caption(f"Version: {mi.get('model_version') or 'n/a'}")
            if mi.get("is_generic_pretrained"):
                st.warning("Generic pretrained weights - not a defect model.")
        except ApiError:
            pass
        st.divider()
        visible = [(name, fn) for name, perm, fn in PAGES if allowed(perm, me["permissions"])]
        choice = st.radio("Navigation", [n for n, _ in visible], label_visibility="collapsed")
        st.divider()
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()

    try:
        dict(visible)[choice](api, me)
    except ApiError as e:
        if e.status == 401:
            logout(call_api=False)
            st.warning("Your session expired. Please log in again.")
            st.rerun()
        st.error(e.detail)


main()
