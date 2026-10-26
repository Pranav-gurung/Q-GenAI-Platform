"""Security & audit dashboard: audit log viewer and hash-chain verification."""
import json

import pandas as pd
import streamlit as st

from api_client import ApiClient, ApiError
from views.common import status_badge


def render(api: ApiClient, me: dict) -> None:
    st.header("Security & audit")
    is_admin = "audit:verify" in me["permissions"]

    if is_admin:
        st.subheader("Audit-chain integrity")
        st.caption("Recomputes every SHA-256 link from the first event. Detects edited, deleted, "
                   "inserted or reordered events. It is tamper-EVIDENT, not tamper-proof: someone who "
                   "can rewrite the whole database and recompute all hashes can only be caught by "
                   "comparing against a signed checkpoint stored elsewhere.")
        cp_text = st.text_area("Optional: paste a signed checkpoint JSON", height=90)
        c1, c2 = st.columns(2)
        if c1.button("🔐 Verify audit chain", type="primary", use_container_width=True):
            try:
                cp = json.loads(cp_text) if cp_text.strip() else None
                if cp:
                    cp = {k: cp[k] for k in ("seq", "hash", "created_at", "signature")}
                st.session_state.verify_result = api.verify_audit(cp)
            except (ValueError, KeyError):
                st.error("Checkpoint must be the JSON produced by 'Create checkpoint'.")
            except ApiError as e:
                st.error(e.detail)
        if c2.button("📌 Create signed checkpoint", use_container_width=True):
            try:
                st.session_state.checkpoint = api.create_checkpoint()
            except ApiError as e:
                st.error(e.detail)

        r = st.session_state.get("verify_result")
        if r:
            if r["valid"]:
                st.markdown(status_badge("CHAIN VALID", "valid"), unsafe_allow_html=True)
                st.success(f"All {r['verified_events']} events verified. Head #{r['last_seq']}: "
                           f"`{(r['last_hash'] or '')[:16]}…`"
                           + (" · checkpoint matched" if r["checkpoint_checked"] else ""))
            else:
                st.markdown(status_badge("INTEGRITY FAILURE", "invalid"), unsafe_allow_html=True)
                st.error(f"First failure at event #{r['first_failure_seq']}: {r['failure_reason']}")
                st.caption(f"{r['verified_events']} event(s) verified before the failure; "
                           f"{r['total_events']} events stored.")
            st.caption(f"Checked at {r['checked_at']}")
        cp = st.session_state.get("checkpoint")
        if cp:
            st.info("Save this checkpoint OUTSIDE the application (separate server, USB, print-out).")
            st.code(json.dumps({k: cp[k] for k in ("seq", "hash", "created_at", "signature")}, indent=2),
                    language="json")
        st.divider()

    st.subheader("Audit log" + ("" if is_admin else f" (site '{me['site']}')"))
    c1, c2, c3 = st.columns(3)
    action = c1.text_input("Filter by action (e.g. auth.login_failed)")
    target = c2.text_input("Filter by target ID")
    limit = c3.number_input("Rows", 10, 1000, 100, 10)
    try:
        params = {"action": action.strip(), "target_id": target.strip(), "limit": int(limit)}
        data = api.audit_logs(**params) if is_admin else api.scoped_audit_logs(**params)
    except ApiError as e:
        if e.status == 401:
            raise
        st.error(e.detail)
        return
    if not data["items"]:
        st.info("No audit events match.")
        return
    df = pd.DataFrame(data["items"])
    df["hash"] = df["hash"].str[:16] + "…"
    df["prev_hash"] = df["prev_hash"].str[:16] + "…"
    df["metadata"] = df["metadata"].apply(lambda m: json.dumps(m)[:200])
    st.write(f"{data['total']} event(s)")
    st.dataframe(df[["seq", "timestamp", "action", "actor_role", "actor_id", "target_type",
                     "target_id", "metadata", "prev_hash", "hash"]],
                 hide_index=True, use_container_width=True)
