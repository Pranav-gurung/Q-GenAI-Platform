"""Inspection policy (all roles can view; admins edit) and model management (admin)."""
import pandas as pd
import streamlit as st

from api_client import ApiClient, ApiError


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def render(api: ApiClient, me: dict) -> None:
    st.header("Inspection policy & models")
    p = api.policy()
    st.subheader(f"Active policy — version {p['version']}")
    st.json({k: p[k] for k in ("default_confidence_threshold", "operator_min_threshold",
                               "operator_max_threshold", "fail_confidence", "fail_classes",
                               "ignored_classes", "max_allowed_defects", "require_approved_model",
                               "change_reason")})
    st.caption("PASS = no disqualifying defects. FAIL = more disqualifying detections "
               "(fail class, confidence ≥ fail confidence) than allowed. REVIEW = model unavailable/"
               "generic/unapproved, or uncertain detections. Confidence ≠ physical severity; "
               "results are not a BIS/ISO conformity statement.")

    if "policy:edit" in me["permissions"]:
        with st.expander("Create new policy version"):
            with st.form("policy"):
                c1, c2, c3 = st.columns(3)
                dct = c1.number_input("Default confidence threshold", 0.01, 0.99, p["default_confidence_threshold"], 0.01)
                omin = c2.number_input("Operator min threshold", 0.01, 0.99, p["operator_min_threshold"], 0.01)
                omax = c3.number_input("Operator max threshold", 0.01, 0.99, p["operator_max_threshold"], 0.01)
                fconf = c1.number_input("Fail confidence", 0.01, 0.99, p["fail_confidence"], 0.01)
                maxd = c2.number_input("Max allowed disqualifying defects", 0, 1000, p["max_allowed_defects"])
                req = c3.checkbox("Require approved model", p["require_approved_model"])
                fcls = st.text_input("Fail classes (comma-separated, empty = all)", ", ".join(p["fail_classes"]))
                icls = st.text_input("Ignored classes (comma-separated)", ", ".join(p["ignored_classes"]))
                reason = st.text_area("Change reason (mandatory)")
                if st.form_submit_button("Save as new version", type="primary"):
                    try:
                        api.update_policy({
                            "default_confidence_threshold": dct, "operator_min_threshold": omin,
                            "operator_max_threshold": omax, "fail_confidence": fconf,
                            "fail_classes": _csv(fcls), "ignored_classes": _csv(icls),
                            "max_allowed_defects": int(maxd), "require_approved_model": req,
                            "change_reason": reason})
                        st.success("New policy version saved and audit-logged.")
                        st.rerun()
                    except ApiError as e:
                        if e.status == 401:
                            raise
                        st.error(e.detail)
        with st.expander("Policy history"):
            st.dataframe(pd.DataFrame(api.policy_history()), hide_index=True, use_container_width=True)

    if "model:manage" not in me["permissions"]:
        return

    st.divider()
    st.subheader("Model versions")
    m = api.models()
    if m["active"]:
        st.write(f"**Active:** {m['active']['model_name']} · {m['active']['model_version']} · "
                 f"{m['active']['model_status']} — {m['active']['message']}")
    if m["registered"]:
        st.dataframe(pd.DataFrame(m["registered"]), hide_index=True, use_container_width=True)
    st.caption("Weight files in the models directory: " + (", ".join(m["weight_files"]) or "none"))

    with st.form("register_model"):
        c1, c2 = st.columns(2)
        wf = c1.selectbox("Weights file", m["weight_files"] or ["(none found)"])
        label = c2.text_input("Version label (e.g. defects-v1.0)")
        notes = st.text_input("Notes (dataset, metrics, training date)")
        appr = st.checkbox("Approved for production use")
        if st.form_submit_button("Register model"):
            try:
                api.register_model({"weights_file": wf, "version_label": label, "notes": notes,
                                    "approved": appr})
                st.success("Registered.")
                st.rerun()
            except ApiError as e:
                if e.status == 401:
                    raise
                st.error(e.detail)

    if m["registered"]:
        opts = {f"{r['version_label']} ({r['weights_file']})": r for r in m["registered"]}
        sel = opts[st.selectbox("Registered version", list(opts))]
        c1, c2 = st.columns(2)
        if c1.button("Activate (load for new inspections)", use_container_width=True):
            try:
                with st.spinner("Loading model..."):
                    info = api.activate_model(sel["id"])
                st.success(f"Activated: {info['message']}")
                st.rerun()
            except ApiError as e:
                st.error(e.detail)
        if c2.button("Unapprove" if sel["approved"] else "Approve", use_container_width=True):
            try:
                api.patch_model(sel["id"], not sel["approved"])
                st.rerun()
            except ApiError as e:
                st.error(e.detail)
