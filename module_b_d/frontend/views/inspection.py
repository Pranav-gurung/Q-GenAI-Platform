"""Operator view: upload / capture an image, run inspection, see results."""
import streamlit as st

from api_client import ApiClient, ApiError
from views.common import defects_table, verdict

MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}


def show_result(api: ApiClient, res: dict, original: bytes | None = None) -> None:
    st.markdown(verdict(res["final_decision"]), unsafe_allow_html=True)
    if res["final_decision"] != res["inspection_status"]:
        st.caption(f"Original AI decision: **{res['inspection_status']}** (changed by human review)")
    st.caption(f"Inspection ID: `{res['inspection_id']}` · {res['timestamp']} · by {res['inspected_by']}")

    if res["model_status"] != "loaded":
        st.warning("No trained defect model is active. No AI detections were produced; this part "
                   "has been routed to manual REVIEW.")

    c = st.columns(5)
    c[0].metric("Defects detected", res["defect_count"])
    c[1].metric("Processing time", f"{res['processing_time_ms']:.0f} ms")
    c[2].metric("Model inference", f"{res['inference_time_ms']:.0f} ms" if res["inference_time_ms"] else "n/a")
    c[3].metric("Confidence threshold", f"{res['confidence_threshold']:.2f}",
                "override" if res["threshold_overridden"] else None, delta_color="off")
    c[4].metric("Policy version", res["policy_version"])
    st.caption(f"Model: {res['model_name'] or 'none'} · version {res['model_version'] or 'n/a'} · "
               f"status {res['model_status']}")

    with st.expander("Decision reasons", expanded=True):
        for r in res["ai_result"]["decision_reasons"]:
            st.write(f"- {r}")

    left, right = st.columns(2)
    if original is not None:
        left.image(original, caption="Original", use_container_width=True)
    annotated = None
    if res["annotated_image_available"]:
        try:
            annotated = api.annotated_image(res["inspection_id"])
            (right if original is not None else left).image(annotated, caption="Annotated",
                                                            use_container_width=True)
        except ApiError as e:
            right.info(f"Annotated image unavailable: {e.detail}")

    if res["defects"]:
        st.dataframe(defects_table(res["defects"]), hide_index=True, use_container_width=True)
    else:
        st.info("No detections above the confidence threshold."
                if res["model_status"] == "loaded" else "No detections (no model was run).")

    if res["overrides"]:
        st.subheader("Human review history")
        for o in res["overrides"]:
            st.write(f"- {o['created_at']}: **{o['old_decision']} → {o['new_decision']}** "
                     f"({o['actor_role']}) — {o['reason']}")

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download JSON report", api.report_bytes(res["inspection_id"]),
                       file_name=f"qgenai-report-{res['inspection_id']}.json",
                       mime="application/json", use_container_width=True)
    if annotated:
        d2.download_button("⬇️ Download annotated image", annotated,
                           file_name=f"qgenai-annotated-{res['inspection_id']}.png",
                           mime="image/png", use_container_width=True)


def render(api: ApiClient, me: dict) -> None:
    st.header("Product inspection")
    policy = api.policy()
    lo, hi = policy["operator_min_threshold"], policy["operator_max_threshold"]
    can_override = "quality:threshold_override" in me["permissions"]

    src = st.radio("Image source", ["Upload file", "Webcam capture"], horizontal=True)
    if src == "Upload file":
        f = st.file_uploader("Product image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    else:
        f = st.camera_input("Capture product image (browser webcam)")

    col1, col2 = st.columns([2, 1])
    with col2:
        if can_override:
            thr = st.slider("Confidence threshold", 0.01, 0.99,
                            float(policy["default_confidence_threshold"]), 0.01)
            st.caption(f"Operator range: {lo:.2f}–{hi:.2f}. Values outside it are a recorded override.")
        else:
            thr = st.slider("Confidence threshold", float(lo), float(hi),
                            float(policy["default_confidence_threshold"]), 0.01)
            st.caption(f"Allowed range for your role: {lo:.2f}–{hi:.2f}")
        reason = None
        if not (lo <= thr <= hi):
            reason = st.text_area("Override reason (required, min. 10 characters)", max_chars=2000)
        go = st.button("🔍 Inspect product", type="primary", use_container_width=True, disabled=f is None)

    with col1:
        if f is None:
            st.info("Upload or capture an image to begin.")
        else:
            st.image(f, caption="Preview", use_container_width=True)

    if go and f is not None:
        data = f.getvalue()
        name = getattr(f, "name", None) or "webcam.jpg"
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else "jpg"
        if ext not in MIME:  # camera_input returns e.g. "image.jpg" / no name in some versions
            name, ext = "webcam.jpg", "jpg"
        with st.spinner("Running defect detection..."):
            try:
                res = api.predict(name, data, MIME[ext], thr, reason)
            except ApiError as e:
                if e.status == 401:
                    raise
                st.error(e.detail)
                return
        st.session_state.last_result = res
        st.session_state.last_original = data

    if st.session_state.get("last_result"):
        st.divider()
        show_result(api, st.session_state.last_result, st.session_state.get("last_original"))
