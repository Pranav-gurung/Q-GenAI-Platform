"""Quality-manager view: review FAIL/REVIEW inspections and record decisions."""
import streamlit as st

from api_client import ApiClient, ApiError
from views.inspection import show_result


def render(api: ApiClient, me: dict) -> None:
    st.header("Review queue")
    st.caption("FAIL and REVIEW inspections at your site without a recorded human decision. "
               "The original AI result is never modified; your decision is stored separately.")
    queue = api.review_queue()
    manual = st.text_input("…or enter any inspection ID in your scope")
    options = ([manual.strip()] if manual.strip() else []) + queue
    if not options:
        st.success("Nothing waiting for review.")
        return
    sel = st.selectbox("Inspection", options)
    try:
        res = api.inspection(sel)
    except ApiError as e:
        st.error(e.detail)
        return
    show_result(api, res)

    st.subheader("Record review decision")
    with st.form("override"):
        new = st.radio("Final decision", ["PASS", "FAIL", "REVIEW"], horizontal=True,
                       index=["PASS", "FAIL", "REVIEW"].index(res["final_decision"]))
        reason = st.text_area("Reason (mandatory, min. 10 characters)", max_chars=2000)
        ok = st.form_submit_button("Save decision", type="primary")
    if ok:
        try:
            api.override(sel, new, reason)
            st.success("Decision recorded and audit-logged.")
            st.session_state.pop("last_result", None)
            st.rerun()
        except ApiError as e:
            if e.status == 401:
                raise
            st.error(e.detail)
