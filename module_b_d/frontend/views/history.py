"""Inspection history (scoped by the backend to what the user may see)."""
import pandas as pd
import streamlit as st

from api_client import ApiClient, ApiError
from views.inspection import show_result


def render(api: ApiClient, me: dict) -> None:
    st.header("Inspection history")
    scope = {"operator": "your own inspections", "quality_manager": f"inspections at site '{me['site']}'",
             "admin": "all inspections"}.get(me["role"], "")
    st.caption(f"Showing {scope}.")

    c1, c2, c3 = st.columns(3)
    ai = c1.selectbox("AI decision", ["", "PASS", "FAIL", "REVIEW"])
    final = c2.selectbox("Final decision", ["", "PASS", "FAIL", "REVIEW"])
    limit = c3.number_input("Max rows", 10, 500, 100, 10)
    data = api.inspections(ai_decision=ai, final_decision=final, limit=int(limit))

    if not data["items"]:
        st.info("No inspections yet.")
        return
    st.write(f"{data['total']} inspection(s)")
    df = pd.DataFrame(data["items"])
    st.dataframe(df[["timestamp", "inspection_id", "inspected_by", "site", "ai_decision",
                     "final_decision", "defect_count", "model_version", "has_override"]],
                 hide_index=True, use_container_width=True)

    sel = st.selectbox("Open inspection", [""] + [i["inspection_id"] for i in data["items"]])
    if sel:
        try:
            show_result(api, api.inspection(sel))
        except ApiError as e:
            st.error(e.detail)
        with st.expander("Section C payload (image-free data for root-cause analysis)"):
            st.json(api.rca_payload(sel))
