"""Administrator view: create users, change roles, enable/disable accounts."""
import pandas as pd
import streamlit as st

from api_client import ApiClient, ApiError

ROLES = ["operator", "quality_manager", "admin"]


def render(api: ApiClient, me: dict) -> None:
    st.header("User management")
    users = api.users()
    st.dataframe(pd.DataFrame(users)[["username", "full_name", "role", "site", "is_active", "created_at"]],
                 hide_index=True, use_container_width=True)

    st.subheader("Create user")
    with st.form("create_user", clear_on_submit=True):
        c1, c2 = st.columns(2)
        username = c1.text_input("Username (3-64 chars: letters, digits, . _ -)")
        full_name = c2.text_input("Full name")
        role = c1.selectbox("Role", ROLES)
        site = c2.text_input("Site / production line", "default")
        pw = st.text_input("Initial password (min 10 chars, letters + digits)", type="password")
        if st.form_submit_button("Create user", type="primary"):
            try:
                api.create_user({"username": username, "full_name": full_name, "role": role,
                                 "site": site, "password": pw})
                st.success(f"User '{username}' created.")
                st.rerun()
            except ApiError as e:
                if e.status == 401:
                    raise
                st.error(e.detail)

    st.subheader("Edit user")
    by_name = {u["username"]: u for u in users}
    sel = st.selectbox("User", list(by_name))
    u = by_name[sel]
    with st.form("edit_user"):
        c1, c2, c3 = st.columns(3)
        role = c1.selectbox("Role", ROLES, index=ROLES.index(u["role"]))
        site = c2.text_input("Site", u["site"])
        active = c3.checkbox("Active", u["is_active"])
        new_pw = st.text_input("Reset password (leave empty to keep)", type="password")
        if st.form_submit_button("Save changes"):
            body = {}
            if role != u["role"]:
                body["role"] = role
            if site != u["site"]:
                body["site"] = site
            if active != u["is_active"]:
                body["is_active"] = active
            if new_pw:
                body["password"] = new_pw
            if not body:
                st.info("Nothing changed.")
            else:
                try:
                    api.update_user(u["id"], body)
                    st.success("Saved. The user's existing sessions were invalidated where required.")
                    st.rerun()
                except ApiError as e:
                    if e.status == 401:
                        raise
                    st.error(e.detail)
