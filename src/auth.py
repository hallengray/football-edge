"""Simple password gate for the Streamlit app.

Reads the password from APP_PASSWORD (env var or Streamlit secrets).
Once entered correctly, stays authenticated for the session.
"""

from __future__ import annotations

import os

import streamlit as st


def _expected_password() -> str:
    """Pull the password from env first, falling back to st.secrets."""
    pw = os.getenv("APP_PASSWORD")
    if pw:
        return pw
    try:
        return st.secrets.get("APP_PASSWORD", "")
    except (FileNotFoundError, KeyError):
        return ""


def check_password() -> bool:
    """Render the login form and return True only if the user is authenticated."""
    expected = _expected_password()

    if not expected:
        st.error(
            "APP_PASSWORD is not configured. "
            "Set it in `.env` locally or in Streamlit secrets when deployed."
        )
        return False

    if st.session_state.get("authenticated"):
        return True

    st.markdown("# ⚽ Football Edge")
    st.caption("Personal value betting tool")

    with st.form("login_form", clear_on_submit=True):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

        if submitted:
            if password == expected:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")

    return False
