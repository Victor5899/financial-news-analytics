"""Common per-page setup: page config, global styling, and the shared sidebar."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.styles import apply_global_styles


def bootstrap_page(title: str, icon: str = "📊") -> dict[str, Any]:
    """
    Call this as the first line of every page (and the main entry point).

    Sets wide layout + page title/icon, injects shared CSS, and renders the
    sidebar (ticker selector, status badges, settings). Returns the current
    global settings dict from ``st.session_state``.
    """
    st.set_page_config(
        page_title=f"{title} · FinNews AI",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_global_styles()
    return render_sidebar()
