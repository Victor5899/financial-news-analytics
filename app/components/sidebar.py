"""
Global sidebar: branding, live status badges, and cross-page settings.

Values are stored in ``st.session_state`` so every page shares the same
active ticker / auto-refresh / save-prediction preferences without any
page needing to duplicate the widgets.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit_option_menu import option_menu

from app.components import styles
from app.services import database_service, prediction_service

_DEFAULTS: dict[str, Any] = {
    "active_ticker": "AAPL",
    "watchlist_tickers": list(prediction_service.SUPPORTED_TICKERS),
    "auto_refresh": False,
    "refresh_interval": 60,
    "save_predictions": True,
    "chart_theme": "Light",
}

_TICKER_ICONS = {
    "AAPL": "apple",
    "TSLA": "lightning-charge",
    "MSFT": "windows",
    "NVDA": "gpu-card",
    "AMZN": "cart3",
}


def _init_state() -> None:
    for key, value in _DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _status_row(label: str, ok: bool, detail: str = "") -> None:
    kind = "positive" if ok else "negative"
    icon = "🟢" if ok else "🔴"
    st.markdown(
        f"{icon} **{label}** {styles.badge_html('OK' if ok else 'DOWN', kind)}",
        unsafe_allow_html=True,
    )
    if detail and not ok:
        st.caption(detail)


def render_sidebar() -> dict[str, Any]:
    """Render the shared sidebar and return the current global settings dict."""
    _init_state()

    with st.sidebar:
        st.markdown("## 📊 Financial News Analytics")
        st.caption("AI-powered market intelligence dashboard")
        st.divider()

        st.markdown("**Active Ticker**")
        selected = option_menu(
            menu_title=None,
            options=prediction_service.SUPPORTED_TICKERS,
            icons=[_TICKER_ICONS.get(t, "graph-up") for t in prediction_service.SUPPORTED_TICKERS],
            default_index=prediction_service.SUPPORTED_TICKERS.index(st.session_state["active_ticker"])
            if st.session_state["active_ticker"] in prediction_service.SUPPORTED_TICKERS
            else 0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "nav-link": {"font-size": "0.85rem", "padding": "0.4rem 0.6rem"},
            },
            key="sidebar_active_ticker_menu",
        )
        st.session_state["active_ticker"] = selected

        st.divider()
        st.markdown("**System Status**")
        db_status = database_service.database_status()
        _status_row("Database", db_status["connected"], db_status.get("error", ""))

        model_status = prediction_service.model_file_status()
        _status_row("Model Artifact", model_status["exists"], f"Missing: {model_status['path']}")

        _status_row("Finnhub API Key", prediction_service.api_key_status())

        st.divider()
        with st.expander("⚙️ Quick Settings", expanded=False):
            st.session_state["watchlist_tickers"] = st.multiselect(
                "Watchlist tickers",
                options=prediction_service.SUPPORTED_TICKERS,
                default=st.session_state["watchlist_tickers"],
                key="sidebar_watchlist",
            )
            st.session_state["save_predictions"] = st.toggle(
                "Save live predictions to database",
                value=st.session_state["save_predictions"],
                key="sidebar_save_predictions",
            )
            st.session_state["auto_refresh"] = st.toggle(
                "Auto-refresh data",
                value=st.session_state["auto_refresh"],
                key="sidebar_auto_refresh",
            )
            if st.session_state["auto_refresh"]:
                st.session_state["refresh_interval"] = st.slider(
                    "Refresh interval (seconds)", 15, 300,
                    value=st.session_state["refresh_interval"], step=15,
                    key="sidebar_refresh_interval",
                )

        st.divider()
        st.caption("Phase 9 · Streamlit presentation layer over the Phase 1-8 ML pipeline")
        st.caption("Built with Streamlit · Plotly · SHAP · XGBoost · FinBERT")

    return {k: st.session_state[k] for k in _DEFAULTS}
