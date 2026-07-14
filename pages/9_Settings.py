"""Settings — ticker/theme/refresh preferences, save behaviour, exports, and system status."""

from __future__ import annotations

import json

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components.styles import section_header
from app.services import database_service, prediction_service

bootstrap_page("Settings", icon="⚙️")

section_header("Settings", "Preferences shared across every page in this session.", icon="⚙️")

# ── Preferences ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("**Ticker Selection**")
        st.session_state["watchlist_tickers"] = st.multiselect(
            "Tracked tickers",
            options=prediction_service.SUPPORTED_TICKERS,
            default=st.session_state.get("watchlist_tickers", prediction_service.SUPPORTED_TICKERS),
            key="settings_watchlist",
        )
        st.session_state["active_ticker"] = st.selectbox(
            "Default active ticker",
            options=prediction_service.SUPPORTED_TICKERS,
            index=prediction_service.SUPPORTED_TICKERS.index(st.session_state.get("active_ticker", "AAPL")),
            key="settings_active_ticker",
        )

    with st.container(border=True):
        st.markdown("**Theme**")
        st.session_state["chart_theme"] = st.radio(
            "Chart theme",
            options=["Light", "Dark"],
            index=0 if st.session_state.get("chart_theme", "Light") == "Light" else 1,
            horizontal=True,
            key="settings_chart_theme",
        )
        st.caption(
            "App-wide dark/light mode follows your Streamlit client setting "
            "(top-right menu → Settings → Theme). This toggle affects chart styling."
        )

with col2:
    with st.container(border=True):
        st.markdown("**Auto Refresh**")
        st.session_state["auto_refresh"] = st.toggle(
            "Enable auto-refresh on data pages",
            value=st.session_state.get("auto_refresh", False),
            key="settings_auto_refresh",
        )
        st.session_state["refresh_interval"] = st.slider(
            "Refresh interval (seconds)", 15, 300,
            value=st.session_state.get("refresh_interval", 60), step=15,
            disabled=not st.session_state["auto_refresh"],
            key="settings_refresh_interval",
        )

    with st.container(border=True):
        st.markdown("**Prediction Saving**")
        st.session_state["save_predictions"] = st.toggle(
            "Save live predictions to PostgreSQL",
            value=st.session_state.get("save_predictions", True),
            key="settings_save_predictions",
        )
        st.caption("When disabled, the Live Prediction page still runs inference — it just won't "
                   "persist the result to the `predictions` table.")

st.divider()

# ── Download options ─────────────────────────────────────────────────────────────
section_header("Download Options", icon="⬇️")
d1, d2, d3 = st.columns(3)
with d1:
    preds_df = database_service.get_predictions_df()
    st.download_button(
        "⬇️ Export all predictions (CSV)",
        data=preds_df.to_csv(index=False).encode("utf-8"),
        file_name="all_predictions.csv", mime="text/csv",
        width="stretch", disabled=preds_df.empty,
    )
with d2:
    articles_df = database_service.get_articles_df(limit=5000)
    st.download_button(
        "⬇️ Export recent articles (CSV)",
        data=articles_df.to_csv(index=False).encode("utf-8"),
        file_name="recent_articles.csv", mime="text/csv",
        width="stretch", disabled=articles_df.empty,
    )
with d3:
    settings_export = {k: st.session_state[k] for k in (
        "watchlist_tickers", "active_ticker", "chart_theme",
        "auto_refresh", "refresh_interval", "save_predictions",
    ) if k in st.session_state}
    st.download_button(
        "⬇️ Export current settings (JSON)",
        data=json.dumps(settings_export, indent=2).encode("utf-8"),
        file_name="dashboard_settings.json", mime="application/json",
        width="stretch",
    )

st.divider()

# ── System status ────────────────────────────────────────────────────────────────
section_header("System Status", icon="🩺")
s1, s2, s3 = st.columns(3)

with s1:
    st.markdown("**Database Status**")
    db_status = database_service.database_status()
    if db_status["connected"]:
        st.success("Connected")
    else:
        st.error(f"Unavailable: {db_status['error']}")

with s2:
    st.markdown("**Model Status**")
    model_status = prediction_service.model_file_status()
    if model_status["exists"]:
        meta = model_status.get("metadata", {})
        st.success(
            f"Loaded — {model_status.get('n_features', '?')} features, "
            f"classes {model_status.get('classes', [])}"
        )
        if meta:
            st.caption(f"Trained on {meta.get('train_rows', '?')} rows")
    else:
        st.error(f"Missing artifact: {model_status['path']}")

with s3:
    st.markdown("**API Status**")
    if prediction_service.api_key_status():
        st.success("Finnhub API key configured")
    else:
        st.error("FINNHUB_API_KEY not set")

if st.button("🧹 Clear all cached data"):
    database_service.clear_all_caches()
    st.success("Caches cleared. Data will reload on next page visit.")
