"""
Financial News Analytics — Streamlit entry point (Phase 9).

This is purely the presentation layer's landing page. All ML / data logic
lives in ``src/`` (Phases 1-8) and is reused, never duplicated, via
``app/services/*``. See ``pages/`` for the full multi-page dashboard.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components.metric_cards import metric_grid
from app.components.styles import section_header
from app.services import database_service, prediction_service

bootstrap_page("Home", icon="🏠")

section_header(
    "Financial News Analytics",
    subtitle="Real-time sentiment analysis & AI-driven stock-direction predictions, "
    "powered by FinBERT + XGBoost.",
    icon="📊",
)

db_status = database_service.database_status()

if db_status["connected"]:
    metric_grid(
        [
            {"label": "Total Articles", "value": f"{database_service.count_articles():,}", "icon": "📰"},
            {"label": "Total Predictions", "value": f"{database_service.count_predictions():,}", "icon": "🎯"},
            {"label": "Tracked Tickers", "value": str(len(prediction_service.SUPPORTED_TICKERS)), "icon": "🏷️"},
            {"label": "Sentiment Records", "value": f"{database_service.count_sentiment_results():,}", "icon": "🧠"},
        ]
    )
else:
    st.error(
        "⚠️ Could not connect to the database. Live metrics are unavailable — "
        f"check `DATABASE_URL` in your `.env` file. ({db_status['error']})"
    )

st.write("")
st.markdown("### 🧭 Explore the dashboard")
st.caption("Every page below is powered by the same production pipeline — nothing here retrains "
           "or re-implements the model.")

nav_cards = [
    ("pages/1_Dashboard.py", "📈 Dashboard", "KPIs, recent activity, and distribution charts at a glance."),
    ("pages/2_Live_Prediction.py", "⚡ Live Prediction", "Run the real-time pipeline for any tracked ticker."),
    ("pages/3_Live_News.py", "🗞️ Live News", "Latest Finnhub headlines with live FinBERT sentiment."),
    ("pages/4_Historical_Predictions.py", "🕘 Historical Predictions", "Filter, search, and export past predictions."),
    ("pages/5_Technical_Analysis.py", "📉 Technical Analysis", "Price action with SMA/EMA/RSI/MACD/ATR/Bollinger."),
    ("pages/6_Sentiment_Analytics.py", "💬 Sentiment Analytics", "Sentiment trends and ticker-level breakdowns."),
    ("pages/7_Model_Explainability.py", "🔍 Model Explainability", "SHAP-powered explanations for every prediction."),
    ("pages/8_System_Analytics.py", "🖥️ System Analytics", "Database, model, and training-quality metrics."),
    ("pages/9_Settings.py", "⚙️ Settings", "Tickers, refresh, saving behaviour, and system status."),
]

cols = st.columns(3)
for i, (target, label, desc) in enumerate(nav_cards):
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(desc)
            st.page_link(target, label="Open →")

st.write("")
st.divider()
st.caption(
    "Backend: Finnhub · GDELT · FinBERT · PostgreSQL · XGBoost (Phases 1-8). "
    "This app is a read-mostly presentation layer — it never retrains the model or "
    "changes prediction logic."
)
