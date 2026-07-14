"""Historical Predictions — filter, search, and export the full prediction history."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components import charts
from app.components.metric_cards import metric_grid
from app.components.styles import empty_state, section_header
from app.components.tables import download_csv_button, render_table
from app.services import database_service, prediction_service

bootstrap_page("Historical Predictions", icon="🕘")

section_header("Historical Predictions", "Every prediction ever saved to PostgreSQL by the live pipeline.",
               icon="🕘")

with st.container(border=True):
    st.markdown("**Filters**")
    row1 = st.columns([1.4, 1, 1, 1.2])
    with row1[0]:
        date_range = st.date_input(
            "Date range",
            value=(date.today() - timedelta(days=365), date.today()),
        )
    with row1[1]:
        tickers = st.multiselect("Ticker", options=prediction_service.SUPPORTED_TICKERS)
    with row1[2]:
        pred_filter = st.multiselect("Prediction", options=["BUY", "HOLD", "SELL"])
    with row1[3]:
        min_conf = st.slider("Min. confidence", 0.0, 1.0, 0.0, 0.05)

    search = st.text_input("Search headline", placeholder="e.g. earnings, guidance, tariffs …")

start, end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2
              else (date.today() - timedelta(days=365), date.today()))

df = database_service.get_predictions_df(
    tickers=tuple(tickers) if tickers else None,
    predictions=tuple(pred_filter) if pred_filter else None,
    start=start,
    end=end,
    min_confidence=min_conf,
    search=search or None,
)

st.divider()

if df.empty:
    empty_state("No predictions match the current filters.")
    st.stop()

# ── Summary metrics ──────────────────────────────────────────────────────────────
metric_grid(
    [
        {"label": "Matching Predictions", "value": f"{len(df):,}", "icon": "🔢"},
        {"label": "Avg. Confidence", "value": f"{df['confidence'].mean():.1%}", "icon": "📐"},
        {"label": "Most Common Signal", "value": df["prediction"].mode().iat[0], "icon": "🏆"},
        {"label": "Tickers Covered", "value": str(df["ticker"].nunique()), "icon": "🏷️"},
    ]
)

st.write("")
tab1, tab2 = st.tabs(["Confidence Over Time", "Predictions Per Day"])
with tab1:
    st.plotly_chart(charts.predictions_confidence_line(df), use_container_width=True)
with tab2:
    st.plotly_chart(charts.predictions_volume_line(df), use_container_width=True)

st.divider()
section_header("Prediction History", icon="📋")

display_df = df[
    ["created_at", "ticker", "prediction", "confidence", "buy_probability",
     "hold_probability", "sell_probability", "headline", "published_at"]
].sort_values("created_at", ascending=False)

render_table(display_df, key="historical_predictions_table")
download_csv_button(display_df, "historical_predictions.csv")
