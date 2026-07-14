"""Sentiment Analytics — aggregated views over Phase 2/3 FinBERT results already stored in PostgreSQL."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components import charts
from app.components.metric_cards import metric_grid
from app.components.styles import empty_state, section_header
from app.services import analytics_service, database_service, prediction_service

settings = bootstrap_page("Sentiment Analytics", icon="💬")

section_header("Sentiment Analytics", "Aggregated FinBERT sentiment across all tracked tickers.", icon="💬")

col1, col2 = st.columns([2, 1])
with col1:
    tickers = st.multiselect(
        "Tickers", options=prediction_service.SUPPORTED_TICKERS,
        default=settings["watchlist_tickers"] or prediction_service.SUPPORTED_TICKERS,
    )
with col2:
    days = st.selectbox("Lookback window", options=[30, 90, 180, 365, 3650],
                         format_func=lambda d: f"Last {d} days" if d < 3650 else "All time", index=3)

start = date.today() - timedelta(days=days)
tickers_tuple = tuple(tickers) if tickers else None

trend = analytics_service.get_sentiment_daily_trend(tickers=tickers_tuple, start=start)
ticker_summary = analytics_service.get_ticker_sentiment_summary(tickers=tickers_tuple)

if trend.empty:
    empty_state("No sentiment data available for the selected filters.")
    st.stop()

metric_grid(
    [
        {"label": "Articles Analyzed", "value": f"{int(trend['total'].sum()):,}", "icon": "📰"},
        {"label": "Average Confidence", "value": f"{trend['mean_confidence'].mean():.1%}", "icon": "🎯"},
        {"label": "Positive Share", "value": f"{trend['positive'].sum() / trend['total'].sum():.1%}", "icon": "😊"},
        {"label": "Negative Share", "value": f"{trend['negative'].sum() / trend['total'].sum():.1%}", "icon": "☹️"},
    ]
)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Daily Trend & Rolling Avg", "Positive vs Neutral vs Negative", "Sentiment Timeline", "Ticker-wise Sentiment"]
)

with tab1:
    st.plotly_chart(charts.sentiment_trend_chart(trend), use_container_width=True)
    st.caption("Daily mean FinBERT sentiment score (-1 negative, 0 neutral, +1 positive) with a 7-day rolling average.")

with tab2:
    st.plotly_chart(charts.sentiment_share_chart(trend), use_container_width=True)

with tab3:
    timeline_df = database_service.get_sentiment_articles_df(
        tickers=tickers_tuple, start=start, limit=2000
    )
    if timeline_df.empty:
        empty_state("No articles in this window.")
    else:
        st.plotly_chart(charts.sentiment_timeline_scatter(timeline_df), use_container_width=True)

with tab4:
    if ticker_summary.empty:
        empty_state("No ticker-level sentiment data available.")
    else:
        st.plotly_chart(charts.ticker_sentiment_bar(ticker_summary), use_container_width=True)
        st.dataframe(
            ticker_summary.style.format({"mean_confidence": "{:.1%}"}),
            width="stretch", hide_index=True,
        )
