"""Technical Analysis — price action and indicators computed with Phase 4's exact math."""

from __future__ import annotations

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components import charts
from app.components.styles import empty_state, section_header
from app.services import analytics_service, prediction_service

settings = bootstrap_page("Technical Analysis", icon="📉")

section_header(
    "Technical Analysis",
    "Indicators computed with the same primitives used by the Phase 4 feature engineer "
    "(SMA/EMA/RSI/MACD/Bollinger/ATR) — not re-derived.",
    icon="📉",
)

ticker = st.selectbox(
    "Ticker",
    options=prediction_service.SUPPORTED_TICKERS,
    index=prediction_service.SUPPORTED_TICKERS.index(settings["active_ticker"])
    if settings["active_ticker"] in prediction_service.SUPPORTED_TICKERS
    else 0,
)
st.session_state["active_ticker"] = ticker

with st.spinner(f"Loading price history for {ticker} …"):
    df = analytics_service.compute_technical_indicators_series(ticker)

if df.empty:
    empty_state(f"No price history found for {ticker}. Run `scripts/fetch_prices.py` first.")
    st.stop()

st.caption(f"{len(df)} trading days · {df['trading_date'].min()} → {df['trading_date'].max()}")

tab_price, tab_macd, tab_rsi, tab_bb, tab_atr, tab_vol = st.tabs(
    ["Price & Moving Averages", "MACD", "RSI", "Bollinger Bands", "ATR", "Volume"]
)

with tab_price:
    overlays = st.multiselect(
        "Overlays", options=["sma_10", "sma_20", "ema_10", "ema_20"],
        default=["sma_20", "ema_20"], format_func=str.upper,
    )
    st.plotly_chart(charts.price_candlestick(df, overlays=overlays), use_container_width=True)

with tab_macd:
    st.plotly_chart(charts.macd_chart(df), use_container_width=True)

with tab_rsi:
    st.plotly_chart(charts.rsi_chart(df), use_container_width=True)
    st.caption("Above 70 is commonly read as overbought; below 30 as oversold.")

with tab_bb:
    st.plotly_chart(charts.bollinger_chart(df), use_container_width=True)

with tab_atr:
    st.plotly_chart(charts.atr_chart(df), use_container_width=True)
    st.caption("Average True Range measures volatility, not direction.")

with tab_vol:
    st.plotly_chart(charts.volume_bar_chart(df), use_container_width=True)

st.divider()
with st.expander("📄 Raw indicator data"):
    st.dataframe(df, width="stretch", hide_index=True)
