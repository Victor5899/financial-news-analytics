"""Live Prediction — run the real-time pipeline (Finnhub → FinBERT → features → XGBoost)."""

from __future__ import annotations

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components.prediction_card import render_prediction_result
from app.components.styles import section_header
from app.services import prediction_service
from src.realtime.finnhub_client import FinnhubError
from src.realtime.realtime_pipeline import NoNewsError, RealtimePipelineError

settings = bootstrap_page("Live Prediction", icon="⚡")

section_header(
    "Live Prediction",
    "Runs the full Phase 8 real-time pipeline: Finnhub news → FinBERT sentiment → "
    "feature engineering → XGBoost inference.",
    icon="⚡",
)

col_select, col_button = st.columns([3, 1])
with col_select:
    ticker = st.selectbox(
        "Select a ticker",
        options=prediction_service.SUPPORTED_TICKERS,
        index=prediction_service.SUPPORTED_TICKERS.index(settings["active_ticker"])
        if settings["active_ticker"] in prediction_service.SUPPORTED_TICKERS
        else 0,
    )
with col_button:
    st.write("")
    st.write("")
    predict_clicked = st.button("🔮 Predict", type="primary", width="stretch")

if predict_clicked:
    st.session_state["active_ticker"] = ticker
    save = st.session_state.get("save_predictions", True)

    with st.spinner(f"Fetching news, scoring sentiment, and running inference for {ticker} …"):
        try:
            result = prediction_service.run_live_prediction(ticker, save=save)
        except NoNewsError:
            st.warning(f"No recent news articles were found for **{ticker}**. Try again later.")
            result = None
        except FinnhubError as exc:
            st.error(f"Finnhub API error: {exc}")
            result = None
        except RealtimePipelineError as exc:
            st.error(f"Pipeline error: {exc}")
            result = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while predicting: {exc}")
            result = None

    if result is not None:
        if save and result.saved_id:
            st.toast(f"✅ Prediction saved (id={result.saved_id})", icon="✅")
            st.success(f"Prediction complete for **{ticker}** — saved to the database.")
        else:
            st.success(f"Prediction complete for **{ticker}** (not saved).")

        source_name = None
        try:
            latest_news = prediction_service.fetch_latest_news(ticker)
            if latest_news:
                source_name = latest_news[0].get("source_name")
        except Exception:  # noqa: BLE001
            pass

        st.session_state["last_prediction_result"] = result
        st.session_state["last_prediction_source"] = source_name

if st.session_state.get("last_prediction_result") is not None:
    st.divider()
    render_prediction_result(
        st.session_state["last_prediction_result"],
        source_name=st.session_state.get("last_prediction_source"),
    )
    st.caption(f"Generated at {st.session_state['last_prediction_result'].published_at or 'now'}")
else:
    st.info("Select a ticker and click **Predict** to run the live pipeline.")
