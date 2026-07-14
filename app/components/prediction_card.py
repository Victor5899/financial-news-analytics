"""Renders a :class:`src.realtime.realtime_pipeline.PredictionResult` as a rich card."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.components import styles
from app.components.charts import confidence_gauge, probability_bar_chart
from src.realtime.realtime_pipeline import PredictionResult


def _fmt_time(dt: Any) -> str:
    if dt is None:
        return "Unknown"
    try:
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:  # noqa: BLE001
        return str(dt)


def render_prediction_result(
    result: PredictionResult,
    *,
    source_name: str | None = None,
    show_features: bool = True,
) -> None:
    """Full prediction card: headline, sentiment, prediction, probabilities, features."""
    pred_kind = styles.prediction_kind(result.prediction)
    sent_kind = styles.sentiment_kind(result.sentiment_label)

    top = st.container()
    with top:
        st.markdown(f'<div class="fnx-news-headline">{result.headline or "(no headline)"}</div>',
                    unsafe_allow_html=True)
        meta_bits = [f"🕒 {_fmt_time(result.published_at)}"]
        if source_name:
            meta_bits.append(f"📰 {source_name}")
        meta_bits.append(f"🏷️ {result.ticker}")
        st.markdown(f'<div class="fnx-news-meta">{" &nbsp;·&nbsp; ".join(meta_bits)}</div>',
                    unsafe_allow_html=True)

        badge_row = (
            styles.badge_html(f"Prediction: {result.prediction}", pred_kind)
            + "&nbsp;&nbsp;"
            + styles.badge_html(
                f"Sentiment: {result.sentiment_label.title()} ({result.sentiment_confidence:.0%})",
                sent_kind,
            )
        )
        st.markdown(badge_row, unsafe_allow_html=True)

    st.write("")
    col_gauge, col_probs = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(
            confidence_gauge(result.confidence, result.prediction),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(f"Model confidence in **{result.prediction}**")
    with col_probs:
        st.plotly_chart(
            probability_bar_chart(result.probabilities),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    if show_features and result.feature_values:
        with st.expander(f"🧮 Generated Features ({len(result.feature_values)} used by the model)"):
            technical_keys = {
                "sma_10", "sma_20", "ema_10", "ema_20", "rsi_14",
                "macd", "macd_signal", "macd_histogram",
                "bb_upper", "bb_lower", "bb_width", "atr_14", "volatility_20d",
            }
            technical = {k: v for k, v in result.feature_values.items() if k in technical_keys}
            other = {k: v for k, v in result.feature_values.items() if k not in technical_keys}

            if technical:
                st.markdown("**Technical Indicators**")
                st.dataframe(
                    {"feature": list(technical.keys()), "value": list(technical.values())},
                    width="stretch",
                    hide_index=True,
                )
            if other:
                st.markdown("**Sentiment / Source / Rolling Features**")
                st.dataframe(
                    {"feature": list(other.keys()), "value": list(other.values())},
                    width="stretch",
                    hide_index=True,
                )

    if result.saved_id:
        st.caption(f"💾 Saved to database — prediction id `{result.saved_id}`")
