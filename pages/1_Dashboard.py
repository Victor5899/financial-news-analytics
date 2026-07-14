"""Dashboard — top-level KPIs, recent activity, and distribution charts."""

from __future__ import annotations

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components import charts
from app.components.metric_cards import metric_grid
from app.components.styles import (
    badge_html,
    empty_state,
    prediction_kind,
    section_header,
    sentiment_kind,
)
from app.components.tables import download_csv_button
from app.services import database_service, prediction_service

bootstrap_page("Dashboard", icon="📈")

section_header("Dashboard", "A real-time snapshot of the entire pipeline.", icon="📈")

db_status = database_service.database_status()
if not db_status["connected"]:
    st.error(f"⚠️ Database unavailable: {db_status['error']}")
    st.stop()

if st.button("🔄 Refresh data", help="Clear cached reads and re-query the database"):
    database_service.clear_all_caches()
    st.rerun()

# ── KPI row ──────────────────────────────────────────────────────────────────────
metrics = prediction_service.get_model_metrics()
accuracy_str = f"{metrics['accuracy']:.1%}" if metrics else "N/A"

metric_grid(
    [
        {"label": "Total Articles", "value": f"{database_service.count_articles():,}", "icon": "📰"},
        {"label": "Total Predictions", "value": f"{database_service.count_predictions():,}", "icon": "🎯"},
        {"label": "Tracked Tickers", "value": str(len(prediction_service.SUPPORTED_TICKERS)), "icon": "🏷️"},
        {"label": "Model Accuracy", "value": accuracy_str, "icon": "📐"},
    ]
)

st.write("")

# ── Latest snapshot row ───────────────────────────────────────────────────────────
latest_pred = database_service.get_latest_prediction()
latest_article = database_service.get_latest_article()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Latest Prediction**")
    if latest_pred:
        st.markdown(
            badge_html(f"{latest_pred['ticker']} · {latest_pred['prediction']}",
                       prediction_kind(latest_pred["prediction"])),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No predictions yet")
with col2:
    st.markdown("**Latest Prediction Confidence**")
    st.markdown(f"### {latest_pred['confidence']:.1%}" if latest_pred else "### —")
with col3:
    st.markdown("**Latest News Headline**")
    st.caption(latest_article["title"][:90] + "…" if latest_article and len(latest_article["title"]) > 90
               else (latest_article["title"] if latest_article else "No articles yet"))
with col4:
    st.markdown("**Latest Sentiment**")
    if latest_article and latest_article.get("sentiment_label"):
        st.markdown(
            badge_html(latest_article["sentiment_label"].title(), sentiment_kind(latest_article["sentiment_label"])),
            unsafe_allow_html=True,
        )
    else:
        st.caption("N/A")

st.divider()

# ── Recent headlines / predictions ────────────────────────────────────────────────
col_left, col_right = st.columns(2)
with col_left:
    section_header("Recent Headlines", icon="📰")
    articles_df = database_service.get_articles_df(limit=8)
    if articles_df.empty:
        empty_state("No news articles ingested yet.")
    else:
        for _, row in articles_df.iterrows():
            sent = row.get("sentiment_label")
            badge = badge_html(sent.title(), sentiment_kind(sent)) if sent else ""
            st.markdown(
                f"""<div class="fnx-news-card">
                    <div class="fnx-news-headline">{row['title']}</div>
                    <div class="fnx-news-meta">{row['ticker']} · {row['source_name'] or 'Unknown source'}
                    · {row['published_at']}</div>
                    {badge}
                </div>""",
                unsafe_allow_html=True,
            )

with col_right:
    section_header("Recent Predictions", icon="🎯")
    preds_df = database_service.get_predictions_df(limit=8)
    if preds_df.empty:
        empty_state("No predictions recorded yet — try the Live Prediction page.")
    else:
        for _, row in preds_df.iterrows():
            st.markdown(
                f"""<div class="fnx-news-card">
                    <div class="fnx-news-headline">{row['ticker']} — {row['prediction']}
                    {badge_html(f"{row['confidence']:.0%}", prediction_kind(row['prediction']))}</div>
                    <div class="fnx-news-meta">{row['headline'][:100]}</div>
                    <div class="fnx-news-meta">{row['created_at']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        download_csv_button(preds_df, "recent_predictions.csv")

st.divider()

# ── Distributions ────────────────────────────────────────────────────────────────
section_header("Distributions", icon="📊")
tab1, tab2, tab3 = st.tabs(["Ticker Distribution", "Prediction Distribution", "Confidence Distribution"])

with tab1:
    ticker_dist = database_service.get_ticker_distribution()
    if ticker_dist.empty:
        empty_state("No article data available.")
    else:
        st.plotly_chart(charts.ticker_distribution_pie(ticker_dist), use_container_width=True)

with tab2:
    all_preds = database_service.get_predictions_df()
    if all_preds.empty:
        empty_state("No predictions recorded yet.")
    else:
        st.plotly_chart(charts.prediction_distribution_bar(all_preds), use_container_width=True)

with tab3:
    all_preds = database_service.get_predictions_df()
    if all_preds.empty:
        empty_state("No predictions recorded yet.")
    else:
        st.plotly_chart(charts.confidence_histogram(all_preds["confidence"]), use_container_width=True)

st.divider()

# ── Activity timeline ─────────────────────────────────────────────────────────────
section_header("Recent Activity Timeline", icon="🕒")
activity = database_service.get_recent_activity(limit=30)
if activity.empty:
    empty_state("No recent activity to show.")
else:
    st.plotly_chart(charts.activity_timeline(activity), use_container_width=True)
