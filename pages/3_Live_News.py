"""Live News — latest Finnhub headlines scored live with the FinBERT sentiment analyzer."""

from __future__ import annotations

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components.styles import badge_html, empty_state, section_header, sentiment_kind
from app.services import prediction_service
from src.realtime.finnhub_client import FinnhubError

settings = bootstrap_page("Live News", icon="🗞️")

section_header("Live News", "Latest company news fetched live from Finnhub.", icon="🗞️")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    tickers = st.multiselect(
        "Tickers",
        options=prediction_service.SUPPORTED_TICKERS,
        default=settings["watchlist_tickers"] or [settings["active_ticker"]],
    )
with col2:
    score_sentiment = st.toggle("Score sentiment (FinBERT)", value=True)
with col3:
    st.write("")
    refresh = st.button("🔄 Refresh", type="primary", width="stretch")

if refresh or "live_news_cache" not in st.session_state:
    st.session_state["live_news_cache"] = {}

if refresh:
    with st.spinner("Fetching latest news …"):
        all_articles = []
        for t in tickers:
            try:
                articles = prediction_service.fetch_latest_news(t)
            except FinnhubError as exc:
                st.error(f"{t}: {exc}")
                continue
            all_articles.extend(articles)

        all_articles.sort(key=lambda a: a.get("published_at") or 0, reverse=True)

        if score_sentiment and all_articles:
            with st.spinner("Scoring sentiment with FinBERT …"):
                all_articles = prediction_service.score_articles_sentiment(all_articles)

        st.session_state["live_news_cache"] = {"articles": all_articles, "tickers": tickers}

cache = st.session_state.get("live_news_cache", {})
articles = cache.get("articles", [])

if not articles:
    empty_state("Click **Refresh** to fetch the latest news for the selected tickers.", icon="🗞️")
else:
    st.caption(f"Showing {len(articles)} article(s) across {len(cache.get('tickers', []))} ticker(s).")
    for article in articles:
        sentiment_badge = ""
        if article.get("sentiment_label"):
            sentiment_badge = badge_html(
                f"{article['sentiment_label'].title()} ({article.get('sentiment_confidence', 0):.0%})",
                sentiment_kind(article["sentiment_label"]),
            )

        published = article.get("published_at")
        published_str = published.strftime("%Y-%m-%d %H:%M UTC") if published else "Unknown time"

        st.markdown(
            f"""
            <div class="fnx-news-card">
                <div class="fnx-news-headline">{article.get('title') or '(no title)'}</div>
                <div class="fnx-news-meta">
                    🏷️ {article.get('ticker', '—')} &nbsp;·&nbsp;
                    🕒 {published_str} &nbsp;·&nbsp;
                    📰 {article.get('source_name') or 'Unknown source'}
                </div>
                <div class="fnx-news-desc">{article.get('description') or '<i>No description available.</i>'}</div>
                {sentiment_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if article.get("url"):
            st.markdown(f"[Read full article ↗]({article['url']})")
        st.write("")
