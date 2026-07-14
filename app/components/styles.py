"""
Shared visual language for the dashboard: color palette, global CSS, and
small HTML helpers (badges, section headers) used by every page.

Colors are chosen to read clearly on both Streamlit's light and dark
themes (semi-transparent fills + a saturated border/text color rather than
solid light-mode-only backgrounds).
"""

from __future__ import annotations

import streamlit as st

PALETTE: dict[str, str] = {
    "buy": "#16a34a",
    "hold": "#d97706",
    "sell": "#dc2626",
    "positive": "#16a34a",
    "neutral": "#6b7280",
    "negative": "#dc2626",
    "primary": "#2563eb",
    "info": "#0891b2",
    "muted": "#6b7280",
}

_CSS = """
<style>
:root {
    --fnx-radius: 12px;
}

/* Tighter, more "product" feeling spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Metric cards */
.fnx-card {
    border-radius: var(--fnx-radius);
    padding: 1.1rem 1.3rem;
    border: 1px solid rgba(128, 128, 128, 0.22);
    background: rgba(128, 128, 128, 0.06);
    height: 100%;
}
.fnx-card .fnx-card-label {
    font-size: 0.80rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 0.35rem;
}
.fnx-card .fnx-card-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.15;
}
.fnx-card .fnx-card-delta {
    font-size: 0.85rem;
    margin-top: 0.25rem;
    opacity: 0.85;
}

/* Badges */
.fnx-badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    border: 1px solid currentColor;
    white-space: nowrap;
}

/* Section headers */
.fnx-section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
}
.fnx-section-subtitle {
    font-size: 0.92rem;
    opacity: 0.65;
    margin-bottom: 0.6rem;
}

/* News / prediction cards */
.fnx-news-card {
    border-radius: var(--fnx-radius);
    border: 1px solid rgba(128, 128, 128, 0.22);
    background: rgba(128, 128, 128, 0.05);
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
}
.fnx-news-headline {
    font-weight: 700;
    font-size: 1.02rem;
    margin-bottom: 0.3rem;
}
.fnx-news-meta {
    font-size: 0.80rem;
    opacity: 0.6;
    margin-bottom: 0.45rem;
}
.fnx-news-desc {
    font-size: 0.90rem;
    opacity: 0.85;
    margin-bottom: 0.55rem;
}

/* Empty state */
.fnx-empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    border-radius: var(--fnx-radius);
    border: 1px dashed rgba(128, 128, 128, 0.35);
    opacity: 0.75;
}
.fnx-empty-state .fnx-empty-icon {
    font-size: 2.2rem;
    margin-bottom: 0.4rem;
}

/* Hide default streamlit "Made with" footer clutter on wide dashboards */
footer {visibility: hidden;}
</style>
"""


def apply_global_styles() -> None:
    """Inject the shared CSS block once per session."""
    st.markdown(_CSS, unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    """Render a consistent section title used at the top of every page/tab."""
    prefix = f"{icon} " if icon else ""
    st.markdown(f'<div class="fnx-section-title">{prefix}{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="fnx-section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def badge_html(text: str, kind: str = "neutral") -> str:
    """Return an inline HTML badge (does not render — use with st.markdown)."""
    color = PALETTE.get(kind.lower(), PALETTE["neutral"])
    return f'<span class="fnx-badge" style="color:{color};background:{color}1a;">{text}</span>'


def render_badge(text: str, kind: str = "neutral") -> None:
    st.markdown(badge_html(text, kind), unsafe_allow_html=True)


def empty_state(message: str, icon: str = "\U0001F4ED") -> None:
    """Consistent empty-state placeholder for pages/widgets with no data yet."""
    st.markdown(
        f"""
        <div class="fnx-empty-state">
            <div class="fnx-empty-icon">{icon}</div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_kind(prediction: str) -> str:
    """Map a BUY/HOLD/SELL label to a badge/color 'kind'."""
    return {"BUY": "buy", "HOLD": "hold", "SELL": "sell"}.get(prediction.upper(), "neutral")


def sentiment_kind(label: str) -> str:
    """Map a positive/neutral/negative label to a badge/color 'kind'."""
    return {"positive": "positive", "neutral": "neutral", "negative": "negative"}.get(
        (label or "").lower(), "neutral"
    )
