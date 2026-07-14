"""KPI metric-card grid used on the Dashboard and System Analytics pages."""

from __future__ import annotations

from typing import Any

import streamlit as st


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_color: str = "inherit",
    icon: str = "",
) -> None:
    """Render a single styled metric card (use inside an ``st.columns`` slot)."""
    icon_prefix = f"{icon}  " if icon else ""
    delta_html = (
        f'<div class="fnx-card-delta" style="color:{delta_color};">{delta}</div>'
        if delta
        else ""
    )
    st.markdown(
        f"""
        <div class="fnx-card">
            <div class="fnx-card-label">{icon_prefix}{label}</div>
            <div class="fnx-card-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(items: list[dict[str, Any]], columns: int = 4) -> None:
    """
    Render a responsive grid of metric cards.

    Each item: ``{"label": str, "value": str, "delta": str | None,
    "delta_color": str, "icon": str}``.
    """
    cols = st.columns(columns)
    for i, item in enumerate(items):
        with cols[i % columns]:
            metric_card(
                label=item.get("label", ""),
                value=str(item.get("value", "—")),
                delta=item.get("delta"),
                delta_color=item.get("delta_color", "inherit"),
                icon=item.get("icon", ""),
            )
