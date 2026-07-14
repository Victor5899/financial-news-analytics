"""Table rendering helpers: sortable grid + CSV download, with graceful fallback."""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

    _AGGRID_AVAILABLE = True
except Exception:  # noqa: BLE001
    _AGGRID_AVAILABLE = False


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV") -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width="content",
    )


def render_table(
    df: pd.DataFrame,
    *,
    height: int = 420,
    use_aggrid: bool = True,
    key: str | None = None,
) -> None:
    """
    Render a sortable/filterable table.

    Uses AgGrid when available for a richer commercial-dashboard feel;
    falls back to Streamlit's native (also sortable) dataframe otherwise.
    """
    if df.empty:
        from app.components.styles import empty_state  # noqa: PLC0415

        empty_state("No rows match the current filters.")
        return

    if use_aggrid and _AGGRID_AVAILABLE:
        builder = GridOptionsBuilder.from_dataframe(df)
        builder.configure_default_column(sortable=True, filterable=True, resizable=True)
        builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        grid_options = builder.build()
        AgGrid(
            df,
            gridOptions=grid_options,
            height=height,
            update_mode=GridUpdateMode.NO_UPDATE,
            fit_columns_on_grid_load=True,
            theme="balham",
            key=key,
        )
    else:
        st.dataframe(df, width="stretch", height=height, hide_index=True)
