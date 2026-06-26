"""
Phase 7: Feature importance computation and visualisation for XGBoost models.

Extracts per-feature importance scores from a trained XGBClassifier and
produces both a ranked CSV and a horizontal bar-chart PNG.

Usage
-----
    from src.model.feature_importance import (
        compute_feature_importance,
        plot_feature_importance,
        save_importance_csv,
    )
    from pathlib import Path

    importance_df = compute_feature_importance(model, feature_columns)
    plot_feature_importance(importance_df, Path("artifacts/plots/feature_importance.png"))
    save_importance_csv(importance_df, Path("artifacts/plots/feature_importance.csv"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_TOP_N_DEFAULT = 20


# ── Computation ───────────────────────────────────────────────────────────────

def compute_feature_importance(
    model: Any,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Extract feature importances from a trained XGBClassifier.

    Uses the ``feature_importances_`` attribute (sklearn API), which reports
    *weight* — the number of times each feature is used to split a node
    across all trees.  Rows are sorted by descending importance.

    Parameters
    ----------
    model : XGBClassifier
        A fitted XGBoost classifier with ``feature_importances_`` available.
    feature_columns : list[str]
        Ordered list of feature names used during training.

    Returns
    -------
    pd.DataFrame
        Two columns: ``feature`` (str) and ``importance`` (float),
        sorted by ``importance`` descending.

    Raises
    ------
    ValueError
        If the number of importance scores does not match ``feature_columns``.
    """
    importances = model.feature_importances_
    if len(importances) != len(feature_columns):
        raise ValueError(
            f"Importance array length ({len(importances)}) does not match "
            f"feature_columns length ({len(feature_columns)})."
        )
    df = pd.DataFrame({"feature": feature_columns, "importance": importances.tolist()})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    logger.debug(f"Top feature: {df.iloc[0]['feature']} ({df.iloc[0]['importance']:.4f})")
    return df


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_feature_importance(
    importance_df: pd.DataFrame,
    out_path: Path,
    top_n: int = _TOP_N_DEFAULT,
) -> None:
    """Generate and save a horizontal bar chart of the top N features.

    Uses a non-interactive Matplotlib backend (``Agg``) so it runs safely
    in headless / server environments without a display.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of :func:`compute_feature_importance`.
    out_path : Path
        Destination PNG file (e.g. ``artifacts/plots/feature_importance.png``).
        Parent directories are created automatically.
    top_n : int
        Maximum number of features to include in the chart.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    top = importance_df.head(top_n).copy()
    top = top.sort_values("importance", ascending=True)  # ascending for barh readability

    fig_height = max(4, len(top) // 2 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(top["feature"], top["importance"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance (weight)", fontsize=11)
    ax.set_title(
        f"Top {len(top)} Feature Importances — XGBoost Direction Model",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved → {out_path}")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_importance_csv(importance_df: pd.DataFrame, out_path: Path) -> None:
    """Save the full feature importance table to a CSV file.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of :func:`compute_feature_importance`.
    out_path : Path
        Destination CSV file.  Parent directories are created automatically.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_path, index=False)
    logger.info(f"Feature importance CSV saved → {out_path}")
