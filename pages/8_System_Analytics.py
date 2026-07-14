"""System Analytics — database footprint, model quality metrics, and prediction health."""

from __future__ import annotations

import streamlit as st

from app.bootstrap import bootstrap_page
from app.components import charts
from app.components.metric_cards import metric_grid
from app.components.styles import empty_state, section_header
from app.services import database_service, prediction_service

bootstrap_page("System Analytics", icon="🖥️")

section_header("System Analytics", "Database footprint and model training-quality metrics.", icon="🖥️")

# ── Database statistics ─────────────────────────────────────────────────────────
section_header("Database Statistics", icon="🗄️")
metric_grid(
    [
        {"label": "Articles", "value": f"{database_service.count_articles():,}", "icon": "📰"},
        {"label": "Predictions", "value": f"{database_service.count_predictions():,}", "icon": "🎯"},
        {"label": "Sentiments", "value": f"{database_service.count_sentiment_results():,}", "icon": "🧠"},
        {"label": "Prices", "value": f"{database_service.count_prices():,}", "icon": "📈"},
    ]
)

st.divider()

# ── Model metrics ────────────────────────────────────────────────────────────────
section_header("Model Metrics", icon="🧪")
metrics = prediction_service.get_model_metrics()
model_status = prediction_service.model_file_status()

if metrics is None:
    empty_state("No metrics file found at `artifacts/metrics/xgboost_metrics.json`.")
else:
    metric_grid(
        [
            {"label": "Training Accuracy", "value": f"{metrics['accuracy']:.1%}", "icon": "🎯"},
            {"label": "Balanced Accuracy", "value": f"{metrics['balanced_accuracy']:.1%}", "icon": "⚖️"},
            {"label": "Macro F1", "value": f"{metrics['f1']['macro']:.1%}", "icon": "📐"},
            {"label": "Feature Count", "value": str(model_status.get("n_features", "N/A")), "icon": "🧮"},
        ]
    )
    st.write("")
    metric_grid(
        [
            {"label": "Macro Precision", "value": f"{metrics['precision']['macro']:.1%}", "icon": "🎯"},
            {"label": "Macro Recall", "value": f"{metrics['recall']['macro']:.1%}", "icon": "🔁"},
            {"label": "MCC", "value": f"{metrics.get('mcc', 0):.3f}", "icon": "🔗"},
            {"label": "Classes", "value": ", ".join(metrics.get("labels", [])), "icon": "🏷️"},
        ]
    )

    with st.expander("Per-class precision / recall / F1"):
        import pandas as pd

        per_class = pd.DataFrame(
            {
                "class": metrics["labels"],
                "precision": [metrics["precision"]["per_class"][c] for c in metrics["labels"]],
                "recall": [metrics["recall"]["per_class"][c] for c in metrics["labels"]],
                "f1": [metrics["f1"]["per_class"][c] for c in metrics["labels"]],
            }
        )
        st.dataframe(
            per_class.style.format({"precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"}),
            width="stretch", hide_index=True,
        )

    with st.expander("Confusion Matrix & Classification Report"):
        import numpy as np
        import pandas as pd

        cm = pd.DataFrame(
            np.array(metrics["confusion_matrix"]),
            index=[f"true_{c}" for c in metrics["labels"]],
            columns=[f"pred_{c}" for c in metrics["labels"]],
        )
        st.dataframe(cm, width="stretch")
        st.code(metrics.get("classification_report", ""), language="text")

st.divider()

# ── Prediction distribution / confidence ────────────────────────────────────────
section_header("Prediction Health", icon="📊")
all_preds = database_service.get_predictions_df()
if all_preds.empty:
    empty_state("No predictions saved yet.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prediction Distribution**")
        st.plotly_chart(charts.prediction_distribution_bar(all_preds), use_container_width=True)
    with col2:
        st.markdown("**Confidence Histogram**")
        st.plotly_chart(charts.confidence_histogram(all_preds["confidence"]), use_container_width=True)
