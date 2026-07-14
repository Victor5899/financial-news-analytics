"""Model Explainability — SHAP-powered insight into the trained XGBoost direction model."""

from __future__ import annotations

import numpy as np
import streamlit as st

from app.bootstrap import bootstrap_page
from app.components import charts
from app.components.styles import badge_html, empty_state, prediction_kind, section_header
from app.services import analytics_service, prediction_service
from src.realtime.finnhub_client import FinnhubError
from src.realtime.realtime_pipeline import NoNewsError, RealtimePipelineError

bootstrap_page("Model Explainability", icon="🔍")

section_header(
    "Model Explainability",
    "SHAP (SHapley Additive exPlanations) applied to the trained XGBoost direction model — "
    "no retraining, no model changes.",
    icon="🔍",
)

artifact = prediction_service.get_model_artifact()
feature_columns: list[str] = list(artifact["feature_columns"])
classes: list[str] = list(artifact["label_encoder"].classes_)

tab_global, tab_instance = st.tabs(["🌐 Global Model Insights", "🎯 Explain a Prediction"])

# ── Global insights ──────────────────────────────────────────────────────────────
with tab_global:
    st.markdown("#### Top Feature Importance")
    st.caption("XGBoost's built-in gain-based feature importance (no SHAP needed).")
    importance_df = analytics_service.get_builtin_feature_importance()
    st.plotly_chart(charts.feature_importance_bar(importance_df, top_n=15), use_container_width=True)

    st.divider()
    st.markdown("#### SHAP Bar Chart (global, per class)")
    with st.spinner("Computing SHAP values over a sample of historical feature rows …"):
        global_shap = analytics_service.compute_global_shap(sample_size=200)

    if global_shap is None:
        st.info(
            "No local `data/ml/*.csv` feature dataset was found, so a global SHAP sample "
            "can't be computed here. Instance-level explanations (below) still work "
            "since they only need the model + one feature vector."
        )
    else:
        class_tab = st.radio("Class", options=global_shap["classes"], horizontal=True, key="global_shap_class")
        class_idx = global_shap["classes"].index(class_tab)
        mean_abs = global_shap["mean_abs_per_class"][:, class_idx]
        st.plotly_chart(
            charts.shap_class_bar(global_shap["feature_columns"], mean_abs, top_n=15),
            use_container_width=True,
        )
        st.caption(f"Mean |SHAP value| per feature for the **{class_tab}** class, "
                   f"over {len(global_shap['sample'])} sampled historical rows.")

# ── Instance-level explanation ────────────────────────────────────────────────────
with tab_instance:
    st.markdown("#### Pick a prediction to explain")
    st.caption(
        "SHAP needs the exact feature vector behind a prediction. Historical predictions in "
        "PostgreSQL only store the outcome, not the feature vector — so explanations run on the "
        "most recent live prediction from this session, or a fresh one you generate below."
    )

    last_result = st.session_state.get("last_prediction_result")

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Ticker", options=prediction_service.SUPPORTED_TICKERS, key="shap_ticker")
    with col2:
        st.write("")
        run_new = st.button("⚡ Run live prediction to explain", width="stretch")

    if run_new:
        with st.spinner(f"Running live pipeline for {ticker} …"):
            try:
                last_result = prediction_service.run_live_prediction(ticker, save=False)
                st.session_state["last_prediction_result"] = last_result
                st.session_state["last_prediction_source"] = None
            except NoNewsError:
                st.warning(f"No recent news found for {ticker}.")
                last_result = None
            except (FinnhubError, RealtimePipelineError) as exc:
                st.error(str(exc))
                last_result = None

    if last_result is None or not last_result.feature_values:
        empty_state("No prediction available yet. Run one above, or visit **Live Prediction** first.")
    else:
        predicted_class = last_result.prediction
        st.markdown(
            f"Explaining: **{last_result.ticker}** — "
            + badge_html(predicted_class, prediction_kind(predicted_class))
            + f" ({last_result.confidence:.1%} confidence)",
            unsafe_allow_html=True,
        )
        st.caption(f"Headline: _{last_result.headline}_")

        with st.spinner("Computing SHAP values for this prediction …"):
            explanation = analytics_service.explain_instance(last_result.feature_values)

        class_idx = classes.index(predicted_class) if predicted_class in classes else 0
        instance_exp = explanation[0, :, class_idx]

        col_wf, col_bar = st.columns(2)
        with col_wf:
            st.markdown("**SHAP Waterfall**")
            try:
                import matplotlib.pyplot as plt
                import shap

                fig = plt.figure()
                shap.plots.waterfall(instance_exp, max_display=12, show=False)
                st.pyplot(fig, bbox_inches="tight", clear_figure=True)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not render waterfall plot: {exc}")

        with col_bar:
            st.markdown("**SHAP Bar Chart (this prediction)**")
            values = np.asarray(instance_exp.values)
            st.plotly_chart(
                charts.shap_class_bar(feature_columns, values, top_n=12),
                use_container_width=True,
            )

        st.divider()
        st.markdown("#### Plain-English Explanation")
        bullets = analytics_service.build_plain_english_explanation(
            last_result.feature_values,
            np.asarray(instance_exp.values),
            feature_columns,
            predicted_class,
            top_n=5,
        )
        for b in bullets:
            st.markdown(f"- {b}")

        st.divider()
        st.markdown("#### Feature Contribution Table")
        import pandas as pd

        contrib_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "value": [last_result.feature_values.get(c) for c in feature_columns],
                "shap_contribution": values,
            }
        )
        contrib_df["abs_contribution"] = contrib_df["shap_contribution"].abs()
        contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).drop(columns="abs_contribution")
        st.dataframe(contrib_df, width="stretch", hide_index=True)
