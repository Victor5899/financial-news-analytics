"""Plotly chart builders shared across pages. Pure functions: DataFrame/dict in, ``go.Figure`` out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.styles import PALETTE

_LAYOUT_DEFAULTS = dict(
    margin=dict(l=10, r=10, t=40, b=10),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def _apply_defaults(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(**_LAYOUT_DEFAULTS, height=height)
    if title:
        fig.update_layout(title=dict(text=title, x=0.0, xanchor="left"))
    return fig


# ── Prediction / probability ────────────────────────────────────────────────────

def probability_bar_chart(probabilities: dict[str, float]) -> go.Figure:
    order = ["BUY", "HOLD", "SELL"]
    labels = [k for k in order if k in probabilities] or list(probabilities.keys())
    values = [probabilities[k] * 100 for k in labels]
    colors = [PALETTE.get(k.lower(), PALETTE["neutral"]) for k in labels]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Probability (%)", range=[0, 100])
    return _apply_defaults(fig, height=320)


def confidence_gauge(confidence: float, prediction: str) -> go.Figure:
    color = PALETTE.get(prediction.lower(), PALETTE["primary"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "rgba(128,128,128,0.15)"},
                    {"range": [40, 70], "color": "rgba(128,128,128,0.25)"},
                    {"range": [70, 100], "color": "rgba(128,128,128,0.35)"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    return _apply_defaults(fig, height=220)


# ── Dashboard distributions ─────────────────────────────────────────────────────

def ticker_distribution_pie(df: pd.DataFrame) -> go.Figure:
    fig = px.pie(df, names="ticker", values="article_count", hole=0.45)
    fig.update_traces(textinfo="label+percent")
    return _apply_defaults(fig, height=340)


def prediction_distribution_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["prediction"].value_counts().reindex(["BUY", "HOLD", "SELL"]).fillna(0)
    colors = [PALETTE[k.lower()] for k in counts.index]
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=colors))
    fig.update_yaxes(title="Count")
    return _apply_defaults(fig, height=340)


def confidence_histogram(series: pd.Series, nbins: int = 20) -> go.Figure:
    fig = px.histogram(series, nbins=nbins, color_discrete_sequence=[PALETTE["primary"]])
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Confidence")
    fig.update_yaxes(title="Count")
    return _apply_defaults(fig, height=320)


def activity_timeline(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="timestamp",
        y="ticker",
        color="event_type",
        hover_data=["detail"],
        color_discrete_map={"News Article": PALETTE["info"], "Prediction": PALETTE["primary"]},
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    return _apply_defaults(fig, height=360)


# ── Technical analysis ──────────────────────────────────────────────────────────

def price_candlestick(df: pd.DataFrame, overlays: list[str] | None = None) -> go.Figure:
    fig = go.Figure(
        go.Candlestick(
            x=df["trading_date"],
            open=df["open_price"],
            high=df["high_price"],
            low=df["low_price"],
            close=df["close_price"],
            name="Price",
            increasing_line_color=PALETTE["buy"],
            decreasing_line_color=PALETTE["sell"],
        )
    )
    overlay_colors = ["#2563eb", "#7c3aed", "#0891b2", "#d97706"]
    for i, col in enumerate(overlays or []):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["trading_date"], y=df[col], mode="lines", name=col.upper(),
                    line=dict(width=1.6, color=overlay_colors[i % len(overlay_colors)]),
                )
            )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_defaults(fig, height=440)


def line_chart(df: pd.DataFrame, x: str, cols: list[str], title: str | None = None) -> go.Figure:
    fig = go.Figure()
    for col in cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col.upper()))
    return _apply_defaults(fig, title=title, height=320)


def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trading_date"], y=df["macd"], mode="lines", name="MACD",
                              line=dict(color=PALETTE["primary"])))
    fig.add_trace(go.Scatter(x=df["trading_date"], y=df["macd_signal"], mode="lines", name="Signal",
                              line=dict(color=PALETTE["hold"])))
    hist_colors = [PALETTE["buy"] if v >= 0 else PALETTE["sell"] for v in df["macd_histogram"].fillna(0)]
    fig.add_trace(go.Bar(x=df["trading_date"], y=df["macd_histogram"], name="Histogram",
                          marker_color=hist_colors, opacity=0.5))
    return _apply_defaults(fig, title="MACD (12, 26, 9)", height=320)


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trading_date"], y=df["rsi_14"], mode="lines", name="RSI(14)",
                              line=dict(color=PALETTE["primary"])))
    fig.add_hline(y=70, line_dash="dash", line_color=PALETTE["sell"], opacity=0.6)
    fig.add_hline(y=30, line_dash="dash", line_color=PALETTE["buy"], opacity=0.6)
    fig.update_yaxes(range=[0, 100])
    return _apply_defaults(fig, title="RSI (14)", height=300)


def bollinger_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trading_date"], y=df["bb_upper"], mode="lines", name="Upper Band",
                              line=dict(width=1, color=PALETTE["muted"])))
    fig.add_trace(go.Scatter(x=df["trading_date"], y=df["bb_lower"], mode="lines", name="Lower Band",
                              line=dict(width=1, color=PALETTE["muted"]), fill="tonexty",
                              fillcolor="rgba(128,128,128,0.12)"))
    fig.add_trace(go.Scatter(x=df["trading_date"], y=df["close_price"], mode="lines", name="Close",
                              line=dict(width=1.8, color=PALETTE["primary"])))
    return _apply_defaults(fig, title="Bollinger Bands (20, 2σ)", height=340)


def atr_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df["trading_date"], y=df["atr_14"], mode="lines", name="ATR(14)",
                                line=dict(color=PALETTE["info"]), fill="tozeroy",
                                fillcolor="rgba(8,145,178,0.12)"))
    return _apply_defaults(fig, title="Average True Range (14)", height=280)


def volume_bar_chart(df: pd.DataFrame) -> go.Figure:
    colors = [
        PALETTE["buy"] if c >= o else PALETTE["sell"]
        for c, o in zip(df["close_price"].fillna(0), df["open_price"].fillna(0))
    ]
    fig = go.Figure(go.Bar(x=df["trading_date"], y=df["volume"], marker_color=colors, name="Volume"))
    if "volume_avg_5d" in df.columns:
        fig.add_trace(go.Scatter(x=df["trading_date"], y=df["volume_avg_5d"], mode="lines",
                                  name="5D Avg", line=dict(color=PALETTE["hold"], width=1.6)))
    return _apply_defaults(fig, title="Volume", height=280)


# ── Sentiment analytics ─────────────────────────────────────────────────────────

def sentiment_trend_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["mean_sentiment_score"], mode="lines+markers",
                              name="Daily Mean", line=dict(color=PALETTE["muted"], width=1),
                              marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df["day"], y=df["rolling_7d_mean"], mode="lines", name="7D Rolling Avg",
                              line=dict(color=PALETTE["primary"], width=2.4)))
    fig.add_hline(y=0, line_dash="dot", opacity=0.4)
    return _apply_defaults(fig, title="Daily Sentiment Trend", height=340)


def sentiment_share_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["positive"], mode="lines", stackgroup="one",
                              name="Positive", line=dict(color=PALETTE["positive"])))
    fig.add_trace(go.Scatter(x=df["day"], y=df["neutral"], mode="lines", stackgroup="one",
                              name="Neutral", line=dict(color=PALETTE["neutral"])))
    fig.add_trace(go.Scatter(x=df["day"], y=df["negative"], mode="lines", stackgroup="one",
                              name="Negative", line=dict(color=PALETTE["negative"])))
    return _apply_defaults(fig, title="Positive vs Neutral vs Negative", height=340)


def sentiment_timeline_scatter(df: pd.DataFrame) -> go.Figure:
    color_map = {"positive": PALETTE["positive"], "neutral": PALETTE["neutral"], "negative": PALETTE["negative"]}
    fig = px.scatter(
        df, x="published_at", y="sentiment_score", color="sentiment_label",
        color_discrete_map=color_map, hover_data=["ticker", "title"] if "title" in df.columns else None,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    return _apply_defaults(fig, title="Sentiment Timeline", height=340)


def ticker_sentiment_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["ticker"], y=df["positive"], name="Positive", marker_color=PALETTE["positive"]))
    fig.add_trace(go.Bar(x=df["ticker"], y=df["neutral"], name="Neutral", marker_color=PALETTE["neutral"]))
    fig.add_trace(go.Bar(x=df["ticker"], y=df["negative"], name="Negative", marker_color=PALETTE["negative"]))
    fig.update_layout(barmode="stack")
    return _apply_defaults(fig, title="Ticker-wise Sentiment", height=360)


# ── Historical predictions ───────────────────────────────────────────────────────

def predictions_confidence_line(df: pd.DataFrame) -> go.Figure:
    """Confidence over time, one line per ticker."""
    fig = go.Figure()
    for ticker, group in df.sort_values("created_at").groupby("ticker"):
        fig.add_trace(go.Scatter(x=group["created_at"], y=group["confidence"], mode="lines+markers",
                                  name=ticker, marker=dict(size=5)))
    fig.update_yaxes(title="Confidence", range=[0, 1])
    return _apply_defaults(fig, title="Confidence Over Time", height=340)


def predictions_volume_line(df: pd.DataFrame) -> go.Figure:
    """Daily prediction counts by direction."""
    daily = df.copy()
    daily["day"] = pd.to_datetime(daily["created_at"], utc=True, errors="coerce").dt.date
    pivot = daily.groupby(["day", "prediction"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for label in ["BUY", "HOLD", "SELL"]:
        if label in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[label], mode="lines", name=label,
                                      line=dict(color=PALETTE[label.lower()])))
    return _apply_defaults(fig, title="Predictions Per Day", height=320)


# ── Model explainability ────────────────────────────────────────────────────────

def feature_importance_bar(df: pd.DataFrame, top_n: int = 15, value_col: str = "importance") -> go.Figure:
    top = df.head(top_n).sort_values(value_col)
    fig = go.Figure(go.Bar(x=top[value_col], y=top["feature"], orientation="h",
                            marker_color=PALETTE["primary"]))
    return _apply_defaults(fig, height=max(320, 24 * top_n))


def shap_class_bar(feature_names: list[str], values: np.ndarray, top_n: int = 15) -> go.Figure:
    """Horizontal SHAP value bar for one class, colored by push direction."""
    df = pd.DataFrame({"feature": feature_names, "shap": values})
    df["abs_shap"] = df["shap"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).sort_values("shap")
    colors = [PALETTE["buy"] if v >= 0 else PALETTE["sell"] for v in df["shap"]]
    fig = go.Figure(go.Bar(x=df["shap"], y=df["feature"], orientation="h", marker_color=colors))
    fig.add_vline(x=0, line_color=PALETTE["muted"], opacity=0.5)
    return _apply_defaults(fig, height=max(320, 24 * len(df)))
