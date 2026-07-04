"""
Plotly Dash visualisation dashboard.

Three panels:
  1. Endotype map       — UMAP scatter coloured by cluster + overlay target relevance
  2. Pareto front       — 2D projection of the 4-objective Pareto front
  3. Target scorecard   — ranked table with failure mode badges + literature signals
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, callback

from ..pipeline import PipelineResult


PALETTE = {
    "toxicity":       "#E05A5A",
    "efficacy":       "#F0A500",
    "likely_success": "#4CAF7D",
    "noise":          "#AAAAAA",
}

PARETO_FRONT_COLOR = "#5B8DEF"
DOMINATED_COLOR = "#CCCCCC"


def build_app(result: PipelineResult, port: int = 8050) -> Dash:
    app = Dash(__name__, title="Stratified Precision — Target Analysis")
    app.layout = _build_layout(result)
    _register_callbacks(app, result)
    return app


def run_dashboard(result: PipelineResult, port: int = 8050, debug: bool = True):
    app = build_app(result, port)
    app.run(debug=debug, port=port)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _build_layout(result: PipelineResult) -> html.Div:
    return html.Div(
        style={"fontFamily": "Inter, sans-serif", "maxWidth": "1400px", "margin": "auto", "padding": "24px"},
        children=[
            html.H1("Stratified Precision — Target Identification", style={"marginBottom": "4px"}),
            html.P(
                f"Mode: {'Target-first' if result.mode == 'target' else 'Patient-first'} | "
                f"{len(result.ranked_targets)} candidates | "
                f"{result.endotyping.n_clusters} endotypes",
                style={"color": "#666", "marginBottom": "32px"},
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px"},
                children=[
                    _card("Disease Endotype Map", dcc.Graph(
                        id="umap-plot",
                        figure=_umap_figure(result),
                        style={"height": "420px"},
                    )),
                    _card("Pareto Front", dcc.Graph(
                        id="pareto-plot",
                        figure=_pareto_figure(result),
                        style={"height": "420px"},
                    )),
                ],
            ),
            html.Div(style={"marginTop": "24px"}, children=[
                _card("Ranked Targets", _target_table(result)),
            ]),
        ],
    )


def _card(title: str, content) -> html.Div:
    return html.Div(
        style={
            "background": "#fff",
            "borderRadius": "12px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
            "padding": "20px",
        },
        children=[
            html.H3(title, style={"marginTop": 0, "marginBottom": "16px", "fontSize": "15px", "color": "#222"}),
            content,
        ],
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _umap_figure(result: PipelineResult) -> go.Figure:
    coords = result.endotyping.umap_coords
    labels = result.endotyping.labels.values

    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels.astype(str),
    })

    fig = px.scatter(
        df_plot, x="x", y="y", color="cluster",
        labels={"x": "UMAP 1", "y": "UMAP 2", "cluster": "Endotype"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Endotype")
    return fig


def _pareto_figure(result: PipelineResult) -> go.Figure:
    obj = result.pareto.objective_matrix
    on_front = result.pareto.on_front
    df = result.ranked_targets.copy().reset_index(drop=True)

    gene_labels = df["gene_symbol"].values if "gene_symbol" in df.columns else [""] * len(df)
    failure_modes = df["predicted_failure_mode"].fillna("unknown").values if "predicted_failure_mode" in df.columns else [""] * len(df)

    marker_colors = [
        PALETTE.get(fm, DOMINATED_COLOR) if front else DOMINATED_COLOR
        for fm, front in zip(failure_modes, on_front)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obj["efficacy_potential"],
        y=obj["safety_margin"],
        mode="markers+text",
        text=[g if f else "" for g, f in zip(gene_labels, on_front)],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            color=marker_colors,
            size=[10 if f else 6 for f in on_front],
            line=dict(width=[1.5 if f else 0 for f in on_front], color="#333"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Efficacy potential: %{x:.2f}<br>"
            "Safety margin: %{y:.2f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Efficacy potential",
        yaxis_title="Safety margin",
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def _target_table(result: PipelineResult):
    df = result.ranked_targets.head(20).copy().reset_index(drop=True)

    display_cols = [c for c in [
        "gene_symbol", "endotype_label", "association_score",
        "safety_score", "efficacy_score", "predicted_failure_mode",
        "novelty_score", "pareto_rank",
    ] if c in df.columns]

    df_display = df[display_cols].round(3)

    from dash import dash_table
    return dash_table.DataTable(
        data=df_display.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df_display.columns],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "padding": "8px 12px", "fontFamily": "Inter, sans-serif"},
        style_header={"fontWeight": "bold", "background": "#f5f5f5"},
        style_data_conditional=[
            {
                "if": {"filter_query": '{predicted_failure_mode} = "toxicity"'},
                "backgroundColor": "#fff0f0",
            },
            {
                "if": {"filter_query": '{predicted_failure_mode} = "efficacy"'},
                "backgroundColor": "#fffbf0",
            },
            {
                "if": {"filter_query": '{predicted_failure_mode} = "likely_success"'},
                "backgroundColor": "#f0fff5",
            },
        ],
        sort_action="native",
        page_size=15,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _register_callbacks(app: Dash, result: PipelineResult):
    # Placeholder — clicking a point on the UMAP could filter the Pareto plot
    pass
