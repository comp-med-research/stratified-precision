"""
Plotly Dash results dashboard — mounted at /dashboard/ on the shared Flask server.
The landing page lives in templates/index.html (plain Flask/HTML/JS).
"""

from __future__ import annotations

from flask import Flask, redirect
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table

PALETTE = {
    "toxicity":       "#E05A5A",
    "efficacy":       "#F0A500",
    "likely_success": "#4CAF7D",
    "unknown":        "#AAAAAA",
}


# ---------------------------------------------------------------------------
# Mount Dash on the Flask server
# ---------------------------------------------------------------------------

def create_dash_app(server: Flask) -> Dash:
    app = Dash(
        __name__,
        server=server,
        url_base_pathname="/dashboard/",
        title="Stratified Precision — Results",
        suppress_callback_exceptions=True,
    )
    app.layout = _shell_layout()
    _register_callbacks(app)
    return app


def _shell_layout() -> html.Div:
    """Shell that loads the result on page load via a callback."""
    return html.Div(
        id="dashboard-root",
        style={"fontFamily": "Inter, -apple-system, sans-serif",
               "minHeight": "100vh", "background": "#f8f9fa"},
        children=[
            dcc.Location(id="url", refresh=False),
            html.Div(id="dashboard-content"),
        ],
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _register_callbacks(app: Dash):

    @app.callback(
        Output("dashboard-content", "children"),
        Input("url", "pathname"),
    )
    def render_dashboard(pathname):
        from stratified_precision.cache import result_cache
        result = result_cache.get("latest")
        if result is None:
            return html.Div([
                html.P("No analysis results yet.",
                       style={"textAlign": "center", "marginTop": "80px",
                              "color": "#888", "fontSize": "16px"}),
                html.Div(html.A("← Back to search", href="/",
                                style={"color": "#5B8DEF"}),
                         style={"textAlign": "center", "marginTop": "16px"}),
            ])
        return _results_layout(result)


# ---------------------------------------------------------------------------
# Results layout
# ---------------------------------------------------------------------------

def _results_layout(result) -> html.Div:
    return html.Div(
        style={"maxWidth": "1400px", "margin": "auto", "padding": "32px 24px"},
        children=[
            # Top bar
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "justifyContent": "space-between", "marginBottom": "28px"},
                children=[
                    html.Div([
                        html.H2("Stratified Precision",
                                style={"margin": "0 0 4px 0", "fontSize": "20px",
                                       "fontWeight": "700", "color": "#111"}),
                        html.P(
                            f"{'Target-first' if result.mode == 'target' else 'Patient-first'} · "
                            f"{len(result.ranked_targets)} candidates · "
                            f"{result.endotyping.n_clusters} endotypes · "
                            f"{len(result.pareto.objective_names)} objectives",
                            style={"margin": 0, "color": "#888", "fontSize": "13px"},
                        ),
                    ]),
                    html.A("← New search", href="/",
                           style={
                               "background": "transparent",
                               "border": "1.5px solid #ddd",
                               "borderRadius": "8px", "padding": "8px 16px",
                               "fontSize": "13px", "color": "#555",
                               "textDecoration": "none",
                           }),
                ],
            ),

            # Objective pills
            _objective_pills(result),

            # Two-column: UMAP + Pareto
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                       "gap": "20px", "marginBottom": "20px"},
                children=[
                    _card("Disease Endotype Map", dcc.Graph(
                        figure=_umap_figure(result),
                        style={"height": "380px"},
                        config={"displayModeBar": False},
                    )),
                    _card("Dynamic Pareto Front", dcc.Graph(
                        figure=_pareto_figure(result),
                        style={"height": "380px"},
                        config={"displayModeBar": False},
                    )),
                ],
            ),

            # KG sources (if any)
            _kg_sources_section(result) if result.kg_sources else html.Div(),

            # Ranked table
            _card("Ranked Targets", _target_table(result)),
        ],
    )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _objective_pills(result) -> html.Div:
    base = {"efficacy_potential", "safety_margin", "tissue_specificity", "novelty"}
    pills = []
    for name in result.pareto.objective_names:
        if name in base:
            pills.append(_pill(name.replace("_", " ").title(), "#5B8DEF"))
        else:
            weight = result.objective_weights.get(name.replace("kg_", ""), 0)
            label  = name.replace("kg_", "").replace("_", " ")
            pills.append(_pill(f"KG: {label} ({weight:.2f})", "#4CAF7D"))
    return html.Div(pills,
                    style={"display": "flex", "flexWrap": "wrap",
                           "gap": "8px", "marginBottom": "20px"})


def _pill(label: str, color: str) -> html.Span:
    return html.Span(label, style={
        "background": color + "18", "color": color,
        "border": f"1px solid {color}44",
        "borderRadius": "12px", "padding": "3px 10px",
        "fontSize": "12px", "fontWeight": "500",
    })


def _card(title: str, content) -> html.Div:
    return html.Div(style={
        "background": "#fff", "borderRadius": "12px",
        "border": "1px solid #ebebeb", "padding": "20px",
    }, children=[
        html.P(title, style={"margin": "0 0 16px 0", "fontSize": "13px",
                              "fontWeight": "600", "color": "#555",
                              "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        content,
    ])


def _kg_sources_section(result) -> html.Div:
    return html.Div(style={"marginBottom": "20px"}, children=[
        html.P("Knowledge sources selected for this disease context",
               style={"fontSize": "12px", "color": "#888", "marginBottom": "8px",
                      "fontWeight": "600", "textTransform": "uppercase",
                      "letterSpacing": "0.05em"}),
        html.Div([
            html.Div(style={
                "background": "#fff", "border": "1px solid #eee",
                "borderRadius": "8px", "padding": "10px 14px", "fontSize": "13px",
            }, children=[
                html.Span(s.name, style={"fontWeight": "600", "color": "#111"}),
                html.Span(f" · {s.rationale}", style={"color": "#666"}),
            ])
            for s in result.kg_sources
        ], style={"display": "flex", "flexDirection": "column", "gap": "6px"}),
    ])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _umap_figure(result) -> go.Figure:
    coords = result.endotyping.umap_coords
    labels = result.endotyping.labels.values
    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "Endotype": [f"Subgroup {l}" if l >= 0 else "Noise" for l in labels],
    })
    fig = px.scatter(df, x="x", y="y", color="Endotype",
                     labels={"x": "UMAP 1", "y": "UMAP 2"},
                     template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                      plot_bgcolor="#fff", paper_bgcolor="#fff")
    return fig


def _pareto_figure(result) -> go.Figure:
    obj    = result.pareto.objective_matrix
    front  = result.pareto.on_front
    df     = result.ranked_targets.reset_index(drop=True)
    genes  = df["gene_symbol"].values if "gene_symbol" in df.columns else [""] * len(df)
    modes  = df["predicted_failure_mode"].fillna("unknown").values if "predicted_failure_mode" in df.columns else ["unknown"] * len(df)
    x_col  = obj.columns[0] if len(obj.columns) > 0 else None
    y_col  = obj.columns[1] if len(obj.columns) > 1 else None

    fig = go.Figure()
    for i, (gene, mode, on_f) in enumerate(zip(genes, modes, front)):
        color = PALETTE.get(mode, PALETTE["unknown"])
        fig.add_trace(go.Scatter(
            x=[obj[x_col].iloc[i]] if x_col else [0],
            y=[obj[y_col].iloc[i]] if y_col else [0],
            mode="markers+text" if on_f else "markers",
            text=[gene] if on_f else [""],
            textposition="top center",
            textfont=dict(size=9, color="#333"),
            marker=dict(color=color if on_f else "#ddd",
                        size=12 if on_f else 6,
                        line=dict(width=1.5 if on_f else 0, color="#333"),
                        symbol="diamond" if on_f else "circle"),
            showlegend=False,
            hovertemplate=f"<b>{gene}</b><br>{mode}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>",
        ))
    for mode, color in PALETTE.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(color=color, size=8),
                                 name=mode.replace("_", " ").title(), showlegend=True))
    fig.update_layout(
        xaxis_title=(x_col or "").replace("_", " ").title(),
        yaxis_title=(y_col or "").replace("_", " ").title(),
        template="plotly_white", margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
    )
    return fig


def _target_table(result) -> dash_table.DataTable:
    df = result.ranked_targets.head(30).reset_index(drop=True)
    display_cols = [c for c in [
        "gene_symbol", "endotype_label", "predicted_failure_mode",
        "association_score", "safety_score", "efficacy_score",
        "novelty_score", "pareto_rank", "pareto_front",
    ] if c in df.columns]
    df_display = df[display_cols].round(3)
    return dash_table.DataTable(
        data=df_display.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df_display.columns],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "padding": "10px 14px",
                    "fontFamily": "Inter, sans-serif", "border": "1px solid #f0f0f0"},
        style_header={"fontWeight": "600", "background": "#fafafa", "color": "#555",
                      "border": "1px solid #ebebeb", "fontSize": "12px",
                      "textTransform": "uppercase", "letterSpacing": "0.03em"},
        style_data_conditional=[
            {"if": {"filter_query": '{predicted_failure_mode} = "toxicity"'},
             "backgroundColor": "#fff5f5"},
            {"if": {"filter_query": '{predicted_failure_mode} = "efficacy"'},
             "backgroundColor": "#fffdf0"},
            {"if": {"filter_query": '{predicted_failure_mode} = "likely_success"'},
             "backgroundColor": "#f3fff7"},
            {"if": {"filter_query": '{pareto_front} = true'}, "fontWeight": "600"},
        ],
        sort_action="native", page_size=15,
    )
