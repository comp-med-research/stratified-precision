"""
Plotly Dash results dashboard — mounted at /dashboard/ on the shared Flask server.
The landing page lives in templates/index.html (plain Flask/HTML/JS).
"""

from __future__ import annotations

from flask import Flask
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
                    _card("Patient Endotype Map" if result.mode == "patient" else "Disease Endotype Map",
                          dcc.Graph(
                              figure=_umap_figure(result),
                              style={"height": "380px"},
                              config={"displayModeBar": False},
                          )),
                    _pareto_panel(result),
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


def _pareto_panel(result) -> html.Div:
    """Single Pareto chart for target mode; tabbed per-endotype for patient mode."""
    if result.mode != "patient" or not result.pareto_per_endotype:
        return _card("Dynamic Pareto Front", dcc.Graph(
            figure=_pareto_figure(result),
            style={"height": "380px"},
            config={"displayModeBar": False},
        ))

    # Build one tab per endotype
    tabs = []
    for eid in sorted(result.pareto_per_endotype.keys()):
        pareto   = result.pareto_per_endotype[eid]
        df_endo  = result.ranked_targets[result.ranked_targets["endotype_id"] == eid]
        label    = df_endo["endotype_label"].iloc[0] if not df_endo.empty else f"Endotype {eid + 1}"
        # Short label for the tab — just "E1", "E2" etc. to keep tabs compact
        tab_label = f"E{eid + 1}"
        tabs.append(dcc.Tab(
            label=tab_label,
            style={"fontSize": "12px", "padding": "6px 10px"},
            selected_style={"fontSize": "12px", "padding": "6px 10px",
                            "fontWeight": "600", "borderTop": "2px solid #5B8DEF"},
            children=[
                html.P(label,
                       style={"fontSize": "11px", "color": "#888", "margin": "8px 0 0 8px",
                               "fontWeight": "600", "textTransform": "uppercase",
                               "letterSpacing": "0.05em"}),
                dcc.Graph(
                    figure=_pareto_figure_for_endotype(df_endo, pareto),
                    style={"height": "330px"},
                    config={"displayModeBar": False},
                ),
            ],
        ))

    return html.Div(style={
        "background": "#fff", "borderRadius": "12px",
        "border": "1px solid #ebebeb", "padding": "20px",
    }, children=[
        html.P("Pareto Front per Endotype",
               style={"margin": "0 0 12px 0", "fontSize": "13px", "fontWeight": "600",
                      "color": "#555", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        dcc.Tabs(tabs, style={"borderBottom": "1px solid #eee"}),
    ])


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
    df     = result.ranked_targets.copy()
    ranks  = df["pareto_rank"].values if "pareto_rank" in df.columns else np.ones(len(df), dtype=int)
    return _pareto_scatter(df, ranks, result.pareto.objective_names)


def _pareto_figure_for_endotype(df_endo: pd.DataFrame, pareto: "ParetoResult") -> go.Figure:
    """Pareto scatter for a single endotype — uses stored obj_* columns."""
    ranks = (df_endo["endotype_pareto_rank"].values
             if "endotype_pareto_rank" in df_endo.columns
             else np.ones(len(df_endo), dtype=int))
    return _pareto_scatter(df_endo, ranks, pareto.objective_names)


_RANK_STYLE = {
    1: dict(size=14, symbol="diamond", opacity=1.0,  line_width=1.5, show_text=True),
    2: dict(size=10, symbol="circle",  opacity=0.75, line_width=1.0, show_text=False),
    3: dict(size=8,  symbol="circle",  opacity=0.50, line_width=0,   show_text=False),
}
_RANK_STYLE_DEFAULT = dict(size=6, symbol="circle", opacity=0.35, line_width=0, show_text=False)


def _pareto_scatter(df: pd.DataFrame, ranks: np.ndarray, obj_names: list[str]) -> go.Figure:
    """
    One trace per Pareto rank so users can toggle ranks on/off via the legend.
    Uses stored obj_* columns for x/y axes — safe after any sort/reindex.
    """
    x_col = f"obj_{obj_names[0]}" if obj_names else None
    y_col = f"obj_{obj_names[1]}" if len(obj_names) > 1 else None

    def _get_col(col):
        if col and col in df.columns:
            return df[col].fillna(0).values
        return np.zeros(len(df))

    xs     = _get_col(x_col)
    ys     = _get_col(y_col)
    genes  = df["gene_symbol"].values if "gene_symbol" in df.columns else [""] * len(df)
    modes  = (df["predicted_failure_mode"].fillna("unknown").values
              if "predicted_failure_mode" in df.columns else ["unknown"] * len(df))

    unique_ranks = sorted(set(ranks))
    fig = go.Figure()

    for rank in unique_ranks:
        mask   = ranks == rank
        style  = _RANK_STYLE.get(rank, _RANK_STYLE_DEFAULT)
        label  = f"Rank {rank}" + (" — Pareto front" if rank == 1 else "")

        r_genes = genes[mask]
        r_modes = modes[mask]
        r_xs    = xs[mask]
        r_ys    = ys[mask]
        colors  = [PALETTE.get(m, PALETTE["unknown"]) for m in r_modes]

        fig.add_trace(go.Scatter(
            x=r_xs,
            y=r_ys,
            mode="markers+text" if style["show_text"] else "markers",
            name=label,
            legendgroup=f"rank_{rank}",
            text=r_genes if style["show_text"] else [""] * len(r_genes),
            textposition="top center",
            textfont=dict(size=9, color="#333"),
            marker=dict(
                color=colors,
                size=style["size"],
                opacity=style["opacity"],
                line=dict(width=style["line_width"], color="#444"),
                symbol=style["symbol"],
            ),
            customdata=list(zip(r_genes, r_modes)),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                          "<br>x: %{x:.3f}  y: %{y:.3f}<extra></extra>",
        ))

    x_title = (obj_names[0] if obj_names else "").replace("_", " ").title()
    y_title = (obj_names[1] if len(obj_names) > 1 else "").replace("_", " ").title()
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=10),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=11),
    )
    return fig


def _target_table(result) -> dash_table.DataTable:
    df = result.ranked_targets.head(50).reset_index(drop=True)
    if result.mode == "patient":
        rank_cols = ["endotype_pareto_rank", "endotype_pareto_front"]
    else:
        rank_cols = ["pareto_rank", "pareto_front"]
    display_cols = [c for c in [
        "gene_symbol", "endotype_label", "predicted_failure_mode",
        "association_score", "safety_score", "efficacy_score",
        "novelty_score", *rank_cols,
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
            {"if": {"filter_query": '{endotype_pareto_front} = true'}, "fontWeight": "600"},
        ],
        sort_action="native", page_size=15,
    )
