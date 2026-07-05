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
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
import dash_cytoscape as cyto

try:
    cyto.load_extra_layouts()
except Exception:
    pass

def _bubble_style(role: str) -> dict:
    base = {
        "maxWidth": "90%", "padding": "10px 14px", "borderRadius": "12px",
        "fontSize": "13px", "lineHeight": "1.65", "wordBreak": "break-word",
    }
    if role == "user":
        return {**base, "background": "#5B8DEF", "color": "#fff",
                "alignSelf": "flex-end", "borderRadius": "12px 12px 4px 12px"}
    return {**base, "background": "#f4f6fa", "color": "#333",
            "alignSelf": "flex-start", "borderRadius": "12px 12px 12px 4px"}


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
            dcc.Store(id="selected-target-store"),
            dcc.Store(id="result-ready-store"),
            dcc.Store(id="chat-open-store", data=False),
            dcc.Store(id="chat-history-store", data=[]),
            html.Div(id="dashboard-content"),
            # Floating chat toggle button (always visible, fixed bottom-right)
            html.Button("💬", id="chat-toggle-btn", n_clicks=0,
                        title="Open AI Co-Scientist chat",
                        style={
                            "position": "fixed", "bottom": "28px", "right": "28px",
                            "width": "52px", "height": "52px", "borderRadius": "50%",
                            "background": "#5B8DEF", "border": "none", "cursor": "pointer",
                            "zIndex": "10000", "fontSize": "22px", "color": "#fff",
                            "boxShadow": "0 4px 16px rgba(91,141,239,0.45)",
                            "display": "flex", "alignItems": "center",
                            "justifyContent": "center",
                        }),
            # Chat panel — hidden by default, toggled via callback
            html.Div(id="chat-panel",
                     style={
                         "position": "fixed", "top": "0", "right": "0",
                         "width": "420px", "height": "100vh", "background": "#fff",
                         "boxShadow": "-4px 0 32px rgba(0,0,0,0.15)",
                         "zIndex": "9999", "display": "none", "flexDirection": "column",
                         "borderLeft": "1px solid #ebebeb",
                         "fontFamily": "Inter, -apple-system, sans-serif",
                     },
                     children=[
                html.Div(style={
                    "padding": "16px 20px", "borderBottom": "1px solid #f0f0f0",
                    "display": "flex", "justifyContent": "space-between",
                    "alignItems": "center", "flexShrink": "0",
                }, children=[
                    html.Div([
                        html.Span("🤖", style={"fontSize": "17px", "marginRight": "8px"}),
                        html.Span("AI Co-Scientist",
                                  style={"fontWeight": "600", "fontSize": "15px",
                                         "color": "#111"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Button("✕", id="chat-close-btn", n_clicks=0, style={
                        "background": "none", "border": "none", "cursor": "pointer",
                        "fontSize": "18px", "color": "#999", "padding": "2px 6px",
                        "lineHeight": "1",
                    }),
                ]),
                dcc.Loading(type="dot", color="#5B8DEF",
                            children=html.Div(
                                id="chat-messages",
                                style={
                                    "flex": "1", "overflowY": "auto",
                                    "padding": "14px 16px",
                                    "display": "flex", "flexDirection": "column", "gap": "10px",
                                },
                                children=[
                                    html.Div(
                                        "Ask me anything about the analysis — targets, "
                                        "endotypes, failure modes, or what to do next.",
                                        style=_bubble_style("assistant"),
                                    )
                                ])),
                html.Div(style={
                    "padding": "12px 14px", "borderTop": "1px solid #f0f0f0",
                    "display": "flex", "gap": "8px",
                    "background": "#fff", "flexShrink": "0",
                }, children=[
                    dcc.Input(
                        id="chat-input", type="text",
                        placeholder="Ask about the results…",
                        debounce=False, n_submit=0,
                        style={
                            "flex": "1", "border": "1px solid #e0e0e0",
                            "borderRadius": "8px", "padding": "10px 14px",
                            "fontSize": "13px", "fontFamily": "Inter, sans-serif",
                            "outline": "none",
                        },
                    ),
                    html.Button("→", id="chat-send-btn", n_clicks=0, style={
                        "background": "#5B8DEF", "color": "#fff",
                        "border": "none", "borderRadius": "8px",
                        "padding": "10px 16px", "cursor": "pointer",
                        "fontSize": "16px", "fontWeight": "700",
                    }),
                ]),
            ]),
        ],
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _register_callbacks(app: Dash):

    # ── NL summary ─────────────────────────────────────────────────────
    @app.callback(
        Output("nl-summary", "children"),
        Input("result-ready-store", "data"),
        prevent_initial_call=True,
    )
    def generate_nl_summary(ready):
        if not ready:
            return no_update
        from stratified_precision.cache import result_cache
        result = result_cache.get("latest")
        if result is None:
            return no_update
        cached = getattr(result, "nl_summary", None)
        if not cached:
            cached = _generate_result_summary(result)
            try:
                result.nl_summary = cached
                ck = getattr(result, "_cache_key", None)
                if ck:
                    from stratified_precision.result_store import save_result
                    save_result(ck, result)
            except Exception:
                pass
        return html.Div(
            style={
                "background": "#fff", "border": "1px solid #ebebeb",
                "borderRadius": "12px", "padding": "20px 24px",
                "marginBottom": "20px",
            },
            children=[
                html.P("Summary", style={
                    "margin": "0 0 12px 0", "fontSize": "13px", "fontWeight": "600",
                    "color": "#555", "textTransform": "uppercase", "letterSpacing": "0.05em",
                }),
                dcc.Markdown(cached,
                             style={"fontSize": "14px", "lineHeight": "1.8",
                                    "color": "#333", "margin": 0}),
            ],
        )

    # ── Chat: toggle open/closed ───────────────────────────────────────
    _PANEL_BASE = {
        "position": "fixed", "top": "0", "right": "0",
        "width": "420px", "height": "100vh", "background": "#fff",
        "boxShadow": "-4px 0 32px rgba(0,0,0,0.15)", "zIndex": "9999",
        "flexDirection": "column", "borderLeft": "1px solid #ebebeb",
        "fontFamily": "Inter, -apple-system, sans-serif",
    }

    @app.callback(
        Output("chat-panel", "style"),
        Output("chat-open-store", "data"),
        Input("chat-toggle-btn", "n_clicks"),
        Input("chat-close-btn", "n_clicks"),
        State("chat-open-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_chat_panel(n_open, n_close, is_open):
        from dash import ctx
        new_open = False if ctx.triggered_id == "chat-close-btn" else not bool(is_open)
        style = {**_PANEL_BASE, "display": "flex" if new_open else "none"}
        return style, new_open

    # ── Chat: send message → call Claude → update history ─────────────
    @app.callback(
        Output("chat-history-store", "data"),
        Output("chat-input", "value"),
        Input("chat-send-btn", "n_clicks"),
        Input("chat-input", "n_submit"),
        State("chat-input", "value"),
        State("chat-history-store", "data"),
        prevent_initial_call=True,
    )
    def send_chat_message(n_clicks, n_submit, message, history):
        if not message or not message.strip():
            return no_update, no_update
        history = list(history or [])
        history.append({"role": "user", "content": message.strip()})
        from stratified_precision.cache import result_cache
        result = result_cache.get("latest")
        system = _build_chat_system_prompt(result)
        reply  = _chat_claude(system, history)
        history.append({"role": "assistant", "content": reply})
        return history, ""

    # ── Chat: render message bubbles ───────────────────────────────────
    @app.callback(
        Output("chat-messages", "children"),
        Input("chat-history-store", "data"),
    )
    def render_chat_messages(history):
        greeting = html.Div(
            "Ask me anything about the analysis — targets, endotypes, "
            "failure modes, or what to do next.",
            style=_bubble_style("assistant"),
        )
        if not history:
            return [greeting]
        bubbles = [greeting]
        for msg in history:
            if msg["role"] == "user":
                bubbles.append(html.Div(msg["content"], style=_bubble_style("user")))
            else:
                bubbles.append(html.Div(
                    dcc.Markdown(msg["content"],
                                 style={"margin": 0, "fontSize": "13px",
                                        "lineHeight": "1.65"}),
                    style=_bubble_style("assistant"),
                ))
        return bubbles

    # Table row click → explanation (always fires; uses original data index)
    @app.callback(
        Output("selected-target-store", "data"),
        Input("ranked-table", "active_cell"),
        State("ranked-table", "data"),
        prevent_initial_call=True,
    )
    def update_selected_target_from_table(active_cell, table_data):
        if not active_cell or not table_data:
            return no_update
        row = table_data[active_cell["row"]]
        return {
            "gene_symbol":       row.get("gene_symbol", ""),
            "disease_name":      row.get("disease_name", row.get("endotype_label", "")),
            "pareto_rank":       row.get("pareto_rank", row.get("endotype_pareto_rank", 1)),
            "failure_mode":      row.get("predicted_failure_mode", "unknown"),
            "association_score": row.get("association_score", 0),
            "novelty_score":     row.get("novelty_score", 0),
            "endotype_label":    row.get("endotype_label", ""),
        }

    # Graph node tap → explanation (target mode only; allow_duplicate shares same output)
    @app.callback(
        Output("selected-target-store", "data", allow_duplicate=True),
        Input("target-graph", "tapNodeData"),
        prevent_initial_call=True,
    )
    def update_selected_target_from_graph(node_data):
        if node_data and node_data.get("type") == "target":
            return node_data
        return no_update

    @app.callback(
        Output("explanation-panel", "children"),
        Input("selected-target-store", "data"),
        prevent_initial_call=True,
    )
    def show_explanation(target_data):
        if not target_data:
            return no_update

        gene     = target_data.get("gene_symbol", "Unknown")
        disease  = target_data.get("disease_name", "Unknown disease")
        rank     = target_data.get("pareto_rank", "?")
        failure  = target_data.get("failure_mode", "unknown")
        assoc    = float(target_data.get("association_score", 0) or 0)
        novelty  = float(target_data.get("novelty_score", 0) or 0)
        endotype = target_data.get("endotype_label", "")
        failure_color = PALETTE.get(failure, PALETTE["unknown"])

        explanation_md = _generate_explanation(target_data)

        endo_line = ""
        if endotype and ":" in endotype:
            endo_line = f" · {endotype.split(':',1)[-1].strip()}"
        elif endotype:
            endo_line = f" · {endotype}"

        return html.Div(
            style={
                "background": "#fff",
                "border": "1.5px solid #ebebeb",
                "borderLeft": f"4px solid {failure_color}",
                "borderRadius": "12px",
                "padding": "24px",
                "marginTop": "20px",
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between",
                           "alignItems": "flex-start", "marginBottom": "14px"},
                    children=[
                        html.Div([
                            html.H3(gene, style={"margin": "0 0 4px 0", "fontSize": "22px",
                                                 "fontWeight": "700", "color": "#111"}),
                            html.P(f"{disease}{endo_line}",
                                   style={"margin": 0, "color": "#888", "fontSize": "13px"}),
                        ]),
                        html.Div(
                            style={"display": "flex", "gap": "8px",
                                   "flexWrap": "wrap", "justifyContent": "flex-end"},
                            children=[
                                _pill(f"Pareto Rank {rank}", "#5B8DEF"),
                                _pill(failure.replace("_", " ").title(), failure_color),
                                _pill(f"Assoc {assoc:.2f}", "#888"),
                                _pill(f"Novelty {novelty:.2f}",
                                      "#4CAF7D" if novelty > 0.7 else "#888"),
                            ],
                        ),
                    ],
                ),
                html.Hr(style={"border": "none", "borderTop": "1px solid #f0f0f0",
                               "margin": "0 0 14px 0"}),
                dcc.Markdown(explanation_md,
                             style={"fontSize": "14px", "lineHeight": "1.75",
                                    "color": "#333", "margin": 0}),
                html.P("Generated by Claude · grounded in OpenTargets evidence",
                       style={"fontSize": "11px", "color": "#bbb",
                              "margin": "12px 0 0 0", "textAlign": "right"}),
            ],
        )

    @app.callback(
        Output("dashboard-content", "children"),
        Output("result-ready-store", "data"),
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
            ]), no_update
        return _results_layout(result), {"ready": True}


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

            # AI narrative summary (populated by callback after page renders)
            dcc.Loading(type="dot", color="#5B8DEF",
                        children=html.Div(id="nl-summary")),

            # Competitive landscape graph (target mode only)
            _graph_section(result) if result.mode == "target" else html.Div(),

            # KG sources (if any)
            _kg_sources_section(result) if result.kg_sources else html.Div(),

            # Ranked targets table — click a row to trigger AI explanation
            _card("Ranked Targets — click a row for AI explanation", _target_table(result)),

            # AI co-scientist explanation panel
            dcc.Loading(
                id="explanation-loading",
                type="dot",
                color="#5B8DEF",
                children=html.Div(id="explanation-panel"),
            ),
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
        id="ranked-table",
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


# ---------------------------------------------------------------------------
# KG Subgraph — Dash Cytoscape
# ---------------------------------------------------------------------------

def _graph_section(result) -> html.Div:
    elements = _build_graph_elements(result)
    if result.mode == "target":
        title   = "Target Competitive Landscape — KG Subgraph"
        caption = ("Disease node (◉) at centre. Competing targets arranged by association strength. "
                   "◆ = Pareto-front (rank 1) · ● = other ranks. "
                   "Border colour: red = toxicity risk · orange = efficacy risk · green = likely success. "
                   "Click a node to trigger AI explanation.")
    else:
        title   = "Patient Endotype × Target Graph"
        caption = ("Endotype hubs (blue) connected to their top candidate targets. "
                   "◆ = Pareto-front within that endotype. "
                   "Click any target node for an AI causal explanation.")

    return html.Div(style={"marginBottom": "20px"}, children=[
        _card(title, html.Div([
            html.P(caption, style={
                "fontSize": "11px", "color": "#aaa",
                "margin": "0 0 10px 0", "lineHeight": "1.5",
            }),
            cyto.Cytoscape(
                id="target-graph",
                elements=elements,
                layout={
                    "name": "cose",
                    "animate": False,
                    "randomize": False,
                    "nodeRepulsion": 900000,
                    "idealEdgeLength": 130,
                    "componentSpacing": 80,
                    "nodeOverlap": 10,
                    "fit": True,
                    "padding": 30,
                },
                style={
                    "width": "100%", "height": "500px",
                    "background": "#fafbfc", "borderRadius": "8px",
                },
                stylesheet=_cyto_stylesheet(),
                userZoomingEnabled=True,
                userPanningEnabled=True,
                minZoom=0.25,
                maxZoom=4.0,
            ),
        ])),
    ])


def _build_graph_elements(result) -> list[dict]:
    df = result.ranked_targets
    if df is None or df.empty:
        return []

    elements: list[dict] = []

    if result.mode == "target":
        disease_name = (df["disease_name"].iloc[0]
                        if "disease_name" in df.columns else "Disease")
        short_d = (disease_name[:28] + "…") if len(disease_name) > 28 else disease_name

        elements.append({"data": {
            "id": "disease",
            "label": short_d,
            "type": "disease",
            "full_label": disease_name,
        }})

        for _, row in df.iterrows():
            rank  = int(row.get("pareto_rank", 99))
            rc    = f"rank{rank}" if rank <= 2 else "rank3plus"
            fail  = str(row.get("predicted_failure_mode", "unknown")).replace(" ", "_")
            nid   = str(row.get("ensembl_id", row.get("gene_symbol", "")))

            elements.append({"data": {
                "id":               nid,
                "label":            str(row.get("gene_symbol", "")),
                "type":             "target",
                "gene_symbol":      str(row.get("gene_symbol", "")),
                "pareto_rank":      rank,
                "failure_mode":     fail,
                "association_score": float(row.get("association_score", 0.5)),
                "novelty_score":    float(row.get("novelty_score", 0.5)),
                "disease_name":     disease_name,
                "endotype_label":   "",
            }, "classes": f"target {rc} {fail}"})

            elements.append({"data": {
                "source": "disease",
                "target": nid,
                "weight": float(row.get("association_score", 0.5)),
            }})

    else:  # patient mode
        seen: set[str] = set()

        for eid in sorted(df["endotype_id"].unique()):
            eid_df = df[df["endotype_id"] == eid]
            label  = (eid_df["endotype_label"].iloc[0]
                      if not eid_df.empty else f"Endotype {eid+1}")
            short  = label.split(":", 1)[1].strip() if ":" in label else label
            short  = (short[:22] + "…") if len(short) > 22 else short
            elements.append({"data": {
                "id":         f"endo_{eid}",
                "label":      short,
                "type":       "endotype",
                "full_label": label,
            }, "classes": "endotype"})

        for _, row in df.iterrows():
            eid    = int(row.get("endotype_id", 0))
            gene   = str(row.get("gene_symbol", ""))
            nid    = f"{gene}_{eid}"
            if nid in seen:
                continue
            seen.add(nid)

            rank  = int(row.get("endotype_pareto_rank", 99))
            rc    = f"rank{rank}" if rank <= 2 else "rank3plus"
            fail  = str(row.get("predicted_failure_mode", "unknown")).replace(" ", "_")
            endo_label = (df[df["endotype_id"] == eid]["endotype_label"].iloc[0]
                          if not df[df["endotype_id"] == eid].empty else "")

            elements.append({"data": {
                "id":               nid,
                "label":            gene,
                "type":             "target",
                "gene_symbol":      gene,
                "pareto_rank":      rank,
                "failure_mode":     fail,
                "association_score": float(row.get("association_score", 0.5)),
                "novelty_score":    float(row.get("novelty_score", 0.5)),
                "disease_name":     endo_label,
                "endotype_label":   endo_label,
            }, "classes": f"target {rc} {fail}"})

            elements.append({"data": {
                "source": f"endo_{eid}",
                "target": nid,
                "weight": float(row.get("association_score", 0.5)),
            }})

    return elements


def _cyto_stylesheet() -> list[dict]:
    return [
        # Base
        {"selector": "node", "style": {
            "label":          "data(label)",
            "font-size":      "10px",
            "text-valign":    "center",
            "text-halign":    "center",
            "text-wrap":      "wrap",
            "text-max-width": "70px",
            "color":          "#333",
            "border-width":   1.5,
            "border-color":   "#ccc",
        }},
        # Disease
        {"selector": "node[type='disease']", "style": {
            "background-color": "#E05A5A",
            "color":            "#fff",
            "width":  72, "height": 72,
            "font-size":   "11px",
            "font-weight": "bold",
            "border-color": "#b83b3b",
            "border-width": 3,
        }},
        # Endotype hubs
        {"selector": "node.endotype", "style": {
            "background-color": "#5B8DEF",
            "color":            "#fff",
            "width":  64, "height": 64,
            "font-size":   "10px",
            "font-weight": "600",
            "border-color": "#3a6bc4",
            "border-width": 2,
        }},
        # Target rank 1 (Pareto front)
        {"selector": "node.rank1", "style": {
            "background-color": "#4CAF7D",
            "shape":       "diamond",
            "width":  52, "height": 52,
            "font-weight": "700",
            "font-size":   "11px",
            "color":       "#fff",
            "border-width": 2.5,
            "border-color": "#2e7d52",
        }},
        # Target rank 2
        {"selector": "node.rank2", "style": {
            "background-color": "#5B8DEF",
            "width":  40, "height": 40,
            "color":       "#fff",
            "border-color": "#3a6bc4",
        }},
        # Target rank 3+
        {"selector": "node.rank3plus", "style": {
            "background-color": "#CCCCCC",
            "width":  26, "height": 26,
            "color":       "#555",
            "border-color": "#aaa",
        }},
        # Failure mode borders
        {"selector": "node.toxicity", "style": {
            "border-color": "#E05A5A", "border-width": 3.5}},
        {"selector": "node.efficacy", "style": {
            "border-color": "#F0A500", "border-width": 3.5}},
        {"selector": "node.likely_success", "style": {
            "border-color": "#4CAF7D", "border-width": 3}},
        # Selected
        {"selector": "node:selected", "style": {
            "border-width": 5,
            "border-color": "#222",
            "z-index": 999,
        }},
        # Edges
        {"selector": "edge", "style": {
            "width":          1.5,
            "line-color":     "#ddd",
            "opacity":        0.6,
            "curve-style":    "bezier",
        }},
    ]


# ---------------------------------------------------------------------------
# Claude co-scientist explanation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NL summary + chatbot helpers
# ---------------------------------------------------------------------------

def _generate_result_summary(result) -> str:
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "_Set `ANTHROPIC_API_KEY` to enable AI narrative summaries._"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_summary_prompt(result)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in msg.content if hasattr(b, "text")), "")
    except Exception as exc:
        return f"_Summary unavailable: {exc}_"


def _build_summary_prompt(result) -> str:
    df = result.ranked_targets
    top5 = df.head(5)

    if result.mode == "target":
        disease = (df["disease_name"].iloc[0]
                   if "disease_name" in df.columns and not df.empty else "unknown disease")
        gene = df["gene_symbol"].iloc[0] if not df.empty else "unknown"
        targets_txt = "\n".join(
            f"- {r['gene_symbol']}: rank {r.get('pareto_rank','?')}, "
            f"assoc {float(r.get('association_score',0)):.3f}, "
            f"novelty {float(r.get('novelty_score',0)):.3f}, "
            f"failure mode: {r.get('predicted_failure_mode','unknown')}"
            for _, r in top5.iterrows()
        )
        return f"""Summarize these drug target analysis results in 3–4 sentences for a clinician scientist. Be specific, evidence-grounded, and end with a clear therapeutic recommendation.

Mode: Target-first competitive landscape
Focal gene: {gene} | Disease: {disease}
Competing targets analysed: {len(df)}
Pareto objectives: {', '.join(result.pareto.objective_names)}
Top 5 ranked targets:
{targets_txt}

Use **bold** for gene names and key terms. No bullet points. Open with the disease context and the key Pareto-front finding."""

    # patient mode
    endotype_rows = []
    for eid in sorted(df.get("endotype_id", pd.Series(dtype=int)).unique())[:5]:
        eid_df = df[df["endotype_id"] == eid]
        label = eid_df["endotype_label"].iloc[0] if not eid_df.empty else f"Endotype {eid+1}"
        rank_col = "endotype_pareto_rank" if "endotype_pareto_rank" in eid_df.columns else "association_score"
        top = eid_df.sort_values(rank_col).iloc[0] if not eid_df.empty else None
        if top is not None:
            endotype_rows.append(f"- {label}: top target {top['gene_symbol']}")
    endo_txt = "\n".join(endotype_rows) if endotype_rows else "N/A"

    n_patients = int(result.endotyping.labels[result.endotyping.labels >= 0].count())
    return f"""Summarize these patient cohort analysis results in 3–4 sentences for a clinician scientist. Be specific about the patient subgroups, their biological distinctiveness, and the highest-priority targets per endotype.

Mode: Patient-first (Coherent EHR synthetic cohort)
Patients: {n_patients} | Endotypes identified: {result.endotyping.n_clusters}
Top target per endotype:
{endo_txt}

Use **bold** for endotype names and gene symbols. No bullet points. Open with a summary of the cohort stratification."""


def _build_chat_system_prompt(result) -> str:
    if result is None:
        return ("You are an AI co-scientist in a drug discovery platform. "
                "Answer questions about drug targets and precision medicine.")
    df = result.ranked_targets
    top10 = df.head(10)
    targets_txt = "\n".join(
        f"- {r['gene_symbol']}: rank {r.get('pareto_rank', r.get('endotype_pareto_rank','?'))}, "
        f"assoc {float(r.get('association_score',0)):.3f}, "
        f"novelty {float(r.get('novelty_score',0)):.3f}, "
        f"failure mode: {r.get('predicted_failure_mode','unknown')}"
        for _, r in top10.iterrows()
    )
    if result.mode == "target":
        disease = (df["disease_name"].iloc[0]
                   if "disease_name" in df.columns and not df.empty else "unknown")
        gene = df["gene_symbol"].iloc[0] if not df.empty else "unknown"
        ctx = (f"Target-first competitive landscape\n"
               f"Focal gene: {gene} | Disease: {disease}\n"
               f"{len(df)} competing targets analysed")
    else:
        endo_labels = (df["endotype_label"].unique().tolist()[:5]
                       if "endotype_label" in df.columns else [])
        ctx = (f"Patient-first cohort (Coherent EHR synthetic data)\n"
               f"{result.endotyping.n_clusters} endotypes: {', '.join(str(e) for e in endo_labels)}")

    return f"""You are an AI co-scientist embedded in the Stratified Precision drug discovery platform. You have full knowledge of the current analysis and help the clinician interpret findings, assess risks, and plan next steps.

Current analysis:
{ctx}
Pareto objectives: {', '.join(result.pareto.objective_names)}

Top ranked targets:
{targets_txt}

Be concise, clinically grounded, and specific. Use markdown. Reference actual scores when discussing a target. You may ask clarifying questions to help the user dig deeper."""


def _chat_claude(system_prompt: str, history: list[dict]) -> str:
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "_No `ANTHROPIC_API_KEY` found._"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=700,
            system=system_prompt,
            messages=history,
        )
        return next((b.text for b in msg.content if hasattr(b, "text")), "No response.")
    except Exception as exc:
        return f"_Error: {exc}_"


def _generate_explanation(target_data: dict) -> str:
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ("_No `ANTHROPIC_API_KEY` environment variable found. "
                "Set it to enable AI explanations._")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        gene     = target_data.get("gene_symbol", "Unknown")
        disease  = target_data.get("disease_name", "Unknown disease")
        rank     = target_data.get("pareto_rank", "?")
        failure  = target_data.get("failure_mode", "unknown")
        assoc    = float(target_data.get("association_score", 0) or 0)
        novelty  = float(target_data.get("novelty_score", 0) or 0)
        endotype = target_data.get("endotype_label", "")

        context_line = f"Patient subgroup: {endotype}\n" if endotype else ""

        prompt = f"""You are an AI co-scientist in drug discovery helping a clinician interpret a target ranking.

Gene / target: {gene}
Disease context: {disease}
{context_line}Pareto rank: {rank} (rank 1 = Pareto-optimal across all objectives)
OpenTargets association score: {assoc:.3f} (0–1; higher = stronger multi-evidence support)
Novelty score: {novelty:.3f} (1.0 = no approved drugs yet; 0 = well-validated, crowded)
Predicted failure mode: {failure}

Write a concise, precise explanation in 3–4 sentences for a clinician scientist:
1. Biological rationale — why does this gene matter mechanistically for the disease?
2. Evidence quality — what does the association score imply about genetic/clinical validation?
3. Development risk — what does the failure mode prediction mean for a real programme?
4. Opportunity — given the novelty score, what modality (antibody, small molecule, ASO, cell therapy) is most tractable?

Stay grounded in the data above. Do not invent clinical trial names or unpublished results. No bullet points."""

        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=450,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    except Exception as exc:
        return f"_Explanation unavailable: {exc}_"
