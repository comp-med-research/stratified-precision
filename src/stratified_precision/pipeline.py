"""
Shared analysis pipeline — accepts output from either input mode and
runs endotyping → causal analysis → KG enrichment → Bayesian objective
discovery → dynamic Pareto ranking → literature agent.

The KG enrichment step is what makes objectives disease-context-dependent:
a CF target's Pareto front is optimised on different axes than an AD target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional
import numpy as np
import pandas as pd

from .inputs.target_mode import TargetContext
from .inputs.patient_mode import PatientCohortContext, CandidateTarget
from .endotyping.clustering import EndotypingResult, discover_endotypes
from .causal.failure_classifier import FailureMode, classify_failure_modes
from .agents.literature_agent import LiteratureAgent, TrialFailureSignal
from .kg.hetionet import (
    load_hetionet, find_gene_node, find_disease_node,
    extract_subgraph, extract_graph_features,
)
from .kg.kg_selector_agent import KGSelectorAgent, KnowledgeSource
from .optimization.bayesian_weighter import BayesianObjectiveWeighter, WeightedObjectives
from .optimization.pareto import ParetoResult, compute_pareto_front


@dataclass
class PipelineResult:
    """Final output of the full analysis pipeline."""

    # Ranked targets, augmented with failure mode and Pareto rank
    ranked_targets: pd.DataFrame

    # Per-target failure mode predictions
    failure_modes: list[FailureMode]

    # Literature signals retrieved by the agent
    literature_signals: list[TrialFailureSignal]

    # Endotyping result (UMAP coords + cluster labels) for visualisation
    endotyping: EndotypingResult

    # Pareto front details (objectives are dynamic — check pareto.objective_names)
    pareto: ParetoResult

    # Per-endotype Pareto fronts (patient mode only; empty dict in target mode)
    pareto_per_endotype: dict[int, ParetoResult] = field(default_factory=dict)

    # KG features extracted per candidate (gene_symbol → feature dict)
    kg_features: dict[str, dict[str, float]] = field(default_factory=dict)

    # Which KG sources were selected for this disease context
    kg_sources: list[KnowledgeSource] = field(default_factory=list)

    # Bayesian objective weights (feature → posterior weight)
    objective_weights: dict[str, float] = field(default_factory=dict)

    # Which input mode produced this result
    mode: str = "target"  # "target" | "patient"


def run_pipeline(
    context: Union[TargetContext, PatientCohortContext],
    run_literature_agent: bool = True,
    use_hetionet: bool = True,
    anthropic_api_key: str | None = None,
    kg_hops: int = 2,
) -> PipelineResult:
    """
    Run the full analysis pipeline from either input mode.

    Parameters
    ----------
    context:
        Either a TargetContext (Mode 1) or PatientCohortContext (Mode 2).
    run_literature_agent:
        Whether to call the Claude literature agent. Set False for fast/offline runs.
    use_hetionet:
        Whether to load Hetionet and extract graph features for dynamic objectives.
        Requires the Hetionet JSON to be downloaded (~200MB, one-time).
        Set False for fast/offline runs.
    anthropic_api_key:
        Required if run_literature_agent=True. Falls back to ANTHROPIC_API_KEY env var.
    kg_hops:
        Number of hops for Hetionet subgraph extraction (2 is usually enough).
    """
    hetionet_graph = None
    if use_hetionet:
        try:
            hetionet_graph = load_hetionet()
        except Exception as e:
            print(f"[Pipeline] Hetionet unavailable ({e}), skipping KG enrichment.")

    if isinstance(context, TargetContext):
        return _run_target_pipeline(
            context, run_literature_agent, anthropic_api_key,
            hetionet_graph=hetionet_graph, kg_hops=kg_hops,
        )
    elif isinstance(context, PatientCohortContext):
        return _run_patient_pipeline(
            context, run_literature_agent, anthropic_api_key,
            hetionet_graph=hetionet_graph, kg_hops=kg_hops,
        )
    else:
        raise TypeError(f"Unknown context type: {type(context)}")


def _run_target_pipeline(
    ctx: TargetContext,
    run_agent: bool,
    api_key: str | None,
    hetionet_graph=None,
    kg_hops: int = 2,
) -> PipelineResult:
    # ── Competitive landscape (one row per competing target for this disease) ──
    # If we have it, Pareto-rank the competitors; focal gene is highlighted.
    # Fallback: single-target mode using disease associations.
    if not ctx.competitive_landscape.empty:
        candidate_df = _landscape_to_candidate_df(ctx)
    else:
        candidate_df = _target_to_candidate_df_legacy(ctx)

    endotyping = _endotype_from_candidates(candidate_df)

    failure_modes = classify_failure_modes(target_context=ctx, endotyping=endotyping)
    # Attach failure mode to the focal gene rows only
    fm = failure_modes[0] if failure_modes else None
    candidate_df["safety_score"]           = candidate_df.get("safety_score",
                                                pd.Series([fm.safety_score if fm else 0.5] * len(candidate_df))).fillna(
                                                fm.safety_score if fm else 0.5)
    candidate_df["efficacy_score"]         = candidate_df.get("efficacy_score",
                                                pd.Series([fm.efficacy_score if fm else 0.5] * len(candidate_df))).fillna(
                                                fm.efficacy_score if fm else 0.5)
    candidate_df["predicted_failure_mode"] = candidate_df.get("predicted_failure_mode",
                                                pd.Series([fm.predicted_mode if fm else "unknown"] * len(candidate_df))).fillna(
                                                fm.predicted_mode if fm else "unknown")

    # ── If competitive landscape, use datatype scores as Pareto objectives ──
    landscape_extra: Optional[dict[str, np.ndarray]] = None
    if not ctx.competitive_landscape.empty:
        landscape_extra = {}
        score_map = {
            "score_genetic_association": "genetic_causal_evidence",
            "score_animal_model":        "translational_evidence",
            "score_genetic_literature":  "literature_support",
        }
        for col, label in score_map.items():
            if col in candidate_df.columns:
                landscape_extra[label] = candidate_df[col].fillna(0.0).values.astype(float)
        # Clinical score inverted → novelty (no drugs = opportunity)
        if "score_clinical" in candidate_df.columns:
            landscape_extra["clinical_novelty"] = (
                1.0 - candidate_df["score_clinical"].fillna(0.0).clip(0, 1).values
            ).astype(float)

    # ── KG enrichment ──
    kg_features: dict[str, dict[str, float]] = {}
    kg_sources: list[KnowledgeSource] = []
    weighted_objs: Optional[WeightedObjectives] = None
    disease_name = ctx.disease_name or ""

    if hetionet_graph is not None:
        gene_node    = find_gene_node(hetionet_graph, ctx.gene_symbol)
        disease_node = find_disease_node(hetionet_graph, disease_name) if disease_name else None
        if gene_node:
            subgraph = extract_subgraph(hetionet_graph, gene_node, disease_node, n_hops=kg_hops)
            feats    = extract_graph_features(hetionet_graph, subgraph, gene_node, disease_node)
            kg_features[ctx.gene_symbol] = feats
        selector   = KGSelectorAgent(api_key=api_key)
        kg_sources = selector.select_sources(disease_name, ctx.gene_symbol)

    extra_objectives: Optional[dict[str, np.ndarray]] = None
    if kg_features:
        weighter = BayesianObjectiveWeighter(disease_name=disease_name)
        weighter.fit(np.zeros((1, 1)), [], None)
        base_obj_arrays = {
            "efficacy_potential": candidate_df["association_score"].fillna(0.5).values
                                  * (1 - candidate_df["efficacy_score"].fillna(0.5).values),
            "safety_margin":      1 - candidate_df["safety_score"].fillna(0.5).values,
        }
        candidate_feature_dicts = [
            kg_features.get(g, {}) for g in candidate_df["gene_symbol"]
        ]
        weighted_objs = weighter.compute_weighted_objectives(candidate_feature_dicts, base_obj_arrays)
        extra_objectives = {
            k: weighted_objs.objective_matrix[:, i]
            for i, k in enumerate(weighted_objs.objective_names)
            if k.startswith("kg_")
        }

    literature_signals: list[TrialFailureSignal] = []
    if run_agent:
        agent = LiteratureAgent(api_key=api_key)
        literature_signals = agent.scan_target(ctx.gene_symbol)

    # Merge landscape + KG extra objectives; landscape takes precedence for target mode
    merged_extra = {**(landscape_extra or {}), **(extra_objectives or {})}
    pareto  = compute_pareto_front(candidate_df, extra_objectives=merged_extra or None)
    ranked  = _attach_pareto_ranks(candidate_df, pareto)
    for col_name in pareto.objective_matrix.columns:
        ranked[f"obj_{col_name}"] = pareto.objective_matrix[col_name].values

    return PipelineResult(
        ranked_targets=ranked,
        failure_modes=failure_modes,
        literature_signals=literature_signals,
        endotyping=endotyping,
        pareto=pareto,
        kg_features=kg_features,
        kg_sources=kg_sources,
        objective_weights=weighted_objs.weights if weighted_objs else {},
        mode="target",
    )


def _run_patient_pipeline(
    ctx: PatientCohortContext,
    run_agent: bool,
    api_key: str | None,
    hetionet_graph=None,
    kg_hops: int = 2,
) -> PipelineResult:
    endotyping = ctx.endotyping

    failure_modes = classify_failure_modes(
        candidates=ctx.candidate_targets,
        endotyping=endotyping,
    )

    # --- KG enrichment: one subgraph per candidate ---
    kg_features: dict[str, dict[str, float]] = {}
    kg_sources: list[KnowledgeSource] = []

    # Infer disease from the most common endotype label or top candidate
    top_disease = ""
    if ctx.candidate_targets:
        top_disease = ctx.candidate_targets[0].endotype_label

    if hetionet_graph is not None:
        disease_node = find_disease_node(hetionet_graph, top_disease) if top_disease else None

        for candidate in ctx.candidate_targets:
            gene_node = find_gene_node(hetionet_graph, candidate.gene_symbol)
            if gene_node:
                subgraph = extract_subgraph(
                    hetionet_graph, gene_node, disease_node, n_hops=kg_hops
                )
                feats = extract_graph_features(hetionet_graph, subgraph, gene_node, disease_node)
                kg_features[candidate.gene_symbol] = feats

        selector = KGSelectorAgent(api_key=api_key)
        kg_sources = selector.select_sources(top_disease, "")

    candidate_df = _candidates_to_df(ctx.candidate_targets, failure_modes, [])
    extra_objectives: Optional[dict[str, np.ndarray]] = None
    weighted_objs: Optional[WeightedObjectives] = None

    if kg_features:
        weighter = BayesianObjectiveWeighter(disease_name=top_disease)
        weighter.fit(np.zeros((1, 1)), [], None)

        base_obj_arrays = {
            "efficacy_potential": (
                candidate_df["association_score"].fillna(0.5).values
                * (1 - candidate_df["efficacy_score"].fillna(0.5).values)
            ),
            "safety_margin": 1 - candidate_df["safety_score"].fillna(0.5).values,
        }

        gene_col = candidate_df["gene_symbol"] if "gene_symbol" in candidate_df.columns else []
        candidate_feature_dicts = [
            kg_features.get(g, {}) for g in gene_col
        ]

        weighted_objs = weighter.compute_weighted_objectives(
            candidate_feature_dicts, base_obj_arrays
        )
        extra_objectives = {
            k: weighted_objs.objective_matrix[:, i]
            for i, k in enumerate(weighted_objs.objective_names)
            if k.startswith("kg_")
        }

    literature_signals: list[TrialFailureSignal] = []
    if run_agent:
        agent = LiteratureAgent(api_key=api_key)
        # Deduplicate — same gene can appear in multiple endotypes; cap at 30
        # to keep the pipeline under ~5 min even for large cohorts.
        seen: dict[str, None] = {}
        for c in ctx.candidate_targets:
            seen[c.gene_symbol] = None
        gene_symbols = list(seen.keys())[:30]
        literature_signals = agent.scan_candidates(gene_symbols)

    # ── Per-endotype Pareto fronts ──────────────────────────────────────
    pareto_per_endotype: dict[int, ParetoResult] = {}
    ranked_parts = []

    for eid, group in candidate_df.groupby("endotype_id"):
        # Slice extra_objectives to just the rows in this group
        group_extra: Optional[dict[str, np.ndarray]] = None
        if extra_objectives:
            idx = group.index
            group_extra = {k: v[idx] for k, v in extra_objectives.items()}

        p = compute_pareto_front(group.reset_index(drop=True), extra_objectives=group_extra)
        pareto_per_endotype[int(eid)] = p

        g = group.copy()
        g["endotype_pareto_rank"]  = p.ranks
        g["endotype_pareto_front"] = p.on_front
        # Store per-row objective values so the dashboard can read them after sort
        for col_name in p.objective_matrix.columns:
            g[f"obj_{col_name}"] = p.objective_matrix[col_name].values
        ranked_parts.append(g)

    ranked = pd.concat(ranked_parts).sort_values(
        ["endotype_id", "endotype_pareto_front", "endotype_pareto_rank"],
        ascending=[True, False, True],
    )

    # Global pareto (used by the summary header and target-mode compatibility)
    pareto = compute_pareto_front(candidate_df, extra_objectives=extra_objectives)

    return PipelineResult(
        ranked_targets=ranked,
        failure_modes=failure_modes,
        literature_signals=literature_signals,
        endotyping=endotyping,
        pareto=pareto,
        pareto_per_endotype=pareto_per_endotype,
        kg_features=kg_features,
        kg_sources=kg_sources,
        objective_weights=weighted_objs.weights if weighted_objs else {},
        mode="patient",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _landscape_to_candidate_df(ctx: TargetContext) -> pd.DataFrame:
    """
    Build a candidate DataFrame from the competitive landscape —
    one row per competing target for the chosen disease context.
    The focal gene is flagged with is_focal=True.
    """
    df = ctx.competitive_landscape.copy().fillna(0.0)
    df["disease_id"]   = ctx.disease_id or ""
    df["disease_name"] = ctx.disease_name or ""
    df["is_focal"]     = df["ensembl_id"] == ctx.ensembl_id

    # Map OpenTargets datatype scores to pipeline objectives:
    #   genetic_association → primary efficacy signal (higher = stronger causal evidence)
    #   clinical            → clinical validation (invert → novelty; higher clinical = less novel)
    #   animal_model        → translational evidence (folds into efficacy)
    genetic = df.get("score_genetic_association", pd.Series(0.5, index=df.index)).clip(0, 1)
    animal  = df.get("score_animal_model",        pd.Series(0.5, index=df.index)).clip(0, 1)
    clinical = df.get("score_clinical",           pd.Series(0.0, index=df.index)).clip(0, 1)

    # efficacy_score here is the *failure* probability (high = bad); used by _resolve_mode
    df["efficacy_score"]  = (1.0 - (genetic + animal) / 2).clip(0, 1)
    df["safety_score"]    = pd.Series(0.4, index=df.index)   # uniform prior (no safety data yet)
    df["novelty_score"]   = (1.0 - clinical).clip(0, 1)
    df["predicted_failure_mode"] = "unknown"
    df["literature_risk"] = 0.0
    return df


def _endotype_from_candidates(candidate_df: pd.DataFrame) -> EndotypingResult:
    """Cluster competing targets by their score profile."""
    score_cols = [c for c in candidate_df.columns if c.startswith("score_")]
    if not score_cols:
        score_cols = ["association_score", "efficacy_score", "novelty_score"]
    feature_matrix = candidate_df[
        [c for c in score_cols if c in candidate_df.columns]
    ].select_dtypes("number").fillna(0.0)
    return discover_endotypes(feature_matrix)


def _target_to_candidate_df_legacy(ctx: TargetContext) -> pd.DataFrame:
    """Fallback: one row per disease association (used when no competitive landscape)."""
    rows = []
    for _, row in ctx.disease_associations.iterrows():
        rows.append({
            "ensembl_id":     ctx.ensembl_id,
            "gene_symbol":    ctx.gene_symbol,
            "gene_name":      ctx.gene_symbol,
            "disease_id":     row.get("disease_id", ""),
            "disease_name":   row.get("disease_name", ""),
            "association_score": row.get("score", 0.0),
            "safety_score":   0.5,
            "efficacy_score": 0.5,
            "novelty_score":  0.5,
            "predicted_failure_mode": "unknown",
            "literature_risk": 0.0,
            "is_focal": True,
        })
    return pd.DataFrame(rows)


def _candidates_to_df(
    candidates: list[CandidateTarget],
    failure_modes: list[FailureMode],
    lit_signals: list[TrialFailureSignal],
) -> pd.DataFrame:
    fm_map = {fm.target_id: fm for fm in failure_modes}
    lit_map = _build_lit_map(lit_signals)

    rows = []
    for c in candidates:
        fm = fm_map.get(c.ensembl_id)
        rows.append({
            "ensembl_id": c.ensembl_id,
            "gene_symbol": c.gene_symbol,
            "endotype_id": c.endotype_id,
            "endotype_label": c.endotype_label,
            "association_score": c.association_score,
            "novelty_score": c.novelty_score,
            "expression_specificity_score": c.expression_specificity_score,
            "safety_score": fm.safety_score if fm else 0.0,
            "efficacy_score": fm.efficacy_score if fm else 0.0,
            "predicted_failure_mode": fm.predicted_mode if fm else None,
            "literature_risk": lit_map.get(c.gene_symbol, 0.0),
        })
    return pd.DataFrame(rows)


def _build_lit_map(signals: list[TrialFailureSignal]) -> dict[str, float]:
    """Aggregate literature risk signals per gene symbol (0–1 scale)."""
    result: dict[str, float] = {}
    for s in signals:
        result[s.gene_symbol] = max(result.get(s.gene_symbol, 0.0), s.risk_score)
    return result


def _attach_pareto_ranks(df: pd.DataFrame, pareto: ParetoResult) -> pd.DataFrame:
    df = df.copy()
    df["pareto_rank"] = pareto.ranks
    df["pareto_front"] = pareto.on_front
    return df.sort_values(["pareto_front", "pareto_rank"], ascending=[False, True])
