"""
Mode 1 — Target-first entry point.

Start with a gene symbol or Ensembl ID. Fetches biological context from
OpenTargets, then hands a TargetContext to the shared pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from ..data.opentargets import OpenTargetsClient


@dataclass
class TargetContext:
    """All data associated with a single drug target."""
    gene_symbol: str
    ensembl_id: str

    # The disease context chosen by the user (or auto-selected as top hit)
    disease_id: Optional[str] = None
    disease_name: Optional[str] = None

    # Competitive landscape: top targets for the chosen disease (for Pareto ranking)
    # columns = [ensembl_id, gene_symbol, gene_name, association_score, score_*, ...]
    competitive_landscape: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Disease associations: columns = [disease_id, disease_name, score, ...]
    disease_associations: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Tissue expression: columns = [tissue, tpm_mean, tpm_sd, ...]
    tissue_expression: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Biological network: list of (gene_a, gene_b, interaction_type, score)
    network_edges: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Clinical evidence: columns = [trial_id, phase, outcome, failure_reason, ...]
    clinical_evidence: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Safety signals pulled from OpenTargets safety section
    safety_liabilities: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_target(
    gene_symbol: str,
    ensembl_id: Optional[str] = None,
    disease_id: Optional[str] = None,
    disease_name: Optional[str] = None,
    client: Optional[OpenTargetsClient] = None,
) -> TargetContext:
    """
    Resolve a gene symbol and pull all relevant data from OpenTargets.

    If disease_id is provided the competitive landscape for that disease is fetched
    (top co-associated targets) so the pipeline can Pareto-rank them.
    If disease_id is None the top disease association is used automatically.
    """
    ot = client or OpenTargetsClient()

    if ensembl_id is None:
        ensembl_id = ot.resolve_gene_symbol(gene_symbol)
        if ensembl_id is None:
            raise ValueError(f"Gene '{gene_symbol}' not found in OpenTargets.")

    disease_df = ot.get_disease_associations(ensembl_id)
    tissue_df  = ot.get_tissue_expression(ensembl_id)
    network_df = ot.get_network_edges(ensembl_id)
    clinical_df = ot.get_clinical_evidence(ensembl_id)
    safety_df  = ot.get_safety_liabilities(ensembl_id)

    # Auto-select top disease if none specified
    if disease_id is None and not disease_df.empty:
        top = disease_df.sort_values("score", ascending=False).iloc[0]
        disease_id   = top["disease_id"]
        disease_name = top["disease_name"]

    landscape_df = pd.DataFrame()
    if disease_id:
        landscape_df = ot.get_disease_competitive_landscape(disease_id, size=15)

    return TargetContext(
        gene_symbol=gene_symbol,
        ensembl_id=ensembl_id,
        disease_id=disease_id,
        disease_name=disease_name,
        competitive_landscape=landscape_df,
        disease_associations=disease_df,
        tissue_expression=tissue_df,
        network_edges=network_df,
        clinical_evidence=clinical_df,
        safety_liabilities=safety_df,
    )
