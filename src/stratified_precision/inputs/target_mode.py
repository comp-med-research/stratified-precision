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
    client: Optional[OpenTargetsClient] = None,
) -> TargetContext:
    """
    Resolve a gene symbol (or Ensembl ID) and pull all relevant data
    from OpenTargets to populate a TargetContext.

    Parameters
    ----------
    gene_symbol:
        HGNC symbol, e.g. "BACE1", "PCSK9"
    ensembl_id:
        Optional Ensembl gene ID. If None, resolved via OpenTargets search.
    client:
        Optional pre-constructed client (useful for testing / shared sessions).
    """
    ot = client or OpenTargetsClient()

    if ensembl_id is None:
        ensembl_id = ot.resolve_gene_symbol(gene_symbol)

    disease_df = ot.get_disease_associations(ensembl_id)
    tissue_df = ot.get_tissue_expression(ensembl_id)
    network_df = ot.get_network_edges(ensembl_id)
    clinical_df = ot.get_clinical_evidence(ensembl_id)
    safety_df = ot.get_safety_liabilities(ensembl_id)

    return TargetContext(
        gene_symbol=gene_symbol,
        ensembl_id=ensembl_id,
        disease_associations=disease_df,
        tissue_expression=tissue_df,
        network_edges=network_df,
        clinical_evidence=clinical_df,
        safety_liabilities=safety_df,
    )
