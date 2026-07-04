"""
Mode 2 — Patient-first entry point.

Start with a clinical dataset (phenotypic, multi-omics, or EHR-derived).
Discovers disease endotypes, enriches each with GWAS signal, and maps
to OpenTargets candidate targets — one ranked list per subgroup.

This is the clinician's natural starting point: you have patients, not
a pre-selected molecule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from ..data.opentargets import OpenTargetsClient
from ..endotyping.clustering import EndotypingResult, discover_endotypes


@dataclass
class CandidateTarget:
    """A target surfaced from a patient cohort analysis."""
    ensembl_id: str
    gene_symbol: str

    # Which patient subgroup this target is most relevant to
    endotype_id: int
    endotype_label: str

    # OpenTargets association scores for this target × disease
    association_score: float = 0.0
    genetic_association_score: float = 0.0
    expression_specificity_score: float = 0.0

    # Novelty: 1 = no approved drugs, 0 = fully saturated target
    novelty_score: float = 0.0

    # Filled in later by the pipeline
    predicted_failure_mode: Optional[str] = None   # "toxicity" | "efficacy" | None
    pareto_rank: Optional[int] = None


@dataclass
class PatientCohortContext:
    """Everything derived from the input patient dataset."""
    source_path: str
    feature_matrix: pd.DataFrame            # cleaned features used for endotyping
    endotyping: EndotypingResult            # cluster assignments + UMAP coords
    candidate_targets: list[CandidateTarget] = field(default_factory=list)


def load_patient_cohort(
    data_path: str | Path,
    feature_cols: Optional[list[str]] = None,
    disease_ontology_id: Optional[str] = None,
    n_endotypes: Optional[int] = None,
    client: Optional[OpenTargetsClient] = None,
) -> PatientCohortContext:
    """
    Load a clinical/omics dataset, discover patient subgroups (endotypes),
    and surface candidate targets for each subgroup via OpenTargets.

    Parameters
    ----------
    data_path:
        Path to CSV/TSV with one row per patient. Columns can be clinical
        variables, gene expression values, or any mix.
    feature_cols:
        Columns to use for endotyping. If None, uses all numeric columns.
    disease_ontology_id:
        EFO/MONDO ID to restrict GWAS enrichment (e.g. "EFO_0000270" for asthma).
        If None, inferred from the most enriched disease in OpenTargets.
    n_endotypes:
        Number of clusters. If None, HDBSCAN selects automatically.
    client:
        Optional pre-constructed OpenTargets client.
    """
    ot = client or OpenTargetsClient()

    df = _load_and_clean(data_path, feature_cols)
    endotyping = discover_endotypes(df, n_clusters=n_endotypes)

    candidates: list[CandidateTarget] = []
    for cluster_id in sorted(endotyping.labels.unique()):
        if cluster_id == -1:
            continue  # HDBSCAN noise points

        cluster_mask = endotyping.labels == cluster_id
        cluster_features = df[cluster_mask]

        # Find differentially expressed / associated features for this subgroup
        top_genes = _top_genes_for_cluster(cluster_features, df[~cluster_mask])

        # Map each top gene to OpenTargets
        for gene_symbol in top_genes[:20]:
            ensembl_id = ot.resolve_gene_symbol(gene_symbol, silent=True)
            if ensembl_id is None:
                continue

            scores = ot.get_association_scores(
                ensembl_id, disease_ontology_id=disease_ontology_id
            )
            novelty = ot.get_novelty_score(ensembl_id)

            candidates.append(
                CandidateTarget(
                    ensembl_id=ensembl_id,
                    gene_symbol=gene_symbol,
                    endotype_id=cluster_id,
                    endotype_label=f"Subgroup {cluster_id + 1}",
                    association_score=scores.get("overall", 0.0),
                    genetic_association_score=scores.get("genetics", 0.0),
                    expression_specificity_score=scores.get("expression", 0.0),
                    novelty_score=novelty,
                )
            )

    return PatientCohortContext(
        source_path=str(data_path),
        feature_matrix=df,
        endotyping=endotyping,
        candidate_targets=candidates,
    )


def _load_and_clean(
    data_path: str | Path,
    feature_cols: Optional[list[str]],
) -> pd.DataFrame:
    path = Path(data_path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)

    if feature_cols:
        df = df[feature_cols]
    else:
        df = df.select_dtypes(include=[np.number])

    # Drop columns with >30% missing, then median-impute the rest
    df = df.loc[:, df.isnull().mean() < 0.3]
    df = df.fillna(df.median(numeric_only=True))

    return df


def _top_genes_for_cluster(
    cluster_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    top_n: int = 50,
) -> list[str]:
    """Return column names with highest mean fold-change in this cluster vs rest."""
    cluster_mean = cluster_df.mean()
    rest_mean = rest_df.mean().replace(0, np.nan)
    fold_change = (cluster_mean / rest_mean).dropna()
    return fold_change.nlargest(top_n).index.tolist()
