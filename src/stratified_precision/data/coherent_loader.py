"""
Loader for the Coherent synthetic EHR dataset.

Merges patients / conditions / observations / per-patient DNA variants into
a single per-patient feature matrix suitable for endotyping and target discovery.

Dataset: https://github.com/synthetichealth/synthea (Coherent 11-07-2022 build)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd

from ..inputs.patient_mode import CandidateTarget, PatientCohortContext
from ..endotyping.clustering import discover_endotypes
from ..data.opentargets import OpenTargetsClient


# ---------------------------------------------------------------------------
# Key lab names to extract (substring match on DESCRIPTION)
# ---------------------------------------------------------------------------
_LAB_MAP = {
    "bmi":            "Body Mass Index",
    "systolic_bp":    "Systolic Blood Pressure",
    "diastolic_bp":   "Diastolic Blood Pressure",
    "cholesterol":    "Total Cholesterol",
    "ldl":            "Low Density Lipoprotein Cholesterol",
    "hdl":            "High Density Lipoprotein Cholesterol",
    "triglycerides":  "Triglycerides",
    "hba1c":          "Hemoglobin A1c/Hemoglobin.total in Blood",
    "glucose":        "Glucose",
    "creatinine":     "Creatinine",
}

# Map top condition flags to OpenTargets disease EFO IDs for supplementary target discovery
_FLAG_TO_DISEASE_ID = {
    "flag_alzheimers":     "MONDO_0004975",
    "flag_chd":            "MONDO_0004989",
    "flag_stroke":         "MONDO_0005098",
    "flag_diabetes":       "MONDO_0005148",
    "flag_hypertension":   "MONDO_0001134",
    "flag_hyperlipidemia": "MONDO_0021187",
    "flag_metabolic_syn":  "MONDO_0015626",
    "flag_ckd":            "MONDO_0005300",
    "flag_afib":           "MONDO_0004981",
    "flag_obesity":        "MONDO_0011122",
    "flag_osteoporosis":   "MONDO_0005298",
}

# Condition keywords → binary flag column names
_CONDITION_FLAGS = {
    "flag_obesity":          "obesity",
    "flag_prediabetes":      "prediabetes",
    "flag_diabetes":         "diabetes",
    "flag_hypertension":     "hypertension",
    "flag_hyperlipidemia":   "hyperlipidemia",
    "flag_chd":              "coronary heart disease",
    "flag_stroke":           "stroke",
    "flag_afib":             "atrial fibrillation",
    "flag_metabolic_syn":    "metabolic syndrome",
    "flag_alzheimers":       "alzheimer",
    "flag_osteoporosis":     "osteoporosis",
    "flag_ckd":              "chronic kidney disease",
    "flag_copd":             "chronic obstructive pulmonary disease",
    "flag_depression":       "depression",
    "flag_asthma":           "asthma",
}

# Genes to aggregate variant burden from DNA files
# (Pathogenic / Likely Pathogenic / Risk Factor variants)
_RELEVANT_GENES = [
    "PCSK9", "APOE", "APOB", "LDLR", "HMGCR",   # cholesterol / CVD
    "ACE", "AGT", "AGTR1", "ADRB1", "ADRB2",     # hypertension / cardiac
    "ADIPOQ", "PPARG", "TCF7L2", "KCNJ11",        # T2D / metabolic
    "APP", "PSEN1", "PSEN2", "CLU", "BIN1",       # Alzheimer's
    "BRCA1", "BRCA2", "TP53",                      # oncology
]

_RISK_SIGS = {"Pathogenic", "Likely Pathogenic", "Risk Factor"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_coherent_cohort(
    data_dir: str | Path,
    n_endotypes: int = 5,
    client: Optional[OpenTargetsClient] = None,
    max_dna_files: int = 500,
) -> PatientCohortContext:
    """
    Build a PatientCohortContext from the Coherent EHR dataset directory.

    Parameters
    ----------
    data_dir:
        Root of the Coherent dataset (contains csv/ and dna/ subdirs).
    n_endotypes:
        Force a fixed number of endotypes. None = let HDBSCAN choose.
    client:
        Optional pre-built OpenTargets client.
    max_dna_files:
        Cap on DNA files to read (keeps startup fast; 500 ≈ 56% of cohort).
    """
    data_dir = Path(data_dir)
    ot = client or OpenTargetsClient()

    feature_matrix = _build_feature_matrix(data_dir, max_dna_files=max_dna_files)
    endotyping = discover_endotypes(feature_matrix, n_clusters=n_endotypes)

    candidates = _discover_candidates(feature_matrix, endotyping, ot)

    return PatientCohortContext(
        source_path=str(data_dir),
        feature_matrix=feature_matrix,
        endotyping=endotyping,
        candidate_targets=candidates,
    )


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def _build_feature_matrix(data_dir: Path, max_dna_files: int) -> pd.DataFrame:
    pts   = _load_demographics(data_dir)
    labs  = _load_labs(data_dir)
    flags = _load_condition_flags(data_dir)
    genes = _load_genetic_burden(data_dir, max_dna_files)

    df = pts.join(labs, how="left").join(flags, how="left").join(genes, how="left")
    df = df.fillna(0.0)

    # Drop near-zero-variance columns (e.g. rare conditions with <2% prevalence)
    prev = df.mean()
    keep = prev[(prev > 0.02) | (prev.index.str.startswith(("bmi", "systolic", "diastolic",
                                                              "cholesterol", "ldl", "hdl",
                                                              "triglycerides", "hba1c",
                                                              "glucose", "creatinine",
                                                              "age", "sex")))]
    df = df[[c for c in df.columns if c in keep.index or not c.startswith("flag_")]]

    return df.select_dtypes(include=[np.number])


def _load_demographics(data_dir: Path) -> pd.DataFrame:
    pts = pd.read_csv(data_dir / "csv" / "patients.csv", index_col="Id")
    pts["age"] = pd.to_datetime("today").year - pd.to_datetime(pts["BIRTHDATE"]).dt.year
    pts["sex_male"] = (pts["GENDER"] == "M").astype(float)
    # Simple 4-group race encoding
    race_map = {"white": 0, "black": 1, "asian": 2, "hispanic": 3}
    pts["race_group"] = pts["RACE"].str.lower().map(race_map).fillna(4)
    return pts[["age", "sex_male", "race_group"]]


def _load_labs(data_dir: Path) -> pd.DataFrame:
    obs = pd.read_csv(data_dir / "csv" / "observations.csv")
    obs = obs[obs["TYPE"] == "numeric"].copy()
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")

    frames = []
    for col_name, description_substr in _LAB_MAP.items():
        sub = obs[obs["DESCRIPTION"].str.contains(description_substr, case=False, na=False)]
        if sub.empty:
            continue
        median_per_patient = (
            sub.groupby("PATIENT")["VALUE"]
            .median()
            .rename(col_name)
        )
        frames.append(median_per_patient)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _load_condition_flags(data_dir: Path) -> pd.DataFrame:
    conds = pd.read_csv(data_dir / "csv" / "conditions.csv")

    frames = []
    for col_name, keyword in _CONDITION_FLAGS.items():
        mask = conds["DESCRIPTION"].str.lower().str.contains(keyword, na=False)
        patients_with = conds[mask]["PATIENT"].unique()
        flag = pd.Series(1.0, index=patients_with, name=col_name)
        frames.append(flag)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _load_genetic_burden(data_dir: Path, max_files: int) -> pd.DataFrame:
    """
    One row per patient: for each relevant gene, 1.0 if the patient carries
    a Pathogenic / Likely Pathogenic / Risk Factor variant, else 0.0.
    """
    dna_dir = data_dir / "dna"
    if not dna_dir.exists():
        return pd.DataFrame()

    files = sorted(dna_dir.glob("*_dna.csv"))[:max_files]
    rows: dict[str, dict[str, float]] = {}

    for fpath in files:
        # Extract patient UUID from filename: Name_UUID_dna.csv
        # The UUID is always the last segment before _dna.csv
        stem = fpath.stem  # e.g. "Abe604_Frami345_b8dd1798-beef-094d-1be4-f90ee0e6b7d5_dna"
        # UUID = last token split by _
        parts = stem.replace("_dna", "").rsplit("_", 1)
        patient_id = parts[-1]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dna = pd.read_csv(fpath, low_memory=False)
        except Exception:
            continue

        # VARIANT==True means the patient actually carries the alternate allele
        carrier = dna[
            dna["CLINICAL_SIGNIFICANCE"].isin(_RISK_SIGS)
            & (dna["VARIANT"].astype(str).str.lower() == "true")
        ]
        gene_flags: dict[str, float] = {}
        for gene in _RELEVANT_GENES:
            has_variant = (carrier["GENE"] == gene).any()
            gene_flags[f"gene_{gene}"] = 1.0 if has_variant else 0.0

        rows[patient_id] = gene_flags

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Candidate target discovery
# ---------------------------------------------------------------------------

def _discover_candidates(
    feature_matrix: pd.DataFrame,
    endotyping,
    ot: OpenTargetsClient,
) -> list[CandidateTarget]:
    """
    For each endotype, find the most differentially enriched features
    (genes + condition flags), map genes to OpenTargets, return candidates.
    """
    candidates: list[CandidateTarget] = []
    labels = endotyping.labels

    for cluster_id in sorted(labels.unique()):
        if cluster_id == -1:
            continue

        cluster_mask = labels == cluster_id
        cluster_df   = feature_matrix[cluster_mask]
        rest_df      = feature_matrix[~cluster_mask]

        # Top differentially enriched gene columns
        gene_cols = [c for c in feature_matrix.columns if c.startswith("gene_")]
        top_genes = _top_features(cluster_df, rest_df, gene_cols, top_n=15)

        # Strip "gene_" prefix → HGNC symbols
        gene_symbols = [g.replace("gene_", "") for g in top_genes]

        # Top enriched condition flags (to label the endotype)
        flag_cols = [c for c in feature_matrix.columns if c.startswith("flag_")]
        top_flags = _top_features(cluster_df, rest_df, flag_cols, top_n=3)
        endotype_label = _flags_to_label(top_flags, cluster_id)

        for gene_symbol in gene_symbols:
            ensembl_id = ot.resolve_gene_symbol(gene_symbol, silent=True)
            if ensembl_id is None:
                continue
            scores  = ot.get_association_scores(ensembl_id)
            novelty = ot.get_novelty_score(ensembl_id)
            candidates.append(CandidateTarget(
                ensembl_id=ensembl_id,
                gene_symbol=gene_symbol,
                endotype_id=int(cluster_id),
                endotype_label=endotype_label,
                association_score=scores.get("overall", 0.0),
                genetic_association_score=scores.get("genetics", 0.0),
                expression_specificity_score=scores.get("expression", 0.0),
                novelty_score=novelty,
            ))

        # Supplement with top targets for the most enriched disease in this endotype
        top_flags_all = _top_features(cluster_df, rest_df, flag_cols, top_n=5)
        disease_id = next(
            (_FLAG_TO_DISEASE_ID[f] for f in top_flags_all if f in _FLAG_TO_DISEASE_ID),
            None,
        )
        if disease_id:
            try:
                landscape = ot.get_disease_competitive_landscape(disease_id, size=6)
                seen = {c.ensembl_id for c in candidates if c.endotype_id == int(cluster_id)}
                for _, row in landscape.head(6).iterrows():
                    if row["ensembl_id"] in seen:
                        continue
                    seen.add(row["ensembl_id"])
                    novelty = ot.get_novelty_score(row["ensembl_id"])
                    candidates.append(CandidateTarget(
                        ensembl_id=row["ensembl_id"],
                        gene_symbol=row["gene_symbol"],
                        endotype_id=int(cluster_id),
                        endotype_label=endotype_label,
                        association_score=row["association_score"],
                        genetic_association_score=row.get("score_genetic_association", 0.0),
                        expression_specificity_score=0.0,
                        novelty_score=novelty,
                    ))
            except Exception:
                pass

    return candidates


def _top_features(
    cluster_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    cols: list[str],
    top_n: int,
) -> list[str]:
    """Columns with highest mean in this cluster relative to the rest."""
    if not cols:
        return []
    avail = [c for c in cols if c in cluster_df.columns]
    if not avail:
        return []
    cluster_mean = cluster_df[avail].mean()
    rest_mean    = rest_df[avail].mean().replace(0, np.nan)
    fold_change  = (cluster_mean / rest_mean).dropna().sort_values(ascending=False)
    return fold_change.head(top_n).index.tolist()


def _flags_to_label(top_flags: list[str], cluster_id: int) -> str:
    """Turn flag column names into a human-readable endotype label."""
    _label_map = {
        "flag_obesity":        "Obesity",
        "flag_prediabetes":    "Prediabetes",
        "flag_diabetes":       "T2 Diabetes",
        "flag_hypertension":   "Hypertension",
        "flag_hyperlipidemia": "Hyperlipidaemia",
        "flag_chd":            "Coronary HD",
        "flag_stroke":         "Stroke",
        "flag_afib":           "AF",
        "flag_metabolic_syn":  "Metabolic Syndrome",
        "flag_alzheimers":     "Alzheimer's",
        "flag_osteoporosis":   "Osteoporosis",
        "flag_ckd":            "CKD",
        "flag_copd":           "COPD",
        "flag_depression":     "Depression",
        "flag_asthma":         "Asthma",
    }
    labels = [_label_map.get(f, f.replace("flag_", "").replace("_", " ").title())
              for f in top_flags]
    suffix = " / ".join(labels) if labels else f"Subgroup {cluster_id + 1}"
    return f"Endotype {cluster_id + 1}: {suffix}"
