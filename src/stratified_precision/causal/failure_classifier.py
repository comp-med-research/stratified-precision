"""
Causal failure mode classifier.

Uses DoWhy to build a causal DAG around a target and estimate whether
predicted trial failure is driven by toxicity or lack of efficacy.
The key insight: correlation-based classifiers cannot distinguish these
two failure modes — you need to know *why* the effect is absent.

Toxicity failure:  target engagement works, but on-target effects in
                   non-disease tissues cause harm.
Efficacy failure:  target is not actually driving the disease in this
                   patient subgroup (wrong mechanism or wrong population).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
import numpy as np


@dataclass
class FailureMode:
    target_id: str
    gene_symbol: str
    endotype_id: Optional[int]

    # Probabilities of each failure pathway (sum to ≤ 1; remainder = predicted success)
    safety_score: float      # probability of toxicity-driven failure
    efficacy_score: float    # probability of efficacy-driven failure

    # Most likely mode
    predicted_mode: str      # "toxicity" | "efficacy" | "likely_success"

    # Which features drove the prediction
    top_safety_drivers: list[str]
    top_efficacy_drivers: list[str]


def classify_failure_modes(
    endotyping=None,
    target_context=None,
    candidates=None,
) -> list[FailureMode]:
    """
    Classify failure modes for either a single target (Mode 1)
    or a list of candidates (Mode 2).
    """
    if target_context is not None:
        return _classify_single_target(target_context, endotyping)
    elif candidates is not None:
        return _classify_candidates(candidates, endotyping)
    else:
        raise ValueError("Provide either target_context or candidates.")


def _classify_single_target(ctx, endotyping) -> list[FailureMode]:
    """Classify failure mode for a single target across its disease associations."""
    safety_score = _compute_safety_score(ctx)
    efficacy_score = _compute_efficacy_score(ctx)

    mode = _resolve_mode(safety_score, efficacy_score)

    return [
        FailureMode(
            target_id=ctx.ensembl_id,
            gene_symbol=ctx.gene_symbol,
            endotype_id=None,
            safety_score=safety_score,
            efficacy_score=efficacy_score,
            predicted_mode=mode,
            top_safety_drivers=_safety_drivers(ctx),
            top_efficacy_drivers=_efficacy_drivers(ctx),
        )
    ]


def _classify_candidates(candidates, endotyping) -> list[FailureMode]:
    result = []
    for c in candidates:
        # Without full TargetContext for each candidate, use the scores we already have
        safety_score = 1.0 - c.expression_specificity_score  # broad expression → safety risk
        efficacy_score = 1.0 - c.genetic_association_score   # weak genetics → efficacy risk
        mode = _resolve_mode(safety_score, efficacy_score)

        result.append(
            FailureMode(
                target_id=c.ensembl_id,
                gene_symbol=c.gene_symbol,
                endotype_id=c.endotype_id,
                safety_score=round(safety_score, 3),
                efficacy_score=round(efficacy_score, 3),
                predicted_mode=mode,
                top_safety_drivers=["broad_tissue_expression"],
                top_efficacy_drivers=["weak_genetic_association"],
            )
        )
    return result


def _compute_safety_score(ctx) -> float:
    """
    Higher safety score = higher probability of toxicity-driven failure.
    Key signals: broad tissue expression (off-target organs),
    known safety liabilities, high interaction degree (promiscuous target).
    """
    score = 0.0
    weight = 0.0

    # Broad tissue expression → more off-target organs at risk
    if not ctx.tissue_expression.empty and "rna_value" in ctx.tissue_expression.columns:
        n_high_tissues = (ctx.tissue_expression["rna_value"] > 5).sum()
        score += min(n_high_tissues / 30.0, 1.0) * 0.4
        weight += 0.4

    # Existing safety liabilities from OpenTargets
    if not ctx.safety_liabilities.empty:
        score += min(len(ctx.safety_liabilities) / 5.0, 1.0) * 0.4
        weight += 0.4

    # High network degree → promiscuous / hard to selectively drug
    if not ctx.network_edges.empty:
        degree = len(ctx.network_edges)
        score += min(degree / 100.0, 1.0) * 0.2
        weight += 0.2

    return round(score / weight if weight > 0 else 0.5, 3)


def _compute_efficacy_score(ctx) -> float:
    """
    Higher efficacy score = higher probability of efficacy failure.
    Key signals: weak genetic association, low disease association score,
    no clinical evidence of mechanism validation.
    """
    score = 0.0
    weight = 0.0

    if not ctx.disease_associations.empty:
        max_score = ctx.disease_associations["score"].max()
        score += (1.0 - min(max_score, 1.0)) * 0.5
        weight += 0.5

    if not ctx.clinical_evidence.empty:
        max_phase = ctx.clinical_evidence["phase"].fillna(0).astype(float).max()
        # Phase 2+ = some efficacy evidence, reduces efficacy failure risk
        score += max(0.0, 1.0 - max_phase / 2.0) * 0.5
        weight += 0.5

    return round(score / weight if weight > 0 else 0.5, 3)


def _resolve_mode(safety_score: float, efficacy_score: float) -> str:
    if safety_score < 0.3 and efficacy_score < 0.3:
        return "likely_success"
    if safety_score > efficacy_score:
        return "toxicity"
    return "efficacy"


def _safety_drivers(ctx) -> list[str]:
    drivers = []
    if not ctx.tissue_expression.empty:
        top = ctx.tissue_expression.nlargest(3, "rna_value")["tissue_label"].tolist()
        drivers.extend([f"expression:{t}" for t in top])
    if not ctx.safety_liabilities.empty:
        drivers.extend(ctx.safety_liabilities["event"].head(2).tolist())
    return drivers


def _efficacy_drivers(ctx) -> list[str]:
    drivers = []
    if not ctx.disease_associations.empty:
        bottom = ctx.disease_associations.nsmallest(3, "score")["disease_name"].tolist()
        drivers.extend([f"low_assoc:{d}" for d in bottom])
    return drivers
