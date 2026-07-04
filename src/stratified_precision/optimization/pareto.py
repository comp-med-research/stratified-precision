"""
Multi-objective Pareto optimisation for target ranking.

Base objectives (always present):
  1. Efficacy potential    = association_score × (1 - efficacy_score)
  2. Safety margin         = 1 - safety_score
  3. Tissue specificity    = expression_specificity_score
  4. Novelty               = novelty_score (no approved drugs yet)

Dynamic objectives (disease-context-dependent, discovered by BayesianObjectiveWeighter):
  e.g. kg_path_to_disease_score, kg_n_anatomy_neighbors, kg_n_compound_neighbors
  These are appended at runtime — the set is unknown at design time.

A target on the Pareto front is not dominated by any other candidate
on ALL objectives simultaneously. With dynamic objectives, what "not dominated"
means changes per disease — which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

try:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False


@dataclass
class ParetoResult:
    """Pareto front analysis results."""
    ranks: np.ndarray      # Pareto rank per row (1 = Pareto front)
    on_front: np.ndarray   # Boolean mask — True if on first Pareto front
    objective_matrix: pd.DataFrame  # All objectives used (base + dynamic KG features)
    objective_names: list[str] = field(default_factory=list)  # human-readable labels


def compute_pareto_front(
    df: pd.DataFrame,
    extra_objectives: Optional[dict[str, np.ndarray]] = None,
) -> ParetoResult:
    """
    Compute Pareto ranks for all candidates in the DataFrame.

    Parameters
    ----------
    df:
        Candidate DataFrame with base score columns.
    extra_objectives:
        Dict of {name: array} for dynamically discovered objectives from the
        Bayesian weighter / KG features. Added on top of the base four.
        Unknown at design time — depends on disease context.
    """
    def _get(col):
        return df[col].fillna(0.5).values if col in df.columns else np.full(len(df), 0.5)

    base = {
        "efficacy_potential": _get("association_score") * (1 - _get("efficacy_score")),
        "safety_margin":      1 - _get("safety_score"),
        "tissue_specificity": _get("expression_specificity_score"),
        "novelty":            _get("novelty_score"),
    }

    if extra_objectives:
        base.update(extra_objectives)

    obj_df = pd.DataFrame(base)

    if HAS_PYMOO:
        ranks, on_front = _pareto_ranks_pymoo(obj_df.values)
    else:
        ranks, on_front = _pareto_ranks_simple(obj_df.values)

    return ParetoResult(
        ranks=ranks,
        on_front=on_front,
        objective_matrix=obj_df,
        objective_names=list(obj_df.columns),
    )


def _pareto_ranks_pymoo(F: np.ndarray):
    # pymoo minimises, so negate (we want to maximise all objectives)
    nds = NonDominatedSorting()
    fronts = nds.do(-F)
    ranks = np.zeros(len(F), dtype=int)
    for rank, front in enumerate(fronts, start=1):
        ranks[front] = rank
    on_front = ranks == 1
    return ranks, on_front


def _pareto_ranks_simple(F: np.ndarray):
    """Fallback non-dominated sort if pymoo isn't installed."""
    n = len(F)
    ranks = np.ones(n, dtype=int)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= i on all objectives and > on at least one
            if np.all(F[j] >= F[i]) and np.any(F[j] > F[i]):
                dominated[i] = True
                break

    on_front = ~dominated
    ranks[~on_front] = 2
    return ranks, on_front
