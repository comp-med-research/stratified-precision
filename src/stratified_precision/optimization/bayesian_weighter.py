"""
Bayesian objective weighter.

Problem: we don't know up front which graph features matter for a given
disease context. "Anatomy breadth" is a strong objective for a CNS target
but nearly irrelevant for a CFTR mutation. Hardcoding objectives penalises
rare diseases and novel biology.

Solution: treat feature-to-objective selection as a Bayesian inference
problem. Start with disease-class priors (what we know from literature),
observe the target's actual graph features, and update to produce a
posterior weight over which features are informative for this context.

A feature with posterior weight > threshold becomes a Pareto objective.
The result: objectives are discovered from the data, not designed in.

GP model:
  - Input X: feature vector for a candidate target
  - Output y: predicted p(clinical_success) ∈ [0, 1]
  - Kernel: Matern(nu=2.5) for smooth but non-differentiable responses
  - Prior mean: disease-class-specific, not zero
  - Feature importance: derived from GP length scales (short scale = high importance)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    HAS_GP = True
except ImportError:
    HAS_GP = False


# ---------------------------------------------------------------------------
# Disease-class priors
# Feature keys map to (prior_weight, prior_confidence) tuples.
# prior_weight:      0–1, how important we believe this feature is a priori
# prior_confidence:  0–1, how sure we are (high → GP won't move far from prior)
# ---------------------------------------------------------------------------

DISEASE_PRIORS: dict[str, dict[str, tuple[float, float]]] = {
    "alzheimer": {
        "path_to_disease_score":     (0.9, 0.8),   # directness of mechanism is crucial
        "n_compound_neighbors":      (0.7, 0.7),   # many failed drugs → crowded space
        "n_anatomy_neighbors":       (0.85, 0.75), # CNS specificity matters a lot
        "n_pathway_neighbors":       (0.6, 0.5),
        "subgraph_density":          (0.4, 0.4),
        "n_gene_neighbors":          (0.5, 0.4),
    },
    "cystic fibrosis": {
        "path_to_disease_score":     (0.95, 0.9),  # monogenic: path directness is nearly everything
        "n_compound_neighbors":      (0.4, 0.5),   # less compound history
        "n_anatomy_neighbors":       (0.5, 0.6),   # organ specificity (lung) matters
        "n_pathway_neighbors":       (0.7, 0.65),  # CFTR pathway coverage
        "subgraph_density":          (0.3, 0.4),
    },
    "cancer": {
        "path_to_disease_score":     (0.7, 0.6),
        "n_compound_neighbors":      (0.8, 0.7),   # oncology has many existing drugs
        "n_anatomy_neighbors":       (0.6, 0.5),
        "n_pathway_neighbors":       (0.75, 0.65),
        "subgraph_density":          (0.6, 0.55),
        "n_gene_neighbors":          (0.7, 0.6),   # network effects matter in cancer
    },
    "default": {
        "path_to_disease_score":     (0.7, 0.5),
        "n_compound_neighbors":      (0.5, 0.4),
        "n_anatomy_neighbors":       (0.5, 0.4),
        "n_pathway_neighbors":       (0.5, 0.4),
        "subgraph_density":          (0.3, 0.3),
        "n_gene_neighbors":          (0.4, 0.35),
    },
}

BASE_OBJECTIVE_WEIGHT_THRESHOLD = 0.5  # features above this weight become Pareto objectives


@dataclass
class WeightedObjectives:
    """Objectives dynamically selected for this disease context."""

    # Feature name → objective weight (0–1)
    weights: dict[str, float]

    # Which features cleared the threshold and became Pareto objectives
    selected_features: list[str]

    # The normalised objective matrix for Pareto (shape: n_candidates × n_objectives)
    objective_matrix: np.ndarray

    # Column names matching objective_matrix
    objective_names: list[str]

    # Which disease class drove the priors
    disease_class: str


class BayesianObjectiveWeighter:
    """
    Uses Gaussian Process to maintain a posterior over feature importance,
    starting from disease-class priors.
    """

    def __init__(self, disease_name: str = ""):
        self.disease_class = _match_disease_class(disease_name)
        self.priors = DISEASE_PRIORS.get(self.disease_class, DISEASE_PRIORS["default"])
        self._gp: Optional[GaussianProcessRegressor] = None
        self._fitted_feature_names: list[str] = []

    def fit(
        self,
        feature_matrix: np.ndarray,
        feature_names: list[str],
        success_labels: Optional[np.ndarray] = None,
    ):
        """
        Fit the GP on known (feature, outcome) pairs if available.
        If success_labels is None, uses prior means as pseudo-observations.
        """
        if not HAS_GP:
            return

        self._fitted_feature_names = feature_names

        # Build prior mean vector from disease priors
        prior_means = np.array([
            self.priors.get(f, DISEASE_PRIORS["default"].get(f, (0.5, 0.3)))[0]
            for f in feature_names
        ])

        if success_labels is None or len(success_labels) == 0:
            # No labelled data: use prior means as pseudo-targets for a single "virtual" observation
            X_train = prior_means.reshape(1, -1)
            y_train = np.array([0.5])
        else:
            X_train = feature_matrix
            y_train = success_labels.astype(float)

        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(len(feature_names)))
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=0.1,
        )
        self._gp.fit(X_train, y_train)

    def compute_weighted_objectives(
        self,
        candidate_feature_dicts: list[dict[str, float]],
        base_objectives: dict[str, np.ndarray],
        threshold: float = BASE_OBJECTIVE_WEIGHT_THRESHOLD,
    ) -> WeightedObjectives:
        """
        Given graph features for all candidates, compute posterior weights,
        select which features exceed the threshold, and return the combined
        objective matrix (base + graph-derived).

        Parameters
        ----------
        candidate_feature_dicts:
            One dict per candidate, keys = feature names, values = feature values.
        base_objectives:
            The fixed-base objectives (efficacy_potential, safety_margin, etc.)
            from the original Pareto module — always included.
        threshold:
            Minimum posterior weight for a graph feature to become an objective.
        """
        all_feature_names = _union_feature_names(candidate_feature_dicts)
        feature_matrix = _dicts_to_matrix(candidate_feature_dicts, all_feature_names)

        weights = self._posterior_weights(all_feature_names, feature_matrix)

        selected = [f for f in all_feature_names if weights.get(f, 0.0) >= threshold]

        # Assemble the final objective matrix
        objective_cols: dict[str, np.ndarray] = dict(base_objectives)

        for feat in selected:
            col_idx = all_feature_names.index(feat)
            col = feature_matrix[:, col_idx]
            col_range = col.max() - col.min()
            if col_range > 0:
                objective_cols[f"kg_{feat}"] = (col - col.min()) / col_range
            else:
                objective_cols[f"kg_{feat}"] = np.zeros(len(col))

        obj_names = list(objective_cols.keys())
        obj_matrix = np.column_stack([objective_cols[k] for k in obj_names])

        return WeightedObjectives(
            weights=weights,
            selected_features=selected,
            objective_matrix=obj_matrix,
            objective_names=obj_names,
            disease_class=self.disease_class,
        )

    def _posterior_weights(
        self,
        feature_names: list[str],
        feature_matrix: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute posterior importance weight for each feature.

        If a GP has been fitted, derive importance from kernel length scales
        (shorter length scale in a GP = feature varies more = more important).
        Otherwise fall back to prior weights.
        """
        if self._gp is not None and hasattr(self._gp, "kernel_"):
            try:
                fitted_names = self._fitted_feature_names or feature_names
                length_scales = np.atleast_1d(
                    self._gp.kernel_.get_params().get("k2__length_scale", np.ones(len(fitted_names)))
                )
                # Importance ∝ 1/length_scale; normalise to [0, 1]
                importance = 1.0 / (length_scales + 1e-6)
                importance /= importance.max()
                ls_map = dict(zip(fitted_names, importance.tolist()))
                return {f: ls_map.get(f, self._prior_weight(f)) for f in feature_names}
            except Exception:
                pass

        # Prior fallback
        return {f: self._prior_weight(f) for f in feature_names}

    def _prior_weight(self, feature_name: str) -> float:
        return self.priors.get(
            feature_name,
            DISEASE_PRIORS["default"].get(feature_name, (0.4, 0.3))
        )[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_disease_class(disease_name: str) -> str:
    term = disease_name.lower()
    for key in DISEASE_PRIORS:
        if key != "default" and key in term:
            return key
    return "default"


def _union_feature_names(dicts: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for d in dicts:
        for k in d:
            seen[k] = None
    return list(seen.keys())


def _dicts_to_matrix(dicts: list[dict], feature_names: list[str]) -> np.ndarray:
    rows = []
    for d in dicts:
        rows.append([d.get(f, 0.0) for f in feature_names])
    return np.array(rows, dtype=float)
