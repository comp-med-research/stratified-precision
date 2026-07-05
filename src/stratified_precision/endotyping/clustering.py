"""
Disease endotyping via UMAP dimensionality reduction + HDBSCAN clustering.

The goal is not just "find clusters" — it's to identify patient subgroups
with meaningfully different biology so that target analysis is grounded in
the right population.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan


@dataclass
class EndotypingResult:
    """Output of the endotyping step."""
    labels: pd.Series             # cluster assignment per sample (-1 = noise)
    umap_coords: np.ndarray       # shape (n_samples, 2) — for visualisation
    n_clusters: int
    cluster_summary: pd.DataFrame # per-cluster feature means


def discover_endotypes(
    feature_matrix: pd.DataFrame,
    n_clusters: Optional[int] = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    hdbscan_min_cluster_size: int = 10,
    random_state: int = 42,
) -> EndotypingResult:
    """
    Run UMAP → HDBSCAN to discover disease endotypes.

    Parameters
    ----------
    feature_matrix:
        Numeric DataFrame, one row per sample. Can be clinical variables,
        gene expression values, PCA scores from omics, etc.
    n_clusters:
        If provided, falls back to KMeans with this many clusters instead
        of letting HDBSCAN choose automatically.
    """
    n_samples = len(feature_matrix)
    X = StandardScaler().fit_transform(feature_matrix.values)
    # Zero-variance columns → NaN/inf after scaling; near-zero variance → huge finite
    # values that overflow float64 when squared during KMeans distance computation.
    # nan_to_num first, then hard-clip to ±5σ (no real clinical z-score exceeds this).
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -5.0, 5.0)

    # Scale UMAP n_neighbors to dataset size — must be < n_samples
    effective_neighbors = min(umap_n_neighbors, max(2, n_samples - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
        metric="euclidean",
    )
    coords = reducer.fit_transform(X)

    import warnings as _warnings
    if n_clusters is not None:
        from sklearn.cluster import KMeans
        k = min(n_clusters, n_samples - 1)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            labels = KMeans(n_clusters=k, random_state=random_state,
                            n_init=10, init="random").fit_predict(X)
    elif n_samples < 20:
        # Too few samples for HDBSCAN — use KMeans with 2-3 clusters
        from sklearn.cluster import KMeans
        k = min(3, max(2, n_samples // 5))
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            labels = KMeans(n_clusters=k, random_state=random_state,
                            n_init=10, init="random").fit_predict(X)
    else:
        # Scale min_cluster_size: at least 1% of samples (better for large datasets),
        # at least the caller's hint, never more than 5% of samples.
        pct_based = max(hdbscan_min_cluster_size, n_samples // 100)
        effective_min_size = min(pct_based, n_samples // 20)
        effective_min_size = max(effective_min_size, 3)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_size,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(coords)
        # If HDBSCAN assigns everything to noise, fall back to KMeans
        if (labels >= 0).sum() == 0:
            from sklearn.cluster import KMeans
            k = min(3, max(2, n_samples // 8))
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                labels = KMeans(n_clusters=k, random_state=random_state,
                                n_init=10, init="random").fit_predict(X)

    labels_series = pd.Series(labels, index=feature_matrix.index, name="endotype")

    summary = _compute_cluster_summary(feature_matrix, labels_series)

    return EndotypingResult(
        labels=labels_series,
        umap_coords=coords,
        n_clusters=int(labels_series[labels_series >= 0].nunique()),
        cluster_summary=summary,
    )


def _compute_cluster_summary(
    df: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Mean feature values per cluster — useful for biological interpretation."""
    df_labeled = df.copy()
    df_labeled["_cluster"] = labels
    summary = df_labeled[df_labeled["_cluster"] >= 0].groupby("_cluster").mean()
    return summary
