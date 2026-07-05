"""
Disk-backed cache for PipelineResult objects.

Results are pickled to ~/.stratified_precision_cache/ keyed by
(mode, query, disease_id).  Any run that hits a warm cache is instant.

To pre-bake demo examples before a presentation:
    python scripts/prebake_demos.py
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

_CACHE_DIR = Path(os.path.expanduser("~/.stratified_precision_cache"))


def cache_key(mode: str, query: str, disease_id: str = "") -> str:
    return f"{mode}:{query.lower().strip()}:{disease_id.lower().strip()}"


def _key_to_path(key: str) -> Path:
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    safe_label = key.replace(":", "_").replace("/", "_")[:40]
    return _CACHE_DIR / f"{safe_label}_{digest}.pkl"


def load_result(key: str):
    """Return cached PipelineResult or None if not cached / corrupt."""
    path = _key_to_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[result_store] Cache miss (corrupt): {path.name} — {exc}")
        path.unlink(missing_ok=True)
        return None


def save_result(key: str, result) -> None:
    """Persist a PipelineResult to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _key_to_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[result_store] Saved: {path.name}")
    except Exception as exc:
        print(f"[result_store] Save failed: {exc}")


def list_cached() -> list[str]:
    """Return the label portion of every cached result filename."""
    if not _CACHE_DIR.exists():
        return []
    return [p.stem for p in sorted(_CACHE_DIR.glob("*.pkl"))]
