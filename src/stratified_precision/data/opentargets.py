"""
OpenTargets GraphQL API client.

Docs: https://platform-docs.opentargets.org/data-access/graphql-api
GraphQL playground: https://api.platform.opentargets.org/api/v4/graphql/browser
"""

from __future__ import annotations

import time
from typing import Optional
import requests
import pandas as pd

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_PHASE_MAP = {
    "PHASE_1": 1, "PHASE_2": 2, "PHASE_3": 3, "PHASE_4": 4,
    "APPROVED": 4, "CLINICAL_STAGE_I": 1, "CLINICAL_STAGE_II": 2,
    "CLINICAL_STAGE_III": 3, "CLINICAL_STAGE_IV": 4,
}


def _parse_phase(phase_str: str) -> float:
    return float(_PHASE_MAP.get((phase_str or "").upper(), 0))


class OpenTargetsClient:
    def __init__(self, url: str = GRAPHQL_URL, max_retries: int = 3):
        self.url = url
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Gene resolution
    # ------------------------------------------------------------------

    def resolve_gene_symbol(self, symbol: str, silent: bool = False) -> Optional[str]:
        """Return the Ensembl gene ID for a HGNC symbol, or None if not found."""
        query = """
        query resolveGene($symbol: String!) {
          search(queryString: $symbol, entityNames: ["target"]) {
            hits {
              id
              name
              entity
            }
          }
        }
        """
        data = self._query(query, {"symbol": symbol})
        hits = data.get("search", {}).get("hits", [])
        for hit in hits:
            if hit.get("name", "").upper() == symbol.upper():
                return hit["id"]
        if not silent:
            print(f"[OpenTargets] Could not resolve symbol: {symbol}")
        return None

    # ------------------------------------------------------------------
    # Target data
    # ------------------------------------------------------------------

    def get_disease_associations(self, ensembl_id: str) -> pd.DataFrame:
        query = """
        query targetDiseases($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            associatedDiseases {
              rows {
                disease { id name }
                score
                datatypeScores { id score }
              }
            }
          }
        }
        """
        data = self._query(query, {"ensemblId": ensembl_id})
        rows_raw = (data.get("target") or {}).get("associatedDiseases", {}).get("rows", [])

        records = []
        for r in rows_raw:
            rec = {
                "disease_id": r["disease"]["id"],
                "disease_name": r["disease"]["name"],
                "score": r["score"],
            }
            for ds in r.get("datatypeScores", []):
                rec[f"score_{ds['id']}"] = ds["score"]
            records.append(rec)

        return pd.DataFrame(records)

    def get_tissue_expression(self, ensembl_id: str) -> pd.DataFrame:
        query = """
        query tissueExpression($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            expressions {
              tissue { id label }
              rna { value zscore }
              protein { level }
            }
          }
        }
        """
        data = self._query(query, {"ensemblId": ensembl_id})
        exprs = (data.get("target") or {}).get("expressions", [])

        records = [
            {
                "tissue_id": e["tissue"]["id"],
                "tissue_label": e["tissue"]["label"],
                "rna_value": e["rna"]["value"],
                "rna_zscore": e["rna"]["zscore"],
                "protein_level": e["protein"]["level"],
            }
            for e in exprs
        ]
        return pd.DataFrame(records)

    def get_network_edges(self, ensembl_id: str, size: int = 50) -> pd.DataFrame:
        query = """
        query interactions($ensemblId: String!, $size: Int!) {
          target(ensemblId: $ensemblId) {
            interactions(page: {size: $size, index: 0}) {
              rows {
                targetB { id approvedSymbol }
                score
                sourceDatabase
              }
            }
          }
        }
        """
        data = self._query(query, {"ensemblId": ensembl_id, "size": size})
        rows_raw = (data.get("target") or {}).get("interactions", {}).get("rows", [])

        records = [
            {
                "source_id": ensembl_id,
                "target_id": r["targetB"]["id"],
                "target_symbol": r["targetB"]["approvedSymbol"],
                "score": r["score"],
                "source_db": r["sourceDatabase"],
            }
            for r in rows_raw
            if r.get("targetB")
        ]
        return pd.DataFrame(records)

    def get_clinical_evidence(self, ensembl_id: str) -> pd.DataFrame:
        query = """
        query clinicalEvidence($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            drugAndClinicalCandidates {
              rows {
                drug { id name drugType maximumClinicalStage }
                maxClinicalStage
                diseases {
                  disease { id name }
                  diseaseFromSource
                }
              }
            }
          }
        }
        """
        data = self._query(query, {"ensemblId": ensembl_id})
        rows_raw = (data.get("target") or {}).get("drugAndClinicalCandidates", {}).get("rows", [])

        records = []
        for r in rows_raw:
            drug = r["drug"]
            phase_str = r.get("maxClinicalStage") or ""
            phase_num = _parse_phase(phase_str)
            for d in r.get("diseases") or []:
                disease = d.get("disease") or {}
                records.append({
                    "drug_id": drug["id"],
                    "drug_name": drug["name"],
                    "drug_type": drug.get("drugType"),
                    "disease_id": disease.get("id", ""),
                    "disease_name": disease.get("name") or d.get("diseaseFromSource", ""),
                    "phase": phase_num,
                    "status": phase_str,
                })

        return pd.DataFrame(records)

    def get_safety_liabilities(self, ensembl_id: str) -> pd.DataFrame:
        query = """
        query safetyLiabilities($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            safetyLiabilities {
              event
              effects { dosing direction }
              biosamples { tissueLabel }
              datasource
            }
          }
        }
        """
        data = self._query(query, {"ensemblId": ensembl_id})
        liabilities = (data.get("target") or {}).get("safetyLiabilities", [])

        records = [
            {
                "event": l["event"],
                "dosing": l["effects"][0]["dosing"] if l.get("effects") else None,
                "direction": l["effects"][0]["direction"] if l.get("effects") else None,
                "tissue": l["biosamples"][0]["tissueLabel"] if l.get("biosamples") else None,
                "datasource": l["datasource"],
            }
            for l in liabilities
        ]
        return pd.DataFrame(records)

    def get_disease_competitive_landscape(
        self, disease_id: str, size: int = 15
    ) -> pd.DataFrame:
        """
        Top targets associated with a disease, with per-datatype scores.
        Used for the target-first competitive landscape Pareto analysis.
        """
        query = """
        query competitiveLandscape($efoId: String!, $size: Int!) {
          disease(efoId: $efoId) {
            associatedTargets(page: {size: $size, index: 0}) {
              rows {
                target { id approvedSymbol approvedName }
                score
                datatypeScores { id score }
              }
            }
          }
        }
        """
        data = self._query(query, {"efoId": disease_id, "size": size})
        rows_raw = (
            (data.get("disease") or {})
            .get("associatedTargets", {})
            .get("rows", [])
        )

        records = []
        for r in rows_raw:
            t = r["target"]
            rec = {
                "ensembl_id": t["id"],
                "gene_symbol": t["approvedSymbol"],
                "gene_name": t["approvedName"],
                "association_score": r["score"],
            }
            for ds in r.get("datatypeScores", []):
                rec[f"score_{ds['id']}"] = ds["score"]
            records.append(rec)

        return pd.DataFrame(records)

    def get_association_scores(
        self,
        ensembl_id: str,
        disease_ontology_id: Optional[str] = None,
    ) -> dict[str, float]:
        """Return overall and datatype-specific scores for a target, optionally filtered by disease."""
        disease_df = self.get_disease_associations(ensembl_id)
        if disease_df.empty:
            return {"overall": 0.0, "genetics": 0.0, "expression": 0.0}

        if disease_ontology_id:
            row = disease_df[disease_df["disease_id"] == disease_ontology_id]
            if row.empty:
                row = disease_df
        else:
            row = disease_df

        return {
            "overall": float(row["score"].max()),
            "genetics": float(row.get("score_genetic_association", pd.Series([0.0])).max()),
            "expression": float(row.get("score_rna_expression", pd.Series([0.0])).max()),
        }

    def get_novelty_score(self, ensembl_id: str) -> float:
        """Proxy novelty as 1 - (max drug phase / 4). Phase 4 = approved = 0 novelty."""
        clinical_df = self.get_clinical_evidence(ensembl_id)
        if clinical_df.empty:
            return 1.0
        max_phase = clinical_df["phase"].fillna(0).astype(float).max()
        return max(0.0, 1.0 - max_phase / 4.0)

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    def _query(self, query: str, variables: dict) -> dict:
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url,
                    json={"query": query, "variables": variables},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                resp.raise_for_status()
                payload = resp.json()
                if "errors" in payload:
                    raise RuntimeError(f"GraphQL errors: {payload['errors']}")
                return payload.get("data", {})
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}
