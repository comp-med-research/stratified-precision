"""
KG Selector Agent — Claude decides which knowledge sources are relevant
to a specific disease context, beyond the Hetionet baseline.

Different diseases have different evidence ecosystems:
  Alzheimer's  → AlzForum, AGORA (AMP-AD), Alzheimer's disease portal
  Cystic Fibrosis → CFTR2 database, CFTR-specific literature
  Cancer        → COSMIC, cBioPortal, OncoKB
  Rare diseases → OMIM, Orphanet

The agent returns a structured list of additional sources to query, each
with a relevance rationale. The pipeline then fetches what it can and
passes everything to the Bayesian weighter.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional
import anthropic


SYSTEM_PROMPT = """You are a biomedical knowledge engineer specialising in drug target databases.
Your job is to identify the most relevant biological knowledge graphs and databases for a specific
disease context, beyond general-purpose resources.

You return a structured JSON list of additional knowledge sources that would provide signal for
target identification in this specific disease — not generic databases, but ones with strong
disease-specific evidence. For each source, explain what specific features it provides.

Only recommend sources that have open APIs or downloadable data. Be conservative — 2-4 sources max."""

SELECTOR_PROMPT = """Disease context: {disease_name}
Target gene: {gene_symbol}
Patient subgroup characteristics: {subgroup_summary}

Which specialised biological databases or knowledge graphs (beyond Hetionet and OpenTargets)
would provide the most useful additional features for predicting whether this target will succeed
in this disease context?

Return a JSON array where each object has:
  - name: database name
  - rationale: one sentence on why this adds signal for this specific disease
  - feature_type: what kind of features it provides ("pathway", "safety", "genetic", "expression", "clinical")
  - access_url: base URL for the API or data download
  - relevance_score: 0.0–1.0 confidence this is worth querying
"""


@dataclass
class KnowledgeSource:
    name: str
    rationale: str
    feature_type: str   # "pathway" | "safety" | "genetic" | "expression" | "clinical"
    access_url: str
    relevance_score: float


# Hardcoded fallback map — used when the agent can't run (offline / no key)
DISEASE_FALLBACK_SOURCES: dict[str, list[KnowledgeSource]] = {
    "alzheimer": [
        KnowledgeSource(
            name="AGORA",
            rationale="AMP-AD consensus target scores across multi-omic evidence for AD",
            feature_type="genetic",
            access_url="https://agora.adknowledgeportal.org",
            relevance_score=0.95,
        ),
        KnowledgeSource(
            name="AlzForum",
            rationale="Manually curated trial failures and target validations in AD",
            feature_type="clinical",
            access_url="https://www.alzforum.org",
            relevance_score=0.9,
        ),
    ],
    "cystic fibrosis": [
        KnowledgeSource(
            name="CFTR2",
            rationale="Variant-level clinical phenotype data specific to CFTR mutations",
            feature_type="genetic",
            access_url="https://cftr2.org",
            relevance_score=0.95,
        ),
    ],
    "cancer": [
        KnowledgeSource(
            name="OncoKB",
            rationale="Oncogene-specific actionability tiers and clinical evidence",
            feature_type="clinical",
            access_url="https://www.oncokb.org",
            relevance_score=0.9,
        ),
        KnowledgeSource(
            name="cBioPortal",
            rationale="Somatic mutation frequency and co-occurrence patterns across cancer types",
            feature_type="genetic",
            access_url="https://www.cbioportal.org",
            relevance_score=0.85,
        ),
    ],
}


class KGSelectorAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-5"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

    def select_sources(
        self,
        disease_name: str,
        gene_symbol: str,
        subgroup_summary: str = "",
    ) -> list[KnowledgeSource]:
        """
        Return ranked list of additional KG sources for this disease context.
        Falls back to the hardcoded map if the agent can't run.
        """
        if not self.api_key:
            return self._fallback(disease_name)

        try:
            return self._run_agent(disease_name, gene_symbol, subgroup_summary)
        except Exception as e:
            print(f"[KGSelector] Agent failed ({e}), using fallback map.")
            return self._fallback(disease_name)

    def _run_agent(
        self,
        disease_name: str,
        gene_symbol: str,
        subgroup_summary: str,
    ) -> list[KnowledgeSource]:
        client = anthropic.Anthropic(api_key=self.api_key)
        prompt = SELECTOR_PROMPT.format(
            disease_name=disease_name,
            gene_symbol=gene_symbol,
            subgroup_summary=subgroup_summary or "Not characterised yet",
        )

        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = message.content[0].text
        start, end = text.find("["), text.rfind("]") + 1
        if start == -1:
            return self._fallback(disease_name)

        records = json.loads(text[start:end])
        sources = []
        for r in records:
            sources.append(KnowledgeSource(
                name=r.get("name", ""),
                rationale=r.get("rationale", ""),
                feature_type=r.get("feature_type", "unknown"),
                access_url=r.get("access_url", ""),
                relevance_score=float(r.get("relevance_score", 0.5)),
            ))
        return sorted(sources, key=lambda s: s.relevance_score, reverse=True)

    def _fallback(self, disease_name: str) -> list[KnowledgeSource]:
        term = disease_name.lower()
        for key, sources in DISEASE_FALLBACK_SOURCES.items():
            if key in term:
                return sources
        return []
