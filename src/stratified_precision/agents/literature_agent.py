"""
Claude-powered literature agent.

Uses tool use to query Elicit/Amass (or PubMed as fallback) for recent
clinical trial failures associated with a given target, then extracts
structured failure signals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
import anthropic


SYSTEM_PROMPT = """You are an expert clinical pharmacologist specialising in drug target validation.
Your job is to search recent literature and clinical trial databases for evidence of failure
associated with specific drug targets. You return structured, factual summaries — no speculation.
Focus on: Phase 2/3 trial failures, post-market withdrawals, and FDA/EMA safety signals.
Always cite the specific trial or publication."""

EXTRACTION_PROMPT = """Search for recent clinical trial failures or safety signals for the drug target {gene_symbol}.
For each finding, extract:
- trial_id: ClinicalTrials.gov ID or publication DOI
- drug_name: the drug that targeted this gene
- failure_mode: "toxicity" or "efficacy"
- failure_description: one sentence
- year: year of failure
- risk_score: your confidence this represents a genuine failure signal (0.0–1.0)

Return as a JSON array."""


@dataclass
class TrialFailureSignal:
    gene_symbol: str
    trial_id: str
    drug_name: str
    failure_mode: str     # "toxicity" | "efficacy"
    failure_description: str
    year: Optional[int]
    risk_score: float     # 0.0–1.0


class LiteratureAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-5"):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self.model = model

    def scan_target(self, gene_symbol: str) -> list[TrialFailureSignal]:
        """Scan literature for trial failures associated with a single target."""
        return self._run_extraction([gene_symbol])

    def scan_candidates(self, gene_symbols: list[str]) -> list[TrialFailureSignal]:
        """Batch scan for a list of candidate targets."""
        return self._run_extraction(gene_symbols)

    def _run_extraction(self, gene_symbols: list[str]) -> list[TrialFailureSignal]:
        import json
        all_signals: list[TrialFailureSignal] = []

        for symbol in gene_symbols:
            prompt = EXTRACTION_PROMPT.format(gene_symbol=symbol)
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = message.content[0].text
                # Extract JSON array from the response
                start = text.find("[")
                end = text.rfind("]") + 1
                if start == -1 or end == 0:
                    continue

                records = json.loads(text[start:end])
                for r in records:
                    all_signals.append(
                        TrialFailureSignal(
                            gene_symbol=symbol,
                            trial_id=r.get("trial_id", ""),
                            drug_name=r.get("drug_name", ""),
                            failure_mode=r.get("failure_mode", "unknown"),
                            failure_description=r.get("failure_description", ""),
                            year=r.get("year"),
                            risk_score=float(r.get("risk_score", 0.5)),
                        )
                    )
            except Exception as e:
                print(f"[LiteratureAgent] Failed for {symbol}: {e}")

        return all_signals
