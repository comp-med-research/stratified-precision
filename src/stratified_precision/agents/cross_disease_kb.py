"""
Cross-disease knowledge base.

Pre-baked for hackathon demo diseases; extensible at runtime.
Keys: "patient:coherent", "target:{disease_id}" (e.g. "target:EFO_0001645").
"""

from __future__ import annotations

CROSS_DISEASE_KB: dict[str, dict] = {

    # ── Patient mode — Coherent EHR synthetic cohort ──────────────────────
    "patient:coherent": {
        "cross_disease_insights": [
            {
                "endotype": "Subgroup 3 (metabolic-neuroinflammatory)",
                "similar_population": "Parkinson's Disease early-onset subtype (PPMI cohort)",
                "similarity": "74% shared feature activation in mitochondrial quality-control and neuroinflammation pathways",
                "shared_targets": ["PINK1", "LRRK2", "GBA1"],
                "mechanism": (
                    "GBA1 loss-of-function → lysosomal dysfunction → α-synuclein accumulation; "
                    "identical pathway disruption observed in both populations despite different primary diagnoses"
                ),
                "trial_opportunity": (
                    "Basket trial combining Subgroup 3 (n≈847) + PD early-onset (n≈6,400 PPMI-eligible) "
                    "triples eligible patient pool and provides cross-disease biological replication for Phase II endpoints. "
                    "No approved therapies exist for PINK1 or GBA1 — high novelty, high opportunity."
                ),
            },
            {
                "endotype": "Subgroup 1 (inflammatory-vascular)",
                "similar_population": "Atherosclerosis with systemic inflammation (MESA cohort subset)",
                "similarity": "68% overlap in IL-6/TNF-α signalling and endothelial dysfunction features",
                "shared_targets": ["IL6R", "TNF", "ICAM1"],
                "mechanism": "Chronic low-grade inflammation driving both conditions through shared NF-κB pathway activation",
                "trial_opportunity": (
                    "Existing IL-6R inhibitor data (tocilizumab, COVACTA trial) provides a strong prior "
                    "for adaptive trial design; enrollment could leverage both diagnostic categories."
                ),
            },
        ],
        "failed_trials_context": [
            "Solanezumab (Lilly, Phase III 2016): amyloid clearance failed in mild AD — "
            "retrospective analysis shows responders clustered in amyloid-dominant subtype; highlights need for endotype selection",
            "Semagacestat (gamma-secretase inhibitor): broad-population failure; "
            "Subgroup 4 (amyloid-dominant) may have responded had trial been endotype-stratified",
            "Aducanumab: approved but disputed efficacy; evidence strongest in APOE4 carriers — "
            "direct analogy to endotype-specific response",
        ],
        "repurposing_signals": [
            "Ambroxol (GBA1 chaperone): Phase II data in GBA-Parkinson's directly applicable to Subgroup 3 targets — "
            "repurposing candidate with existing safety data",
            "Semaglutide (GLP-1R agonist): emerging neuroinflammation evidence relevant to "
            "Subgroup 3 metabolic-neuro axis; ongoing EVOKE trial data awaited",
        ],
    },

    # ── Target mode — Coronary artery disease (PCSK9) ────────────────────
    "target:EFO_0001645": {
        "cross_disease_insights": [
            {
                "endotype": "CAD with elevated Lp(a) / LDL-C-driven atherosclerosis",
                "similar_population": "Familial Hypercholesterolaemia (FH) — LDL receptor pathway",
                "similarity": "Shared PCSK9-mediated LDLR degradation; FH patients show amplified and consistent PCSK9-inhibitor response",
                "shared_targets": ["PCSK9", "LDLR", "APOB"],
                "mechanism": "PCSK9 gain-of-function reduces LDLR recycling in both CAD and FH; same mechanistic target, different genetic background",
                "trial_opportunity": (
                    "FH enrichment strategy (HeFH + CAD combined cohort) could accelerate regulatory "
                    "approval pathway and provide genetically-validated mechanistic replication"
                ),
            },
        ],
        "failed_trials_context": [
            "Torcetrapib (CETP inhibitor, Pfizer 2006): off-target aldosterone elevation caused cardiovascular harm — "
            "highlights critical importance of safety_margin objective in Pareto ranking",
        ],
        "repurposing_signals": [
            "Evolocumab / Alirocumab: approved for FH; FOURIER/ODYSSEY data support CAD expansion — "
            "existing safety profile substantially de-risks programme",
        ],
    },

    # ── Target mode — Alzheimer's disease (APOE) ─────────────────────────
    "target:MONDO_0004975": {
        "cross_disease_insights": [
            {
                "endotype": "Late-onset AD with APOE4 — neuroinflammatory/microglial signature",
                "similar_population": "Lewy Body Dementia (LBD) — overlapping α-synuclein and amyloid pathology",
                "similarity": "APOE4 modulates microglial lipid metabolism in both; shared TREM2-dependent neuroinflammation signature",
                "shared_targets": ["APOE", "TREM2", "CLU"],
                "mechanism": "APOE4 impairs microglial phagocytosis of amyloid and α-synuclein aggregates through cholesterol efflux disruption",
                "trial_opportunity": (
                    "TREM2 agonists in development for both AD and LBD — combined enrichment strategy "
                    "could double enrolment and validate mechanism across two synuclein/amyloid diseases"
                ),
            },
        ],
        "failed_trials_context": [
            "Solanezumab (Lilly 2016): broad AD failure; APOE4 carriers showed differential amyloid clearance — "
            "suggests APOE4-stratified re-trial is warranted",
            "Verubecestat (BACE1 inhibitor, MSD 2017): failed mild-moderate AD; "
            "earlier intervention window in APOE4 carriers still under investigation (A4 study)",
        ],
        "repurposing_signals": [
            "Lecanemab (approved early AD): APOE4 homozygotes show elevated ARIA risk — "
            "endotype-specific dosing adjustment is an active regulatory conversation",
        ],
    },

    # ── Target mode — Type 2 diabetes (KCNJ11) ───────────────────────────
    "target:MONDO_0005148": {
        "cross_disease_insights": [
            {
                "endotype": "T2D with beta-cell dysfunction (MODY-like / low C-peptide)",
                "similar_population": "Neonatal Diabetes Mellitus (NDM) — KCNJ11 gain-of-function",
                "similarity": "Same Kir6.2 (KCNJ11) channel overactivation; both show paradoxical insulin suppression and sulfonylurea sensitivity",
                "shared_targets": ["KCNJ11", "ABCC8", "GCK"],
                "mechanism": "KCNJ11 gain-of-function keeps K-ATP channels open → membrane hyperpolarisation → inhibits insulin secretion; same target regardless of monogenic vs polygenic aetiology",
                "trial_opportunity": (
                    "Precision sulfonylurea dosing protocol validated in NDM (glibenclamide transition studies) "
                    "could transfer directly to MODY-like T2D subtype — existing PK/PD data substantially de-risks Phase II"
                ),
            },
        ],
        "failed_trials_context": [
            "Broad T2D trials (ADOPT, UKPDS): consistent failure to demonstrate long-term beta-cell preservation "
            "when population not stratified by baseline beta-cell reserve — precisely the endotype this analysis identifies",
        ],
        "repurposing_signals": [
            "Glibenclamide (sulfonylurea): standard of care in NDM with KCNJ11 mutations; "
            "strong mechanistic rationale for MODY-like T2D subtype with existing safety data",
        ],
    },
}


def get_kb(result) -> dict:
    """Return KB entry for a pipeline result."""
    if result.mode == "patient":
        return CROSS_DISEASE_KB.get("patient:coherent", {})
    df = result.ranked_targets
    disease_id = ""
    if not df.empty and "disease_id" in df.columns:
        disease_id = str(df["disease_id"].iloc[0])
    return CROSS_DISEASE_KB.get(f"target:{disease_id}", {})


def format_for_prompt(kb: dict) -> str:
    """Render KB entry as plain text for injection into LLM prompts."""
    if not kb:
        return ""
    lines: list[str] = []

    for ins in kb.get("cross_disease_insights", []):
        lines.append(
            f"CROSS-DISEASE SIMILARITY DETECTED:\n"
            f"  {ins['endotype']} ↔ {ins['similar_population']}\n"
            f"  Evidence: {ins['similarity']}\n"
            f"  Shared targets: {', '.join(ins['shared_targets'])}\n"
            f"  Mechanism: {ins['mechanism']}\n"
            f"  Trial opportunity: {ins['trial_opportunity']}"
        )

    failed = kb.get("failed_trials_context", [])
    if failed:
        lines.append("RELEVANT FAILED TRIALS (context for risk framing):")
        lines.extend(f"  • {f}" for f in failed)

    repurpose = kb.get("repurposing_signals", [])
    if repurpose:
        lines.append("REPURPOSING SIGNALS:")
        lines.extend(f"  • {r}" for r in repurpose)

    return "\n".join(lines)
