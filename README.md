# Stratified Precision

AI co-scientist for target identification — built for the TernaryTx / future.bio / Pluto House / Anthropic Drug Discovery Hackathon.

## What it does

Predicts clinical trial attrition risk for drug targets and classifies failure mode (toxicity vs. efficacy), using disease endotyping to ground molecular analysis in the patient populations that actually matter.

## Two entry points

### Mode 1 — Target-first (drug developer perspective)
Start with a gene/protein you're already interested in.

```
BACE1  →  Who are the patients?  →  Which subgroup does this actually help?
          What drives failure?       Toxicity or lack of efficacy?
          Rank vs. alternative targets on Pareto front
```

### Mode 2 — Patient-first (clinician perspective)
Start with a clinical dataset or patient cohort.

```
Patient cohort  →  Disease endotyping  →  Which subgroups exist?
                   GWAS enrichment        What targets are relevant per subgroup?
                   OpenTargets mapping    Rank novel targets by predicted success
```

Both modes converge into the same shared analysis pipeline.

## Pipeline

```
[Mode 1] target_mode.py          [Mode 2] patient_mode.py
         ↓ TargetContext                   ↓ List[CandidateTarget]
              └──────────┬────────────────┘
                         ↓
               pipeline.py  (shared core)
                    │
                    ├─ endotyping/   UMAP + HDBSCAN — patient subgroup discovery
                    ├─ causal/       DoWhy DAG — why does this target fail?
                    ├─ agents/       Claude + literature scan (Elicit/Amass)
                    └─ optimization/ pymoo Pareto front (safety × efficacy × specificity)
                         ↓
               viz/dashboard.py     Plotly Dash — interactive results
```

## Datasets

- **OpenTargets** — biological network, tissue expression, genetic association, clinical evidence
- **GWAS Catalog** — genetic associations per disease/phenotype
- **GTEx** — tissue expression profiles
- **Elicit / Amass** — literature for recent trial failure signals (via agent)

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
python app.py
```

Navigate to `http://localhost:8050`.

## Structure

```
src/stratified_precision/
├── inputs/
│   ├── target_mode.py      # Mode 1: gene/protein → TargetContext
│   └── patient_mode.py     # Mode 2: clinical data → CandidateTarget list
├── data/
│   └── opentargets.py      # OpenTargets GraphQL client
├── endotyping/
│   └── clustering.py       # UMAP + HDBSCAN disease endotyping
├── causal/
│   └── failure_classifier.py  # DoWhy causal DAG, toxicity vs. efficacy
├── agents/
│   └── literature_agent.py    # Claude tool-use agent for trial failure literature
├── optimization/
│   └── pareto.py           # pymoo NSGA-II multi-objective ranking
└── viz/
    └── dashboard.py        # Plotly Dash app
```
