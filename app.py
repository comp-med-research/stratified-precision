"""
Entry point — run the Stratified Precision dashboard.

Usage:
    # Mode 1: target-first
    python app.py --target BACE1

    # Mode 2: patient-first
    python app.py --patients data/raw/cohort.csv --disease EFO_0000270
"""

import argparse
import os
import sys

sys.path.insert(0, "src")

from stratified_precision.inputs.target_mode import load_target
from stratified_precision.inputs.patient_mode import load_patient_cohort
from stratified_precision.pipeline import run_pipeline
from stratified_precision.viz.dashboard import run_dashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified Precision — Drug Target Analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", metavar="GENE", help="Gene symbol, e.g. BACE1")
    group.add_argument("--patients", metavar="PATH", help="Path to patient CSV")
    parser.add_argument("--disease", metavar="EFO_ID", help="Disease ontology ID (patient mode)")
    parser.add_argument("--no-agent", action="store_true", help="Skip literature agent (faster, offline)")
    parser.add_argument("--no-hetionet", action="store_true", help="Skip Hetionet KG enrichment (faster, offline)")
    parser.add_argument("--kg-hops", type=int, default=2, help="Hetionet subgraph hop depth (default: 2)")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.target:
        print(f"[Mode 1] Loading target: {args.target}")
        context = load_target(args.target)
    else:
        print(f"[Mode 2] Loading patient cohort: {args.patients}")
        context = load_patient_cohort(
            args.patients,
            disease_ontology_id=args.disease,
        )

    print("Running pipeline...")
    result = run_pipeline(
        context,
        run_literature_agent=not args.no_agent,
        use_hetionet=not args.no_hetionet,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        kg_hops=args.kg_hops,
    )

    print(f"Done. {len(result.ranked_targets)} targets ranked. Starting dashboard on port {args.port}...")
    run_dashboard(result, port=args.port)


if __name__ == "__main__":
    main()
