"""
Pre-bake demo pipeline results to disk cache before a presentation.

Run once from the project root:
    python scripts/prebake_demos.py

Each result is saved to ~/.stratified_precision_cache/ and will be served
instantly when the same query is submitted via the web UI.

Flags:
  --force      Re-run and overwrite even if already cached.
  --with-lit   Include Claude literature agent (adds ~3 min for patient mode).
               Off by default so baking is fast.
"""

import argparse
import os
import sys
import traceback
sys.path.insert(0, "src")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from stratified_precision.result_store import cache_key, load_result, save_result, list_cached
from stratified_precision.pipeline import run_pipeline

API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ── Demo examples ────────────────────────────────────────────────────────────
# (gene, disease_id, disease_name, display_label)
TARGET_DEMOS = [
    ("PCSK9",  "EFO_0001645",   "familial hypercholesterolaemia", "PCSK9 / Familial hypercholesterolaemia"),
    ("APOE",   "MONDO_0004975", "Alzheimer disease",              "APOE / Alzheimer's disease"),
    ("KCNJ11", "MONDO_0005148", "type 2 diabetes mellitus",       "KCNJ11 / Type 2 diabetes"),
]

COHERENT_DIR = os.path.expanduser("~/Documents/build_small/data/coherent-11-07-2022")
INCLUDE_PATIENT = os.path.isdir(COHERENT_DIR)


def bake_target(gene: str, disease_id: str, disease_name: str, label: str,
                force: bool, with_lit: bool):
    ck = cache_key("target", gene, disease_id)
    if not force and load_result(ck) is not None:
        print(f"  [skip]  {label}  (already cached)")
        return
    print(f"  [run]   {label}  …")
    try:
        from stratified_precision.inputs.target_mode import load_target
        ctx = load_target(gene, disease_id=disease_id, disease_name=disease_name)
        result = run_pipeline(ctx,
                              run_literature_agent=with_lit and bool(API_KEY),
                              use_hetionet=False, anthropic_api_key=API_KEY)
        save_result(ck, result)
        print(f"  [done]  {label}")
    except Exception:
        print(f"  [FAIL]  {label}")
        traceback.print_exc()


def bake_patient(force: bool, with_lit: bool):
    ck = cache_key("patient", "Cohort E - CVD / Metabolic (Coherent EHR)")
    if not force and load_result(ck) is not None:
        print("  [skip]  Coherent EHR cohort  (already cached)")
        return
    lit_note = "with literature agent (~5 min)" if with_lit else "fast — no literature agent (~90 s)"
    print(f"  [run]   Coherent EHR cohort  ({lit_note}) …")
    try:
        from stratified_precision.data.coherent_loader import load_coherent_cohort
        ctx = load_coherent_cohort(COHERENT_DIR, n_endotypes=5)
        result = run_pipeline(ctx,
                              run_literature_agent=with_lit and bool(API_KEY),
                              use_hetionet=False, anthropic_api_key=API_KEY)
        save_result(ck, result)
        print("  [done]  Coherent EHR cohort")
    except Exception:
        print("  [FAIL]  Coherent EHR cohort")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Pre-bake demo pipeline results.")
    parser.add_argument("--force",    action="store_true",
                        help="Re-run and overwrite even if already cached.")
    parser.add_argument("--with-lit", action="store_true",
                        help="Run Claude literature agent (slower but richer failure modes).")
    args = parser.parse_args()

    print("\nStratified Precision — demo pre-baker")
    print("=" * 42)
    if not args.with_lit:
        print("  Tip: pass --with-lit to include failure-mode literature scanning.")

    for gene, did, dname, label in TARGET_DEMOS:
        bake_target(gene, did, dname, label, args.force, args.with_lit)

    if INCLUDE_PATIENT:
        bake_patient(args.force, args.with_lit)
    else:
        print(f"  [skip]  Coherent EHR cohort  (directory not found: {COHERENT_DIR})")

    print("\nCached results:")
    for name in list_cached():
        print(f"  {name}")

    print("\nAll done. Start the app with:  python app.py\n")


if __name__ == "__main__":
    main()
