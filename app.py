"""
Entry point — Flask serves the landing page, Dash serves /dashboard/.
Run: python app.py
"""

import os
import sys
sys.path.insert(0, "src")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify

# ── Shared result cache (Flask route writes, Dash dashboard reads) ────
from stratified_precision.cache import result_cache
from stratified_precision.result_store import cache_key, load_result, save_result

# ── Flask server ──────────────────────────────────────────────────────
server = Flask(__name__, template_folder="templates", static_folder="static")
server.secret_key = os.urandom(24)

# ── Mount Dash results dashboard at /dashboard/ ───────────────────────
from stratified_precision.viz.dashboard import create_dash_app
create_dash_app(server)

# ── Flask routes ──────────────────────────────────────────────────────

@server.route("/")
def index():
    return render_template("index.html")


@server.route("/get-diseases")
def get_diseases():
    """Lightweight endpoint: resolve gene → return top disease associations."""
    gene = request.args.get("gene", "").strip()
    if not gene:
        return jsonify(diseases=[], error="No gene provided.")
    try:
        from stratified_precision.data.opentargets import OpenTargetsClient
        ot = OpenTargetsClient()
        ensembl_id = ot.resolve_gene_symbol(gene)
        if not ensembl_id:
            return jsonify(diseases=[], error=f"Gene '{gene}' not found in OpenTargets.")
        df = ot.get_disease_associations(ensembl_id)
        if df.empty:
            return jsonify(diseases=[], ensembl_id=ensembl_id,
                           error=f"No disease associations found for {gene}.")
        top = df.sort_values("score", ascending=False).head(6)
        diseases = top[["disease_id", "disease_name", "score"]].to_dict("records")
        return jsonify(diseases=diseases, ensembl_id=ensembl_id)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(diseases=[], error=str(e))


@server.route("/analyse", methods=["POST"])
def analyse():
    query        = request.form.get("query", "").strip()
    mode         = request.form.get("mode", "target")
    disease_id   = request.form.get("disease_id", "").strip() or None
    disease_name = request.form.get("disease_name", "").strip() or None

    if not query:
        return jsonify(ok=False, error="Please enter a gene or disease name.")

    force = request.form.get("force", "0") == "1"

    try:
        from stratified_precision.pipeline import run_pipeline
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if mode == "patient":
            ck = cache_key("patient", query)
            if not force:
                cached = load_result(ck)
                if cached is not None:
                    cached._cache_key = ck
                    result_cache["latest"] = cached
                    return jsonify(ok=True, from_cache=True)

            from stratified_precision.data.coherent_loader import load_coherent_cohort

            COHERENT_DIR = os.path.expanduser(
                "~/Documents/build_small/data/coherent-11-07-2022"
            )
            if not os.path.isdir(COHERENT_DIR):
                return jsonify(ok=False, error="Coherent dataset not found on this machine.")

            context = load_coherent_cohort(COHERENT_DIR, n_endotypes=5)
            # Skip literature agent in the live web path — too slow for 100+ candidates.
            # The AI explanation panel on the dashboard handles on-demand explanations.
            result = run_pipeline(
                context,
                run_literature_agent=False,
                use_hetionet=False,
                anthropic_api_key=api_key,
            )
            result._cache_key = ck
            save_result(ck, result)
            result_cache["latest"] = result
            return jsonify(ok=True, from_cache=False)

        ck = cache_key("target", query, disease_id or "")
        if not force:
            cached = load_result(ck)
            if cached is not None:
                cached._cache_key = ck
                result_cache["latest"] = cached
                return jsonify(ok=True, from_cache=True)

        from stratified_precision.inputs.target_mode import load_target
        context = load_target(query, disease_id=disease_id, disease_name=disease_name)
        result = run_pipeline(
            context,
            run_literature_agent=bool(api_key),
            use_hetionet=True,   # downloads once to ~/.cache/stratified_precision/
            anthropic_api_key=api_key,
        )
        result._cache_key = ck
        save_result(ck, result)
        result_cache["latest"] = result
        return jsonify(ok=True, from_cache=False)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(ok=False, error=str(e))


if __name__ == "__main__":
    server.run(port=8050, debug=False)
