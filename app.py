"""
Entry point — Flask serves the landing page, Dash serves /dashboard/.
Run: python app.py
"""

import os
import sys
sys.path.insert(0, "src")

from flask import Flask, render_template, request, jsonify

# ── Shared result cache (Flask route writes, Dash dashboard reads) ────
from stratified_precision.cache import result_cache

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


@server.route("/analyse", methods=["POST"])
def analyse():
    query = request.form.get("query", "").strip()
    mode  = request.form.get("mode", "target")

    if not query:
        return jsonify(ok=False, error="Please enter a gene or disease name.")

    try:
        from stratified_precision.pipeline import run_pipeline

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if mode == "patient":
            # Patient CSV upload handled separately — fall back to target mode
            return jsonify(ok=False, error="Upload a CSV via patient mode on the dashboard.")

        from stratified_precision.inputs.target_mode import load_target
        context = load_target(query)

        result = run_pipeline(
            context,
            run_literature_agent=bool(api_key),
            use_hetionet=False,
            anthropic_api_key=api_key,
        )

        result_cache["latest"] = result
        return jsonify(ok=True)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(ok=False, error=str(e))


if __name__ == "__main__":
    server.run(port=8050, debug=False)
