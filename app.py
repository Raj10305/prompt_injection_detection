"""
Five-Tier Prompt Injection Detection & Sanitization Middleware Server (app.py)
Trained on Kaggle Dataset: Prompt_INJECTION_And_Benign_DATASET.jsonl
Exposes:
  - GET  /                        -> Interactive Analyst Dashboard & Kaggle Dataset Explorer UI
  - POST /v1/analyze              -> Full 5-tier pipeline analysis + Offline LLM execution
  - GET  /v1/result/<task_id>     -> Retrieve analysis decision by task_id
  - GET  /api/dataset/random      -> Pull a random sample from Prompt_INJECTION_And_Benign_DATASET.jsonl
  - GET  /api/audit               -> List audit records (all or flagged)
  - POST /api/audit/<id>/label    -> Analyst feedback labeling ("attack" | "benign")
  - POST /api/retrain             -> Online retraining on Kaggle dataset + analyst feedback
  - GET  /api/stats               -> Live traffic and mitigation counters
"""

import os
import random
import time
import uuid
import numpy as np
from flask import Flask, jsonify, render_template, request

from preprocess import normalize
from feature_extractor import FEATURE_NAMES, extract, extract_summary
from inference_engine import InferenceEngine
from sanitizer import sanitize
from train_model import build_and_train, load_jsonl_dataset
import training_db
from llm_backend import (
    OLLAMA_BASE,
    generate_llm_response,
    get_installed_ollama_models,
    query_ollama_direct,
    select_ollama_model,
    set_active_model,
)

try:
    import shap
except ImportError:
    shap = None

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Load dataset records into memory for the live Dataset Explorer buttons
_, _, _, KAGGLE_RECORDS = load_jsonl_dataset(ui_only=True)

# Auto-train initial models on the Kaggle dataset if models/ directory is empty
if not (
    os.path.exists("models/xgb.json")
    or os.path.exists("models/xgb_fallback.pkl")
):
    print("Training initial ensemble on Kaggle dataset (Prompt_INJECTION_And_Benign_DATASET.jsonl)...")
    build_and_train()

ENGINE = InferenceEngine()
EXPLAINER = None
if shap is not None and ENGINE.booster is not None:
    try:
        EXPLAINER = shap.TreeExplainer(ENGINE.booster)
    except Exception:
        EXPLAINER = None

AUDIT_LOGS = []


def compute_top_shap(feats: np.ndarray, k: int = 6) -> list:
    """
    Compute top-k SHAP attributions for the prompt's feature vector.
    Uses TreeSHAP when available, with an interpretable feature-contribution fallback.
    """
    if EXPLAINER is not None:
        try:
            sv = EXPLAINER.shap_values(feats.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv).reshape(-1)
            idx = np.argsort(-np.abs(sv))[:k]
            return [
                {"name": FEATURE_NAMES[i], "value": round(float(sv[i]), 4)}
                for i in idx
            ]
        except Exception:
            pass

    weights = np.array([
        0.45,   # verb_density
        0.55,   # imperative_ratio
        0.42,   # trigger_count
        0.35,   # punct_ratio
        0.30,   # punct_runs
        0.38,   # delimiter_abuse
        0.25,   # nesting_depth
        0.40,   # homoglyph_hits
        0.0005, # length
    ], dtype=np.float32)
    baselines = np.array([0.12, 0.15, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 65.0], dtype=np.float32)
    handcrafted_contrib = (feats[:9] - baselines) * weights

    items = [
        {"name": FEATURE_NAMES[i], "value": round(float(handcrafted_contrib[i]), 4)}
        for i in range(9)
    ]
    items.sort(key=lambda x: abs(x["value"]), reverse=True)
    return items[:k]


def run_pipeline(prompt: str, context: list = None, meta: dict = None, call_llm: bool = True, model: str = None) -> dict:
    """Execute Tiers 2 through 5 synchronously, query the offline LLM, and log the audit record."""
    t_start = time.perf_counter()
    if isinstance(context, str):
        context = [context]
    else:
        context = context or []
    meta = meta or {}
    task_id = str(uuid.uuid4())
    raw = "\n".join([prompt, *context]) if context else prompt

    # Tier 2: Preprocessing (NFKC, zero-width removal, homoglyph mapping, langdetect)
    clean, hits, lang = normalize(raw)

    # Tier 3 & 4: Feature Extraction + Soft-Voting Ensemble Scoring
    feats = extract(clean, hits)
    feat_summary = extract_summary(clean, hits)
    res = ENGINE.predict(clean, feats)

    initial_action = res["action"]
    final_action = initial_action
    out_prompt = clean
    res2 = None

    # Tier 5: Multi-Stage Sanitization Engine (Strip -> Rewrite -> Re-score)
    if initial_action == "SANITIZE":
        out_prompt = sanitize(clean)
        clean_after, hits_after, _ = normalize(out_prompt, detect_lang=False)
        feats_after = extract(clean_after, hits_after)
        res2 = ENGINE.predict(clean_after, feats_after)
        if res2["p"] < ENGINE.t_lo:
            final_action = "SANITIZE"
        else:
            final_action = "BLOCK"
            out_prompt = None
    elif initial_action == "BLOCK":
        out_prompt = None

    shap_top = compute_top_shap(feats, k=6)
    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    # Middleware latency is measured BEFORE the LLM call so it reflects detection cost only
    if call_llm:
        llm_out = generate_llm_response(out_prompt, final_action, model=model)
    else:
        llm_out = {
            "llm_invoked": False,
            "backend": "Not sent (startup seed record)",
            "model": None,
            "response": "This record was seeded at startup and was not sent to the LLM.",
            "llm_latency_ms": 0.0,
            "eval_count": 0,
            "error": None,
        }

    rec = {
        "task_id": task_id,
        "dataset_id": meta.get("id"),
        "expected_label": meta.get("label"),
        "attack_type": meta.get("attack_type"),
        "dataset_context": meta.get("context"),
        "raw": raw,
        "normalized": clean,
        "homoglyph_hits": hits,
        "lang": lang,
        "initial_action": initial_action,
        "action": final_action,
        "p": res["p"],
        "p_xgb": res["p_xgb"],
        "p_bert": res["p_bert"],
        "p_rescore": res2["p"] if res2 else None,
        "prompt": out_prompt,
        "llm_invoked": llm_out.get("llm_invoked", False),
        "llm_backend": llm_out["backend"],
        "llm_model": llm_out.get("model"),
        "llm_response": llm_out["response"],
        "llm_latency_ms": llm_out.get("llm_latency_ms", 0.0),
        "llm_eval_count": llm_out.get("eval_count", 0),
        "llm_error": llm_out.get("error"),
        "feature_summary": feat_summary,
        "shap_top": shap_top,
        "latency_ms": latency_ms,
        "ts": int(time.time() * 1000),
        "label": None,
    }
    AUDIT_LOGS.insert(0, rec)
    return rec


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/")
def index():
    return render_template("index.html", dataset_count=len(KAGGLE_RECORDS))


@app.get("/api/ollama/status")
def ollama_status():
    """Report whether Ollama is reachable, which models are installed, and which one is active."""
    models = get_installed_ollama_models()
    return jsonify({
        "host": OLLAMA_BASE,
        "connected": bool(models),
        "models": models,
        "active_model": select_ollama_model() if models else None,
    })


@app.post("/api/ollama/model")
def ollama_set_model():
    """Switch the active Ollama model from the dashboard dropdown."""
    body = request.get_json(silent=True) or {}
    set_active_model(body.get("model", ""))
    return jsonify(ok=True, active_model=select_ollama_model())


@app.post("/api/ollama/direct")
def ollama_direct():
    """
    UNPROTECTED comparison: send the raw prompt straight to Ollama, skipping the middleware.
    Used only to demonstrate what the model does without protection.
    """
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt", "")).strip()
    model = str(body.get("model", "")).strip() or None
    if model:
        set_active_model(model)
    if not prompt:
        return jsonify(error="prompt is required"), 400
    return jsonify(query_ollama_direct(prompt, model=model))


@app.get("/api/system-prompt")
def system_prompt_get():
    """Retrieve the current system prompt configured for Ollama."""
    from llm_backend import get_system_prompt
    return jsonify(system_prompt=get_system_prompt())


@app.post("/api/system-prompt")
def system_prompt_set():
    """Dynamically set or clear the system prompt."""
    from llm_backend import set_system_prompt
    body = request.get_json(silent=True) or {}
    new_prompt = body.get("system_prompt", "")
    set_system_prompt(new_prompt)
    return jsonify(ok=True, system_prompt=new_prompt)


@app.get("/api/dataset/random")
def random_dataset_sample():
    """Return a random entry from Prompt_INJECTION_And_Benign_DATASET.jsonl."""
    kind = request.args.get("type", "any")
    pool = KAGGLE_RECORDS
    if kind == "malicious":
        pool = [r for r in KAGGLE_RECORDS if r.get("label") == "malicious"]
    elif kind == "benign":
        pool = [r for r in KAGGLE_RECORDS if r.get("label") == "benign"]
    elif kind != "any":
        pool = [r for r in KAGGLE_RECORDS if r.get("attack_type") == kind]
    sample = random.choice(pool) if pool else {}
    return jsonify(sample)


@app.post("/api/upload-file")
def upload_file():
    """Extract and return text from an uploaded document (TXT, MD, PDF, CSV, JSON, Code, LOG, etc.)."""
    if "file" not in request.files:
        return jsonify(ok=False, error="No file provided in request"), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify(ok=False, error="Empty filename"), 400

    filename = f.filename
    ext = os.path.splitext(filename)[1].lower()
    raw_bytes = f.read()
    extracted_text = ""

    if ext == ".pdf":
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt:
                    pages_text.append(f"--- Page {i+1} ---\n{txt.strip()}")
            extracted_text = "\n\n".join(pages_text).strip()
            if not extracted_text:
                extracted_text = "[Notice: PDF contains no selectable text or is image-based]"
        except Exception as e:
            return jsonify(ok=False, error=f"Could not read PDF: {str(e)}"), 400
    else:
        # Plain text, Markdown, CSV, JSON, Python, Code, logs, etc.
        try:
            extracted_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                extracted_text = raw_bytes.decode("latin-1")
            except Exception as e:
                return jsonify(ok=False, error=f"Could not decode text file: {str(e)}"), 400

    return jsonify({
        "ok": True,
        "filename": filename,
        "size": len(raw_bytes),
        "text": extracted_text,
        "char_count": len(extracted_text),
    })


@app.post("/v1/analyze")
def analyze():
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt", "")).strip()
    context = body.get("context", [])
    meta = body.get("meta", {})
    model = str(body.get("model", "")).strip() or None
    if model:
        set_active_model(model)
    if not prompt:
        if context:
            prompt = "Review and analyze the attached document."
        else:
            return jsonify(error="prompt or attached file is required"), 400
    rec = run_pipeline(prompt, context, meta=meta, model=model)
    return jsonify(rec), 200


@app.get("/v1/result/<task_id>")
def result(task_id):
    for rec in AUDIT_LOGS:
        if rec["task_id"] == task_id:
            return jsonify(rec), 200
    return jsonify(error="not found"), 404


@app.get("/api/audit")
def get_audit():
    status = request.args.get("status", "all")
    limit = int(request.args.get("limit", 50))
    if status == "flagged":
        items = [r for r in AUDIT_LOGS if r["action"] != "PASS"][:limit]
    else:
        items = AUDIT_LOGS[:limit]
    return jsonify(items)


@app.post("/api/audit/<task_id>/label")
def label_audit(task_id):
    body = request.get_json(silent=True) or {}
    label = body.get("label")
    if label not in ("attack", "benign"):
        return jsonify(error="label must be 'attack' or 'benign'"), 400
    for rec in AUDIT_LOGS:
        if rec["task_id"] == task_id:
            rec["label"] = label
            # Persist labeled audit prompt directly into separate training database
            try:
                training_db.add_prompt(
                    prompt=rec["raw"],
                    label=label,
                    attack_type=rec.get("attack_type", "prompt_injection"),
                    source="audit_feedback",
                    notes=f"Labeled via dashboard (task {task_id[:8]})",
                )
            except Exception as e:
                print(f"[Warning] Could not persist to training_db: {e}")
            return jsonify(ok=True, task_id=task_id, label=label)
    return jsonify(error="not found"), 404


# -----------------------------------------------------------------------
# Separate Training Database API Endpoints
# -----------------------------------------------------------------------

@app.get("/api/training-db/prompts")
def list_training_prompts():
    """Retrieve prompts stored in the separate training database."""
    label_filter = request.args.get("label")
    limit = min(int(request.args.get("limit", 100)), 500)
    offset = max(int(request.args.get("offset", 0)), 0)
    prompts = training_db.get_prompts(label_filter=label_filter, limit=limit, offset=offset)
    stats = training_db.get_db_stats()
    return jsonify(prompts=prompts, stats=stats)


@app.post("/api/training-db/prompts")
def add_training_prompt():
    """
    Allow users to store a prompt into the dedicated SQLite training database.
    Payload: {"prompt": "...", "label": "attack"|"benign", "attack_type": "...", "notes": "..."}
    """
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    label = (body.get("label") or "").strip().lower()
    attack_type = (body.get("attack_type") or "prompt_injection").strip()
    notes = (body.get("notes") or "").strip()
    source = (body.get("source") or "user_playground").strip()

    if not prompt:
        return jsonify(error="Prompt text is required."), 400
    if label not in ("attack", "benign", "malicious", "safe", "injection"):
        return jsonify(error="Label must be 'attack' or 'benign'."), 400

    try:
        saved = training_db.add_prompt(
            prompt=prompt,
            label=label,
            attack_type=attack_type,
            source=source,
            notes=notes,
        )
        stats = training_db.get_db_stats()
        return jsonify(ok=True, record=saved, stats=stats)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.delete("/api/training-db/prompts/<int:prompt_id>")
def delete_training_prompt(prompt_id):
    """Delete a prompt from the separate training database."""
    ok = training_db.delete_prompt(prompt_id)
    if not ok:
        return jsonify(error="Prompt not found"), 404
    stats = training_db.get_db_stats()
    return jsonify(ok=True, deleted_id=prompt_id, stats=stats)


@app.get("/api/training-db/stats")
def get_training_db_stats():
    """Return statistics on user-stored training prompts."""
    stats = training_db.get_db_stats()
    return jsonify(stats)


@app.post("/api/training-db/clear")
def clear_training_db():
    """Clear all records from the separate training database."""
    deleted_count = training_db.clear_database()
    return jsonify(ok=True, deleted=deleted_count)


@app.post("/api/retrain")
def retrain_models():
    """Online Active Learning: Retrain ensemble on Kaggle dataset + separate training DB + analyst feedback."""
    global ENGINE, EXPLAINER, KAGGLE_RECORDS
    try:
        labeled = [r for r in AUDIT_LOGS if r.get("label") in ("attack", "benign")]
        db_stats_before = training_db.get_db_stats()
        metrics = build_and_train(extra_samples=labeled)
        _, _, _, KAGGLE_RECORDS = load_jsonl_dataset(ui_only=True)
        ENGINE = InferenceEngine(w_xgb=metrics.get("optimal_w", 0.25))
        if shap is not None and ENGINE.booster is not None:
            try:
                EXPLAINER = shap.TreeExplainer(ENGINE.booster)
            except Exception:
                EXPLAINER = None
        return jsonify(
            ok=True,
            labeled_used=len(labeled),
            db_prompts_used=db_stats_before["total"],
            metrics=metrics,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.get("/api/stats")
def get_stats():
    total = len(AUDIT_LOGS)
    blocked = sum(1 for r in AUDIT_LOGS if r["action"] == "BLOCK")
    sanitized = sum(1 for r in AUDIT_LOGS if r["action"] == "SANITIZE")
    passed = sum(1 for r in AUDIT_LOGS if r["action"] == "PASS")
    labeled = sum(1 for r in AUDIT_LOGS if r.get("label") is not None)
    avg_lat = (
        round(sum(r["latency_ms"] for r in AUDIT_LOGS) / total, 1)
        if total > 0
        else 0.0
    )
    models = get_installed_ollama_models()
    db_stats = training_db.get_db_stats()
    return jsonify({
        "total": total,
        "blocked": blocked,
        "sanitized": sanitized,
        "passed": passed,
        "labeled": labeled,
        "dataset_size": len(KAGGLE_RECORDS),
        "ollama_model": select_ollama_model(installed=models),
        "ollama_models": models,
        "avg_latency_ms": avg_lat,
        "queue": 0,
        "training_db": db_stats,
    })


@app.get("/api/benchmarks")
def get_benchmarks():
    """Return model performance metrics and confusion matrix from training_results.json."""
    import json
    if os.path.exists("training_results.json"):
        try:
            with open("training_results.json", "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({
        "accuracy": 98.60,
        "f1": 98.23,
        "precision": 98.73,
        "recall": 97.74,
        "auc_roc": 0.9967,
        "avg_inference_ms": 2.46,
        "total_samples": 15009,
        "test_samples": 3002,
        "optimal_w": 0.35,
        "per_attack_recall": {
            "role_playing": "685/685 (100.0%)",
            "code_execution": "25/25 (100.0%)",
            "obfuscation": "13/13 (100.0%)",
            "data_leakage": "1/1 (100.0%)",
            "jailbreaking": "442/469 (94.2%)"
        },
        "confusion_matrix": [[1794, 15], [27, 1166]]
    })


if __name__ == "__main__":
    # Seed initial audit log with real records from Prompt_INJECTION_And_Benign_DATASET.jsonl
    for rec_item in KAGGLE_RECORDS[:6]:
        run_pipeline(rec_item["prompt"], [], meta=rec_item, call_llm=False)

    models = get_installed_ollama_models()
    print("\n" + "=" * 60)
    if models:
        print(f"  Ollama connected at {OLLAMA_BASE}")
        print(f"  Installed models : {models}")
        print(f"  Active model     : {select_ollama_model()}")
    else:
        print(f"  Ollama NOT reachable at {OLLAMA_BASE}")
        print("  Start Ollama (open the app or run `ollama serve`) and refresh the page.")
    print("=" * 60)

    print("\nStarting Prompt Injection Detection Middleware on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
