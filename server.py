"""
Dashboard server for Naukri Auto-Apply Bot.
Provides:
  - REST API for the bot to log jobs and manage Q&A
  - Profile + resume (S3) management
  - Web dashboard at http://localhost:5000
"""

import os
import uuid
import difflib
import logging
import boto3
from datetime import datetime
from flask import Flask, render_template, jsonify, request, g
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("naukri.server")
# Quieten noisy libraries
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB resume limit

# Allow cross-origin requests (needed when frontend is on a different domain/port)
CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*")}})

MONGODB_URI  = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
AWS_BUCKET   = os.getenv("AWS_S3_BUCKET", "")
AWS_REGION   = os.getenv("AWS_REGION", "ap-south-1")

try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    log.info("MongoDB connected: %s", MONGODB_URI.split("@")[-1])  # hide creds if in URI
except Exception as exc:
    log.warning("MongoDB connection warning: %s — will retry on first request", exc)
    client = MongoClient(MONGODB_URI)

db = client.naukri_bot

# Collections
jobs_col      = db.jobs
qa_col        = db.qa_pairs
questions_col = db.unknown_questions
profile_col   = db.profile
context_col   = db.context


# ── Request / response logging ────────────────────────────────────────────────

@app.before_request
def _log_request():
    g.req_id = str(uuid.uuid4())[:8]
    if not request.path.startswith("/api"):
        return
    log.info("[%s] → %s %s", g.req_id, request.method, request.path)

@app.after_request
def _log_response(response):
    if not request.path.startswith("/api"):
        return response
    log.info("[%s] ← %s %s %s", g.req_id, request.method, request.path, response.status_code)
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serial(doc):
    if doc and "_id" in doc:
        doc.pop("_id")
    return doc

def _now():
    return datetime.utcnow().isoformat()

def fuzzy_find(question_text, threshold=0.72):
    all_qa = list(qa_col.find({}, {"_id": 0}))
    best, best_ratio = None, 0.0
    for qa in all_qa:
        ratio = difflib.SequenceMatcher(
            None, question_text.lower(), qa["question"].lower()
        ).ratio()
        if ratio > best_ratio:
            best_ratio, best = ratio, qa
    if best and best_ratio >= threshold:
        best["similarity"] = round(best_ratio, 3)
        return best
    return None

def _s3():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


# ── Dashboard page ────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("index.html")


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    applied = jobs_col.count_documents({"status": "applied"})
    skipped = jobs_col.count_documents({"status": "skipped"})
    failed  = jobs_col.count_documents({"status": "failed"})
    recent  = [_serial(j) for j in
               jobs_col.find({}, {"_id": 0}).sort("timestamp", DESCENDING).limit(30)]
    pending = [_serial(q) for q in
               questions_col.find({"answered": False}, {"_id": 0})
               .sort("created_at", DESCENDING)]
    return jsonify({
        "applied": applied, "skipped": skipped, "failed": failed,
        "recent_jobs": recent,
        "pending_questions": pending,
        "qa_count": qa_col.count_documents({}),
    })


# ── Job logging ───────────────────────────────────────────────────────────────

@app.route("/api/job", methods=["POST"])
def log_job():
    data   = request.json or {}
    status = data.get("status", "unknown")
    title  = data.get("title", "")
    jobs_col.insert_one({
        "url":     data.get("url", ""),
        "title":   title,
        "company": data.get("company", ""),
        "status":  status,
        "timestamp": _now(),
    })
    log.info("Job logged  [%s] %s", status.upper(), title or data.get("url", "?")[:80])
    return jsonify({"ok": True})


# ── Q&A lookup (used by bot) ──────────────────────────────────────────────────

@app.route("/api/answer-lookup", methods=["POST"])
def answer_lookup():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"found": False})
    exact = qa_col.find_one({"question": question}, {"_id": 0})
    if exact:
        log.info("Q&A exact match  q=%r", question[:60])
        return jsonify({"found": True, **exact})
    match = fuzzy_find(question)
    if match:
        log.info("Q&A fuzzy match  sim=%.2f  q=%r", match["similarity"], question[:60])
        return jsonify({"found": True, **match})
    log.info("Q&A no match     q=%r", question[:60])
    return jsonify({"found": False})


# ── Unknown question queue ────────────────────────────────────────────────────

@app.route("/api/unknown-question", methods=["POST"])
def create_unknown_question():
    data  = request.json or {}
    q_id  = str(uuid.uuid4())
    question = data.get("question", "")
    questions_col.insert_one({
        "id":         q_id,
        "question":   question,
        "field_type": data.get("field_type", "text"),
        "options":    data.get("options", []),
        "job_url":    data.get("job_url", ""),
        "answered":   False,
        "answer":     None,
        "created_at": _now(),
    })
    log.info("Unknown question queued  id=%s  q=%r", q_id[:8], question[:80])
    return jsonify({"id": q_id})

@app.route("/api/unknown-question/<q_id>", methods=["GET"])
def poll_answer(q_id):
    q = questions_col.find_one({"id": q_id}, {"_id": 0})
    if not q:
        return jsonify({"error": "not found"}), 404
    return jsonify({"answered": q["answered"], "answer": q.get("answer")})

@app.route("/api/unknown-question/<q_id>/answer", methods=["POST"])
def submit_answer(q_id):
    data       = request.json or {}
    answer     = data.get("answer", "").strip()
    save_to_kb = data.get("save_to_kb", True)
    q = questions_col.find_one({"id": q_id}, {"_id": 0})
    if not q:
        return jsonify({"ok": False, "error": "not found"}), 404
    questions_col.update_one({"id": q_id}, {"$set": {"answered": True, "answer": answer}})
    log.info("Unknown question answered  id=%s  answer=%r  save_kb=%s",
             q_id[:8], answer[:60], save_to_kb)
    if save_to_kb and answer:
        qa_col.update_one(
            {"question": q["question"]},
            {"$set": {
                "question":   q["question"],
                "answer":     answer,
                "field_type": q.get("field_type", "text"),
                "updated_at": _now(),
            }},
            upsert=True,
        )
        log.info("Answer saved to knowledge base  q=%r", q["question"][:60])
    return jsonify({"ok": True})


# ── Q&A knowledge base ────────────────────────────────────────────────────────

@app.route("/api/qa-pairs")
def get_qa_pairs():
    pairs = [_serial(p) for p in
             qa_col.find({}, {"_id": 0}).sort("updated_at", DESCENDING)]
    return jsonify(pairs)

@app.route("/api/qa-pairs", methods=["POST"])
def add_qa_pair():
    data     = request.json or {}
    question = data.get("question", "").strip()
    answer   = data.get("answer", "").strip()
    if not question or not answer:
        return jsonify({"ok": False, "error": "question and answer required"}), 400
    qa_col.update_one(
        {"question": question},
        {"$set": {
            "question":   question,
            "answer":     answer,
            "field_type": data.get("field_type", "text"),
            "updated_at": _now(),
        }},
        upsert=True,
    )
    return jsonify({"ok": True})

@app.route("/api/qa-pairs/<path:question>", methods=["DELETE"])
def delete_qa_pair(question):
    qa_col.delete_one({"question": question})
    return jsonify({"ok": True})


# ── User profile ──────────────────────────────────────────────────────────────

@app.route("/api/profile")
def get_profile():
    p = profile_col.find_one({}, {"_id": 0, "password": 0})  # never return password
    return jsonify(p or {})

@app.route("/api/profile", methods=["POST"])
def save_profile():
    data   = request.json or {}
    update = {k: v for k, v in data.items() if k != "password" or v}
    update["updated_at"] = _now()
    profile_col.update_one({}, {"$set": update}, upsert=True)
    log.info("Profile updated  keys=%s", list(update.keys()))
    return jsonify({"ok": True})

@app.route("/api/profile/password", methods=["POST"])
def save_password():
    pwd = (request.json or {}).get("password", "").strip()
    if not pwd:
        return jsonify({"ok": False, "error": "empty password"}), 400
    profile_col.update_one({}, {"$set": {"password": pwd, "updated_at": _now()}}, upsert=True)
    log.info("Naukri password updated")
    return jsonify({"ok": True})

@app.route("/api/profile/bot-config")
def bot_config():
    """Bot calls this on startup to get credentials + preferences."""
    p = profile_col.find_one({}, {"_id": 0}) or {}
    return jsonify({
        "email":            p.get("email", ""),
        "password":         p.get("password", ""),
        "firstname":        p.get("firstname", ""),
        "lastname":         p.get("lastname", ""),
        "keywords":         p.get("keywords", []),
        "location":         p.get("location", ""),
        "max_applications": p.get("max_applications", 50),
        "pages_per_keyword":p.get("pages_per_keyword", 2),
    })


# ── Resume (S3) ───────────────────────────────────────────────────────────────

ALLOWED_EXT = {"pdf", "doc", "docx"}

def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/api/resume")
def get_resume():
    p = profile_col.find_one({}, {"_id": 0, "resume_url": 1,
                                   "resume_filename": 1, "resume_updated_at": 1})
    return jsonify(p or {})

@app.route("/api/resume/upload", methods=["POST"])
def upload_resume():
    if not AWS_BUCKET:
        return jsonify({"ok": False, "error": "AWS_S3_BUCKET not configured"}), 400

    file = request.files.get("resume")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "no file"}), 400
    if not _allowed(file.filename):
        return jsonify({"ok": False, "error": "only PDF/DOC/DOCX allowed"}), 400

    filename    = secure_filename(file.filename)
    s3_key      = f"resumes/{uuid.uuid4()}_{filename}"
    content_type = file.content_type or "application/octet-stream"

    try:
        s3 = _s3()
        log.info("Uploading resume to S3: bucket=%s key=%s", AWS_BUCKET, s3_key)
        s3.upload_fileobj(
            file,
            AWS_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        url = f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        profile_col.update_one(
            {},
            {"$set": {
                "resume_url":        url,
                "resume_filename":   filename,
                "resume_s3_key":     s3_key,
                "resume_updated_at": _now(),
            }},
            upsert=True,
        )
        log.info("Resume uploaded  url=%s", url)
        return jsonify({"ok": True, "url": url, "filename": filename})
    except Exception as e:
        log.error("Resume upload failed: %s", e, exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Context (free-form user background) ──────────────────────────────────────

@app.route("/api/context")
def get_context():
    items = [_serial(c) for c in
             context_col.find({}, {"_id": 0}).sort("updated_at", DESCENDING)]
    return jsonify(items)

@app.route("/api/context", methods=["POST"])
def save_context():
    data  = request.json or {}
    key   = data.get("key", "").strip()
    value = data.get("value", "").strip()
    if not key or not value:
        return jsonify({"ok": False, "error": "key and value required"}), 400
    context_col.update_one(
        {"key": key},
        {"$set": {"key": key, "value": value, "updated_at": _now()}},
        upsert=True,
    )
    log.info("Context saved  key=%r", key[:60])
    return jsonify({"ok": True})

@app.route("/api/context/<path:key>", methods=["DELETE"])
def delete_context(key):
    context_col.delete_one({"key": key})
    return jsonify({"ok": True})


if __name__ == "__main__":
    port  = int(os.getenv("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    log.info("Dashboard → http://localhost:%s  (debug=%s, log level: %s)", port, debug, LOG_LEVEL)
    app.run(debug=debug, host="0.0.0.0", port=port)
