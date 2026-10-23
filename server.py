"""
Dashboard server for Naukri Auto-Apply Bot — Multi-user edition.
Each user has their own Naukri account, job history, Q&A and bot process.
"""

import os
import sys
import uuid
import difflib
import logging
import subprocess
import threading
import bcrypt
import jwt
import boto3
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Flask, render_template, jsonify, request, g, redirect, url_for, session
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from authlib.integrations.flask_client import OAuth

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("naukri.server")
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MONGODB_URI        = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
AWS_BUCKET         = os.getenv("AWS_S3_BUCKET", "")
AWS_REGION         = os.getenv("AWS_REGION", "ap-south-1")
JWT_SECRET         = os.getenv("JWT_SECRET", "change-me-in-production-" + str(uuid.uuid4()))
JWT_EXP_DAYS       = int(os.getenv("JWT_EXP_DAYS", "30"))
GOOGLE_CLIENT_ID   = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*")}})

app.secret_key = os.getenv("FLASK_SECRET_KEY", JWT_SECRET)

# ── Google OAuth2 ─────────────────────────────────────────────────────────────
_oauth = OAuth(app)
google_oauth = _oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    log.info("MongoDB connected: %s", MONGODB_URI.split("@")[-1])
except Exception as exc:
    log.warning("MongoDB connection warning: %s", exc)
    client = MongoClient(MONGODB_URI)

db = client.naukri_bot

# Collections (all docs include user_id for scoping)
users_col     = db.users
jobs_col      = db.jobs
qa_col        = db.qa_pairs
questions_col = db.unknown_questions
profile_col   = db.profile
context_col   = db.context

# Bot processes per user  {user_id: Popen}
_bot_procs = {}
_bot_logs  = {}   # {user_id: deque}
_bot_lock  = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serial(doc):
    if doc and "_id" in doc:
        doc.pop("_id")
    return doc

def _now():
    return datetime.utcnow().isoformat()

def _uid():
    return str(uuid.uuid4())

def fuzzy_find(question_text, user_id, threshold=0.72):
    all_qa = list(qa_col.find({"user_id": user_id}, {"_id": 0}))
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
        "s3", region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


# ── Auth helpers ──────────────────────────────────────────────────────────────

def make_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXP_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def decode_token(token: str):
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

def require_auth(f):
    """Decorator — validates JWT and sets g.user_id / g.email."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        try:
            payload = decode_token(auth[7:])
            g.user_id = payload["sub"]
            g.email   = payload["email"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return wrapper


# ── Request logging ───────────────────────────────────────────────────────────

@app.before_request
def _log_req():
    g.req_id = _uid()[:8]
    if request.path.startswith("/api"):
        log.info("[%s] → %s %s", g.req_id, request.method, request.path)

@app.after_request
def _log_res(response):
    if request.path.startswith("/api"):
        log.info("[%s] ← %s %s %s", g.req_id, request.method, request.path, response.status_code)
    return response


# ── Dashboard page ────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("index.html")


# ── Google OAuth2 routes ──────────────────────────────────────────────────────

@app.route("/auth/google")
def auth_google():
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Google OAuth not configured — add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env"}), 503
    redirect_uri = url_for("auth_google_callback", _external=True)
    return google_oauth.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_google_callback():
    try:
        token_data = google_oauth.authorize_access_token()
    except Exception as exc:
        log.error("Google OAuth callback error: %s", exc)
        return redirect("/?auth_error=" + str(exc))

    userinfo = token_data.get("userinfo") or {}
    email = (userinfo.get("email") or "").lower()
    name  = userinfo.get("name", "")

    if not email:
        return redirect("/?auth_error=no_email")

    user = users_col.find_one({"email": email})
    if not user:
        user_id = _uid()
        users_col.insert_one({
            "id": user_id, "email": email, "name": name,
            "auth_provider": "google", "created_at": _now(),
        })
        log.info("New user via Google OAuth: %s", email)
    else:
        user_id = user["id"]
        users_col.update_one({"id": user_id}, {"$set": {"auth_provider": "google", "name": name}})
        log.info("Existing user via Google OAuth: %s", email)

    jwt_token = make_token(user_id, email)
    return redirect(f"/?token={jwt_token}&name={name}")


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data     = request.json or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    name     = data.get("name", "").strip()

    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password required"}), 400
    if len(password) < 6:
        return jsonify({"ok": False, "error": "Password must be at least 6 characters"}), 400
    if users_col.find_one({"email": email}):
        return jsonify({"ok": False, "error": "Email already registered"}), 409

    user_id  = _uid()
    pw_hash  = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users_col.insert_one({
        "id": user_id, "email": email, "name": name,
        "password": pw_hash, "created_at": _now(),
    })
    token = make_token(user_id, email)
    log.info("New user signed up: %s", email)
    return jsonify({"ok": True, "token": token, "user": {"id": user_id, "email": email, "name": name}})


@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.json or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    user = users_col.find_one({"email": email})
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return jsonify({"ok": False, "error": "Invalid email or password"}), 401

    token = make_token(user["id"], email)
    log.info("User logged in: %s", email)
    return jsonify({"ok": True, "token": token,
                    "user": {"id": user["id"], "email": email, "name": user.get("name", "")}})


@app.route("/api/auth/me")
@require_auth
def me():
    user = users_col.find_one({"id": g.user_id}, {"_id": 0, "password": 0})
    return jsonify(user or {})


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.route("/api/stats")
@require_auth
def stats():
    uid     = g.user_id
    applied = jobs_col.count_documents({"user_id": uid, "status": "applied"})
    skipped = jobs_col.count_documents({"user_id": uid, "status": "skipped"})
    failed  = jobs_col.count_documents({"user_id": uid, "status": "failed"})
    recent  = [_serial(j) for j in
               jobs_col.find({"user_id": uid}, {"_id": 0}).sort("timestamp", DESCENDING).limit(30)]
    pending = [_serial(q) for q in
               questions_col.find({"user_id": uid, "answered": False}, {"_id": 0})
               .sort("created_at", DESCENDING)]
    return jsonify({
        "applied": applied, "skipped": skipped, "failed": failed,
        "recent_jobs": recent, "pending_questions": pending,
        "qa_count": qa_col.count_documents({"user_id": uid}),
    })


# ── Job logging ───────────────────────────────────────────────────────────────

@app.route("/api/job", methods=["POST"])
@require_auth
def log_job():
    data   = request.json or {}
    status = data.get("status", "unknown")
    title  = data.get("title", "")
    jobs_col.insert_one({
        "user_id": g.user_id, "url": data.get("url", ""),
        "title": title, "company": data.get("company", ""),
        "status": status, "timestamp": _now(),
    })
    log.info("Job [%s] %s %s", g.user_id[:8], status.upper(), title or "?")
    return jsonify({"ok": True})


# ── Q&A lookup (bot uses this) ────────────────────────────────────────────────

@app.route("/api/answer-lookup", methods=["POST"])
@require_auth
def answer_lookup():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"found": False})
    exact = qa_col.find_one({"user_id": g.user_id, "question": question}, {"_id": 0})
    if exact:
        log.info("Q&A exact match  q=%r", question[:60])
        return jsonify({"found": True, **exact})
    match = fuzzy_find(question, g.user_id)
    if match:
        log.info("Q&A fuzzy match  sim=%.2f", match["similarity"])
        return jsonify({"found": True, **match})
    return jsonify({"found": False})


# ── Unknown question queue ────────────────────────────────────────────────────

@app.route("/api/unknown-question", methods=["POST"])
@require_auth
def create_unknown_question():
    data = request.json or {}
    q_id = _uid()
    questions_col.insert_one({
        "user_id": g.user_id, "id": q_id,
        "question": data.get("question", ""),
        "field_type": data.get("field_type", "text"),
        "options": data.get("options", []),
        "job_url": data.get("job_url", ""),
        "answered": False, "answer": None, "created_at": _now(),
    })
    log.info("Unknown Q queued  id=%s  q=%r", q_id[:8], data.get("question", "")[:60])
    return jsonify({"id": q_id})

@app.route("/api/unknown-question/<q_id>", methods=["GET"])
@require_auth
def poll_answer(q_id):
    q = questions_col.find_one({"user_id": g.user_id, "id": q_id}, {"_id": 0})
    if not q:
        return jsonify({"error": "not found"}), 404
    return jsonify({"answered": q["answered"], "answer": q.get("answer")})

@app.route("/api/unknown-question/<q_id>/answer", methods=["POST"])
@require_auth
def submit_answer(q_id):
    data       = request.json or {}
    answer     = data.get("answer", "").strip()
    save_to_kb = data.get("save_to_kb", True)
    q = questions_col.find_one({"user_id": g.user_id, "id": q_id}, {"_id": 0})
    if not q:
        return jsonify({"ok": False, "error": "not found"}), 404
    questions_col.update_one({"id": q_id}, {"$set": {"answered": True, "answer": answer}})
    log.info("Q answered  id=%s  ans=%r", q_id[:8], answer[:60])
    if save_to_kb and answer:
        qa_col.update_one(
            {"user_id": g.user_id, "question": q["question"]},
            {"$set": {"user_id": g.user_id, "question": q["question"],
                      "answer": answer, "field_type": q.get("field_type", "text"),
                      "updated_at": _now()}},
            upsert=True,
        )
    return jsonify({"ok": True})


# ── Q&A Knowledge Base ────────────────────────────────────────────────────────

@app.route("/api/qa-pairs")
@require_auth
def get_qa_pairs():
    pairs = [_serial(p) for p in
             qa_col.find({"user_id": g.user_id}, {"_id": 0}).sort("updated_at", DESCENDING)]
    return jsonify(pairs)

@app.route("/api/qa-pairs", methods=["POST"])
@require_auth
def add_qa_pair():
    data     = request.json or {}
    question = data.get("question", "").strip()
    answer   = data.get("answer", "").strip()
    if not question or not answer:
        return jsonify({"ok": False, "error": "question and answer required"}), 400
    qa_col.update_one(
        {"user_id": g.user_id, "question": question},
        {"$set": {"user_id": g.user_id, "question": question, "answer": answer,
                  "field_type": data.get("field_type", "text"), "updated_at": _now()}},
        upsert=True,
    )
    return jsonify({"ok": True})

@app.route("/api/qa-pairs/<path:question>", methods=["DELETE"])
@require_auth
def delete_qa_pair(question):
    qa_col.delete_one({"user_id": g.user_id, "question": question})
    return jsonify({"ok": True})


# ── Naukri Account (per user) ─────────────────────────────────────────────────

@app.route("/api/naukri/connect", methods=["POST"])
@require_auth
def naukri_connect():
    data         = request.json or {}
    login_method = data.get("login_method", "password")  # "password" | "google"
    update       = {
        "user_id": g.user_id,
        "naukri_login_method":   login_method,
        "naukri_connected_at":   _now(),
        "updated_at":            _now(),
    }
    if login_method == "google":
        g_email = data.get("google_email", "").strip()
        g_pwd   = data.get("google_password", "").strip()
        if not g_email or not g_pwd:
            return jsonify({"ok": False, "error": "Google email and password required"}), 400
        update["naukri_google_email"]    = g_email
        update["naukri_google_password"] = g_pwd
        update["naukri_email"]           = g_email
        display_email = g_email
    else:
        email = data.get("email", "").strip()
        pwd   = data.get("password", "").strip()
        if not email or not pwd:
            return jsonify({"ok": False, "error": "Email and password required"}), 400
        update["naukri_email"]    = email
        update["naukri_password"] = pwd
        display_email = email

    profile_col.update_one({"user_id": g.user_id}, {"$set": update}, upsert=True)
    log.info("Naukri connected  user=%s  method=%s  email=%s", g.user_id[:8], login_method, display_email)
    return jsonify({"ok": True, "email": display_email, "login_method": login_method})

@app.route("/api/naukri/disconnect", methods=["POST"])
@require_auth
def naukri_disconnect():
    profile_col.update_one(
        {"user_id": g.user_id},
        {"$unset": {"naukri_email": "", "naukri_password": "", "naukri_connected_at": ""}},
    )
    log.info("Naukri disconnected  user=%s", g.user_id[:8])
    return jsonify({"ok": True})

@app.route("/api/naukri/status")
@require_auth
def naukri_status():
    p = profile_col.find_one({"user_id": g.user_id},
                              {"_id": 0, "naukri_password": 0, "naukri_google_password": 0})
    connected = bool(p and p.get("naukri_email"))
    return jsonify({
        "connected":    connected,
        "email":        p.get("naukri_email", "") if p else "",
        "login_method": p.get("naukri_login_method", "password") if p else "password",
        "connected_at": p.get("naukri_connected_at", "") if p else "",
    })


# ── Full profile ──────────────────────────────────────────────────────────────

@app.route("/api/profile")
@require_auth
def get_profile():
    p = profile_col.find_one({"user_id": g.user_id}, {"_id": 0, "naukri_password": 0})
    return jsonify(p or {})

@app.route("/api/profile", methods=["POST"])
@require_auth
def save_profile():
    data   = request.json or {}
    update = {k: v for k, v in data.items() if k not in ("naukri_password",)}
    update.update({"user_id": g.user_id, "updated_at": _now()})
    profile_col.update_one({"user_id": g.user_id}, {"$set": update}, upsert=True)
    log.info("Profile updated  user=%s", g.user_id[:8])
    return jsonify({"ok": True})

@app.route("/api/profile/bot-config")
@require_auth
def bot_config():
    """Bot calls this on startup to get credentials + preferences."""
    p = profile_col.find_one({"user_id": g.user_id}, {"_id": 0}) or {}
    return jsonify({
        "email":              p.get("naukri_email", ""),
        "password":           p.get("naukri_password", ""),
        "login_method":       p.get("naukri_login_method", "password"),
        "google_email":       p.get("naukri_google_email", ""),
        "google_password":    p.get("naukri_google_password", ""),
        "firstname":          p.get("firstname", ""),
        "lastname":           p.get("lastname", ""),
        "keywords":           p.get("keywords", []),
        "location":           p.get("location", ""),
        "max_applications":   p.get("max_applications", 50),
        "pages_per_keyword":  p.get("pages_per_keyword", 2),
    })


# ── Resume (S3) ───────────────────────────────────────────────────────────────

ALLOWED_EXT = {"pdf", "doc", "docx"}

@app.route("/api/resume")
@require_auth
def get_resume():
    p = profile_col.find_one({"user_id": g.user_id}, {"_id": 0, "resume_url": 1,
                               "resume_filename": 1, "resume_updated_at": 1})
    return jsonify(p or {})

@app.route("/api/resume/upload", methods=["POST"])
@require_auth
def upload_resume():
    if not AWS_BUCKET:
        return jsonify({"ok": False, "error": "AWS_S3_BUCKET not configured"}), 400
    file = request.files.get("resume")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "no file"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": "only PDF/DOC/DOCX allowed"}), 400
    filename    = secure_filename(file.filename)
    s3_key      = f"resumes/{g.user_id}/{uuid.uuid4()}_{filename}"
    content_type = file.content_type or "application/octet-stream"
    try:
        log.info("Uploading resume  user=%s  key=%s", g.user_id[:8], s3_key)
        _s3().upload_fileobj(file, AWS_BUCKET, s3_key, ExtraArgs={"ContentType": content_type})
        url = f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        profile_col.update_one(
            {"user_id": g.user_id},
            {"$set": {"resume_url": url, "resume_filename": filename,
                      "resume_s3_key": s3_key, "resume_updated_at": _now()}},
            upsert=True,
        )
        log.info("Resume uploaded  url=%s", url)
        return jsonify({"ok": True, "url": url, "filename": filename})
    except Exception as e:
        log.error("Resume upload failed: %s", e, exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Context ───────────────────────────────────────────────────────────────────

@app.route("/api/context")
@require_auth
def get_context():
    items = [_serial(c) for c in
             context_col.find({"user_id": g.user_id}, {"_id": 0}).sort("updated_at", DESCENDING)]
    return jsonify(items)

@app.route("/api/context", methods=["POST"])
@require_auth
def save_context():
    data  = request.json or {}
    key   = data.get("key", "").strip()
    value = data.get("value", "").strip()
    if not key or not value:
        return jsonify({"ok": False, "error": "key and value required"}), 400
    context_col.update_one(
        {"user_id": g.user_id, "key": key},
        {"$set": {"user_id": g.user_id, "key": key, "value": value, "updated_at": _now()}},
        upsert=True,
    )
    log.info("Context saved  user=%s  key=%r", g.user_id[:8], key[:40])
    return jsonify({"ok": True})

@app.route("/api/context/<path:key>", methods=["DELETE"])
@require_auth
def delete_context(key):
    context_col.delete_one({"user_id": g.user_id, "key": key})
    return jsonify({"ok": True})


# ── Bot process management (per user) ────────────────────────────────────────

def _stream_bot_logs(proc, user_id):
    logs = _bot_logs[user_id]
    for line in proc.stdout:
        line = line.rstrip()
        logs.append(line)
        log.info("[BOT:%s] %s", user_id[:8], line)
    proc.wait()
    logs.append(f"--- Bot exited (code {proc.returncode}) ---")
    log.info("Bot exited  user=%s  code=%s", user_id[:8], proc.returncode)


@app.route("/api/bot/status")
@require_auth
def bot_status():
    uid  = g.user_id
    proc = _bot_procs.get(uid)
    running = proc is not None and proc.poll() is None
    logs = list(_bot_logs.get(uid, deque()))
    return jsonify({"running": running, "logs": logs})


@app.route("/api/bot/start", methods=["POST"])
@require_auth
def bot_start():
    uid  = g.user_id
    data = request.json or {}
    tabs = data.get("tabs", "3")

    with _bot_lock:
        proc = _bot_procs.get(uid)
        if proc and proc.poll() is None:
            return jsonify({"ok": False, "error": "Bot already running"}), 400

        logs = deque(maxlen=200)
        logs.append("Starting bot…")
        _bot_logs[uid] = logs

        bot_script = os.path.join(os.path.dirname(__file__), "Naukri-Recommended.py")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["BOT_USER_ID"]      = uid
        env["BOT_AUTH_TOKEN"]   = make_token(uid, g.email)

        proc = subprocess.Popen(
            [sys.executable, bot_script, "--tabs", str(tabs)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
            cwd=os.path.dirname(__file__),
        )
        _bot_procs[uid] = proc
        threading.Thread(target=_stream_bot_logs, args=(proc, uid), daemon=True).start()

    log.info("Bot started  user=%s  pid=%s  tabs=%s", uid[:8], proc.pid, tabs)
    return jsonify({"ok": True, "pid": proc.pid})


@app.route("/api/bot/stop", methods=["POST"])
@require_auth
def bot_stop():
    uid  = g.user_id
    proc = _bot_procs.get(uid)
    if not proc or proc.poll() is not None:
        return jsonify({"ok": False, "error": "Bot is not running"}), 400
    proc.terminate()
    _bot_logs.get(uid, deque()).append("--- Stopped by user ---")
    log.info("Bot stopped by user  user=%s", uid[:8])
    return jsonify({"ok": True})


if __name__ == "__main__":
    port  = int(os.getenv("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    log.info("Dashboard → http://localhost:%s  (debug=%s, log=%s)", port, debug, LOG_LEVEL)
    app.run(debug=debug, host="0.0.0.0", port=port)
