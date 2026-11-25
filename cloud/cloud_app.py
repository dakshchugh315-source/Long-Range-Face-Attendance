"""Online dashboard + student portal (runs on a cloud server).

The classroom laptop does all camera work and face recognition. It pushes only
attendance RECORDS here (no photos, no face embeddings) every few seconds:
  POST /api/sync   Authorization: Bearer <SYNC_TOKEN>

Pages:
  /            teacher dashboard (password)      /student   student portal (roll no + PIN)

Environment:
  SYNC_TOKEN        shared secret with the laptop (long random string)
  TEACHER_PASSWORD  teacher login password
  SECRET_KEY        cookie signing key (random)
  DATA_DIR          where cloud.db lives (default: ./data)
"""
import csv
import hashlib
import hmac
import io
import json
import os
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from flask import Flask, Response, abort, jsonify, redirect, request, session

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).with_name("data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB = DATA_DIR / "cloud.db"
SYNC_TOKEN = os.environ.get("SYNC_TOKEN", "")
TEACHER_PASSWORD = os.environ.get("TEACHER_PASSWORD", "")
MAX_SYNC_BYTES = 5 * 1024 * 1024

app = Flask(__name__, static_folder=None)
if os.environ.get("BEHIND_PROXY", "1") == "1":
    # Render / Caddy sit in front: use the real client IP (last X-Forwarded-For hop) so the
    # wrong-PIN limit applies per student, not to everyone behind the proxy at once
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "1") == "1",
                  PERMANENT_SESSION_LIFETIME=60 * 60 * 12, MAX_CONTENT_LENGTH=MAX_SYNC_BYTES)
_fails = defaultdict(list)


def db():
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS students (name TEXT PRIMARY KEY, roll TEXT, pin_hash TEXT, salt TEXT);
        CREATE TABLE IF NOT EXISTS attendance (session TEXT, name TEXT, time TEXT, similarity REAL,
            method TEXT, votes INTEGER, zoom REAL, UNIQUE(session, name));
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);""")
    return con


def throttle(key, limit=8, window=600):
    _fails[key] = [t for t in _fails[key] if time.time() - t < window]
    if len(_fails[key]) >= limit:
        abort(429, "Too many wrong tries. Wait 10 minutes.")


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ------------------------------------------------------------------ sync from the laptop
@app.post("/api/sync")
def sync():
    auth = request.headers.get("Authorization", "")
    if not SYNC_TOKEN or not hmac.compare_digest(auth, f"Bearer {SYNC_TOKEN}"):
        throttle(f"sync:{request.remote_addr}", limit=20)
        _fails[f"sync:{request.remote_addr}"].append(time.time())
        abort(401)
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        abort(400)
    students = d.get("students") or []
    records = d.get("attendance") or []
    with closing(db()) as con:
        with con:                                   # one transaction: the online copy is replaced
            con.execute("DELETE FROM students")     # atomically (deletes on the laptop propagate)
            con.execute("DELETE FROM attendance")
            con.executemany("INSERT INTO students VALUES (?,?,?,?)",
                            [(str(s["name"]), s.get("roll"), s.get("pin_hash"), s.get("salt"))
                             for s in students])
            con.executemany("INSERT OR IGNORE INTO attendance VALUES (?,?,?,?,?,?,?)",
                            [(str(r["session"]), str(r["name"]), str(r["time"]), r.get("similarity"),
                              r.get("method"), r.get("votes"), r.get("zoom")) for r in records])
            con.execute("INSERT OR REPLACE INTO meta VALUES ('last_sync', ?)", (str(time.time()),))
            con.execute("INSERT OR REPLACE INTO meta VALUES ('live', ?)", (json.dumps(d.get("live") or {}),))
    return jsonify(ok=True, students=len(students), records=len(records))


def meta(con):
    m = {r["k"]: r["v"] for r in con.execute("SELECT * FROM meta")}
    last = float(m.get("last_sync", 0))
    return {"last_sync_s_ago": round(time.time() - last) if last else None,
            "live": json.loads(m.get("live", "{}"))}


# ------------------------------------------------------------------ teacher
@app.post("/login")
def login():
    throttle(f"t:{request.remote_addr}")
    pw = (request.get_json(silent=True) or {}).get("password", "")
    if not TEACHER_PASSWORD or not hmac.compare_digest(pw.encode(), TEACHER_PASSWORD.encode()):
        _fails[f"t:{request.remote_addr}"].append(time.time())
        return jsonify(ok=False, error="Wrong password"), 401
    session.clear()
    session["teacher"] = True
    return jsonify(ok=True)


def teacher_only():
    if not session.get("teacher"):
        abort(401)


@app.get("/api/overview")
def overview():
    teacher_only()
    with closing(db()) as con:
        names = [r["name"] for r in con.execute("SELECT name FROM students ORDER BY name")]
        sess = {}
        for r in con.execute("SELECT * FROM attendance ORDER BY time"):
            sess.setdefault(r["session"], []).append(dict(r))
        m = meta(con)
    out = [{"session": s, "present": rows, "absent": sorted(set(names) - {r["name"] for r in rows}),
            "started": rows[0]["time"]} for s, rows in sess.items()]
    out.sort(key=lambda x: x["started"], reverse=True)
    return jsonify(students=names, sessions=out, **m)


@app.get("/api/export")
def export():
    teacher_only()
    s = request.args.get("session", "")
    with closing(db()) as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM attendance WHERE session=? ORDER BY time", (s,))]
        names = [r["name"] for r in con.execute("SELECT name FROM students ORDER BY name")]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["session", "name", "status", "time", "how", "match"])
    for r in rows:
        w.writerow([s, r["name"], "Present", r["time"], "Face" if r["method"] == "auto" else "Manual", r["similarity"]])
    for n in sorted(set(names) - {r["name"] for r in rows}):
        w.writerow([s, n, "Absent", "", "", ""])
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=attendance_{safe}.csv"})


# ------------------------------------------------------------------ student
def _pin_ok(pin, pin_hash, salt):
    if not pin_hash or not salt:
        return False
    h = hashlib.pbkdf2_hmac("sha256", (pin or "").encode(), bytes.fromhex(salt), 120_000).hex()
    return hmac.compare_digest(h, pin_hash)


@app.post("/student/login")
def student_login():
    throttle(f"s:{request.remote_addr}")
    d = request.get_json(silent=True) or {}
    roll = (d.get("roll") or "").strip().upper()
    with closing(db()) as con:
        row = con.execute("SELECT * FROM students WHERE roll=?", (roll,)).fetchone()
    if not row or not _pin_ok(d.get("pin"), row["pin_hash"], row["salt"]):
        _fails[f"s:{request.remote_addr}"].append(time.time())
        return jsonify(ok=False, error="Roll no or PIN is wrong."), 401
    session.clear()
    session["student"] = row["name"]
    return jsonify(ok=True)


@app.get("/api/me")
def me():
    name = session.get("student")
    if not name:
        abort(401)
    with closing(db()) as con:
        recs = [dict(r) for r in con.execute(
            "SELECT session, time, method FROM attendance WHERE name=? ORDER BY time DESC", (name,))]
        total = con.execute("SELECT COUNT(DISTINCT session) FROM attendance").fetchone()[0]
        roll = con.execute("SELECT roll FROM students WHERE name=?", (name,)).fetchone()
        m = meta(con)
    return jsonify(name=name, roll=roll["roll"] if roll else "", records=recs, attended=len(recs),
                   total_sessions=total, last_sync_s_ago=m["last_sync_s_ago"])


@app.post("/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


# ------------------------------------------------------------------ pages
@app.get("/")
def teacher_page():
    return Response(Path(__file__).with_name("teacher.html").read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/student")
def student_page():
    return Response(Path(__file__).with_name("student.html").read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/healthz")
def health():
    with closing(db()) as con:
        return jsonify(ok=True, **meta(con))


if __name__ == "__main__":                       # local test only; production uses gunicorn
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8090)))
