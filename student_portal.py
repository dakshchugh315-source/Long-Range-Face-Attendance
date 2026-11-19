"""Student portal - separate small web app, reachable from phones on the same Wi-Fi.

Students log in with roll no + PIN and see only their own attendance.
No teacher controls exist here.
"""
import secrets
import time
from collections import defaultdict

from flask import Flask, abort, jsonify, redirect, request, send_from_directory, session

import config
import students_db
from attend import open_db
from notify import load_secrets, save_secret

portal = Flask("student_portal", static_folder=None)
key = load_secrets().get("portal_secret")
if not key:
    key = secrets.token_hex(32)
    save_secret("portal_secret", key)
portal.secret_key = key
portal.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                     PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30)
_fails = defaultdict(list)     # ip -> failed login times


@portal.get("/")
def home():
    return send_from_directory(config.ROOT / "web", "student.html")


@portal.post("/login")
def login():
    ip = request.remote_addr
    _fails[ip] = [t for t in _fails[ip] if time.time() - t < 600]
    if len(_fails[ip]) >= 8:
        return jsonify(ok=False, error="Too many wrong tries. Wait 10 minutes."), 429
    d = request.get_json(force=True)
    con = open_db()
    try:
        name = students_db.verify(con, d.get("roll"), d.get("pin"))
    finally:
        con.close()
    if not name:
        _fails[ip].append(time.time())
        return jsonify(ok=False, error="Roll no or PIN is wrong."), 401
    session.clear()
    session.permanent = True
    session["name"] = name
    return jsonify(ok=True)


@portal.post("/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@portal.get("/api/me")
def me():
    name = session.get("name")
    if not name:
        return jsonify(error="login"), 401
    con = open_db()
    try:
        hist = students_db.history(con, name)
        sessions = [r[0] for r in con.execute("SELECT DISTINCT session FROM attendance")]
        roll = con.execute("SELECT roll FROM students WHERE name=?", (name,)).fetchone()
    finally:
        con.close()
    return jsonify(name=name, roll=roll[0] if roll else "", records=hist,
                   attended=len({h["session"] for h in hist}), total_sessions=len(sessions))


@portal.get("/myface/<fname>")
def myface(fname):
    """Own proof photo only."""
    name = session.get("name")
    if not name:
        abort(401)
    con = open_db()
    try:
        ok = con.execute("SELECT 1 FROM attendance WHERE name=? AND snapshot=?", (name, fname)).fetchone()
    finally:
        con.close()
    if not ok:
        abort(404)
    return send_from_directory(config.SNAPSHOT_DIR, fname)
