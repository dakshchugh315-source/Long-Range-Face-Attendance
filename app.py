"""Face Attendance - desktop web app.

Run:   python app.py        (or double-click run_app.bat)
Opens http://127.0.0.1:5000 in your browser. Only reachable from this laptop.
"""
import os

# RTSP (phone H.264 / CCTV) over TCP: no lost packets / grey smeared frames on Wi-Fi
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import contextlib
import csv
import io
import json
import re
import sqlite3
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

import config
import students_db
from attend import LatestFrame, Voter, mark_manual, open_db, process_frame
from face_engine import Gallery, ensure_bgr, get_app
from notify import TelegramBot

PORTAL_PORT = 5001
bot = TelegramBot(open_db)

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".mp4", ".mov"}
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# ====================================================================== engine
class Engine:
    """Runs camera + recognition in a background thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.jpeg = None
        self.session = None
        self.source = None
        self.fps = 0.0
        self.error = None
        self.enrolled = []
        self.present = set()
        self.events = deque(maxlen=60)
        self.manual_q = deque()          # teacher add/remove, applied by the camera thread only
        self.timing = {}                 # ms per stage (EMA): wait, detect, recognise, encode

    def _make_autozoom(self, source, con, session):
        """Pick a zoom driver for this camera, or explain why there is none."""
        from autozoom import AutoZoom, OnvifDriver, PhoneDriver
        from notify import load_secrets
        src = str(source)

        def log_event(ev):
            con.execute("INSERT INTO zoom_events(session, time, action, zoom, x, y, face_px, reason, "
                        "seconds, recognised) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (session, ev["time"], ev["action"], ev["zoom"], ev["x"], ev["y"],
                         ev["face_px"], ev["reason"], ev["seconds"], ev["recognised"]))
            con.commit()
            self.events.appendleft({"time": ev["time"][11:], "text":
                                    f"Auto-zoom {ev['zoom']}x {ev['action']}: "
                                    f"{ev['recognised'] or 'nobody new'}"})

        from autozoom import is_ip_webcam
        hp = re.match(r"^(?:https?|rtsp)://(?:[^@/]*@)?([^/]+)", src)    # host:port of the stream
        if hp and is_ip_webcam(f"http://{hp.group(1)}"):                # any port, not only 8080
            driver = PhoneDriver(f"http://{hp.group(1)}",
                                 load_secrets().get("phone_pan_mode", "free_range"))
        elif src.startswith(("http://", "https://")):
            raise RuntimeError("Phone zoom control not reachable (is IP Webcam running?).")
        else:
            m = re.match(r"^rtsp://(?:([^:@/]+):([^@/]*)@)?([^:/]+)", src)
            if not m:
                raise RuntimeError("Auto-zoom needs the phone (IP Webcam) or a PTZ CCTV (rtsp://...). "
                                   "Laptop webcam cannot zoom.")
            user, pw, host = m.group(1) or "admin", m.group(2) or "", m.group(3)
            try:
                driver = OnvifDriver(host, config.ONVIF_PORT, user, pw)
            except Exception as e:
                raise RuntimeError(f"This CCTV has no ONVIF zoom ({type(e).__name__}). It is probably "
                                   "a fixed camera: turn auto-zoom off - software zoom (tiling) is "
                                   "already used on every frame.")
        return AutoZoom(driver, log_event)

    def start(self, source, session, autozoom=False):
        if self.running:
            raise RuntimeError("Already running. Stop first.")
        try:
            gallery = Gallery()
        except SystemExit as e:             # Gallery uses SystemExit for CLI messages
            raise RuntimeError(str(e))
        src = int(source) if str(source).strip().isdigit() else str(source).strip()
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera: {source}. "
                               "Phone: same Wi-Fi? URL ends with /video?")
        con = open_db()
        present = {r[0] for r in con.execute(
            "SELECT name FROM attendance WHERE session=?", (session,))}
        self.autozoom = None
        self._az_request = None
        if autozoom:
            try:
                self.autozoom = self._make_autozoom(source, con, session)
            except RuntimeError:
                cap.release()
                con.close()
                raise
        self.con = con
        self.session, self.source, self.error = session, source, None
        self.enrolled, self.present, self.jpeg = gallery.names, present, None
        self.events.clear()
        self.manual_q.clear()
        self.timing = {}
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(cap, gallery, con), daemon=True)
        self.thread.start()

    def _reconnect(self, timeout=60):
        src = int(self.source) if str(self.source).isdigit() else self.source
        t0 = time.time()
        while self.running and time.time() - t0 < timeout:
            self.error = f"Camera lost - reconnecting... ({int(time.time() - t0)} s)"
            self.events.appendleft({"time": datetime.now().strftime("%H:%M:%S"),
                                    "text": "Camera stream dropped, reconnecting"}) \
                if time.time() - t0 < 1 else None
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                reader = LatestFrame(cap)
                ok, _ = reader.read()
                if ok:
                    self.error = None
                    self.events.appendleft({"time": datetime.now().strftime("%H:%M:%S"),
                                            "text": "Camera reconnected"})
                    return cap, reader
                reader.close()
            else:
                cap.release()
            time.sleep(2)
        return None, None

    def set_autozoom(self, on):
        """Switch auto-zoom on/off in the middle of a session.

        Only the camera-loop thread ever installs, steps or stops the controller:
        this just leaves it a request, so two threads can never move the camera."""
        if not self.running:
            raise RuntimeError("Start attendance first.")
        with self.az_lock:
            pending = self._az_request
            if on:
                if pending == "off":
                    self._az_request = None          # cancel a not-yet-done switch-off
                elif self.autozoom is None and pending is None:
                    # driver set-up talks to the phone; the controller is handed to the loop
                    self._az_request = self._make_autozoom(self.source, self.con, self.session)
            else:
                if pending not in (None, "off"):     # created but never installed: discard it
                    pending.stop()
                    self._az_request = None
                elif self.autozoom is not None:
                    self._az_request = "off"

    def autozoom_active(self):
        return self.running and (self.autozoom is not None or self._az_request not in (None, "off"))

    az_lock = threading.Lock()
    _az_request = None
    autozoom = None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self, cap, gallery, con):
        is_file = isinstance(self.source, str) and Path(self.source).is_file()
        reader = cap if is_file else LatestFrame(cap)
        tracks, t_prev, voter = [], time.time(), Voter()
        errors_in_row = 0
        done_times = deque()
        self.fps = 0.0

        def ema(key, seconds):
            self.timing[key] = round(0.8 * self.timing.get(key, seconds * 1000) + 0.2 * seconds * 1000)
        try:
            while self.running:
                t_wait = time.time()
                ok, frame = reader.read()
                ema("wait", time.time() - t_wait)
                if not ok:
                    if is_file:
                        self.error = "Video file finished."
                        break
                    # Wi-Fi / phone hiccup: reconnect instead of ending the class
                    reader.close()
                    cap, reader = self._reconnect()
                    if cap is None:
                        self.error = "Camera disconnected and did not come back for 60 s."
                        break
                    continue
                with self.az_lock:                     # apply on/off requests here, in this thread only
                    req, self._az_request = self._az_request, None
                if req == "off" and self.autozoom is not None:
                    self.autozoom.stop()               # back to 1x; logs the last move
                    self.autozoom = None
                    tracks.clear()
                elif req not in (None, "off") and self.autozoom is None:
                    self.autozoom = req
                    tracks.clear()
                while self.manual_q:                   # teacher's Mark present / Remove
                    action, name = self.manual_q.popleft()
                    (self.present.add if action == "add" else self.present.discard)(name)
                az = self.autozoom
                info = {}
                try:
                    frame = ensure_bgr(frame)
                    newly = process_frame(frame, tracks, gallery, self.present, con,
                                          self.session, self.source, voter,
                                          zoom=az.zoom if az else 1.0, info=info,
                                          camera_moving=bool(az and az.settling()))
                    errors_in_row = 0
                except Exception as e:                 # one bad frame must not end the class
                    errors_in_row += 1
                    self.events.appendleft({"time": datetime.now().strftime("%H:%M:%S"),
                                            "text": f"Frame skipped: {type(e).__name__}: {e}"})
                    if errors_in_row >= 10:
                        raise
                    continue
                ema("detect", info.get("t_detect", 0))
                ema("recognise", info.get("t_recog", 0))
                if az:
                    absent = len(set(self.enrolled) - self.present)
                    if az.step(info.get("faces", []), frame.shape, self.present, absent,
                               info.get("recognised", set())):
                        tracks.clear()                 # view moved: old boxes mean nothing now
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
                    cv2.putText(frame, az.label() + ("  (moving...)" if az.settling() else ""),
                                (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                for name, sim, px in newly:
                    bot.notify(name, self.session, "face", sim)
                    self.events.appendleft({"time": datetime.now().strftime("%H:%M:%S"),
                                            "text": f"{name} marked present "
                                                    f"(match {sim:.2f}, face {px}px)"})
                now = time.time()
                # real throughput: frames actually processed in the last 5 s (an average of
                # 1/frame-time is biased upwards when fast and slow frames alternate)
                done_times.append(now)
                while done_times and now - done_times[0] > 5:
                    done_times.popleft()
                if len(done_times) > 1:
                    self.fps = (len(done_times) - 1) / (done_times[-1] - done_times[0])
                t_prev = now
                t_enc = time.time()
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.jpeg = buf.tobytes()
                ema("encode", time.time() - t_enc)
        except Exception as e:
            self.error = f"Error: {e}"
        finally:
            self.running = False
            if getattr(self, "autozoom", None):
                self.autozoom.stop()               # back to 1x, log the last move
            if isinstance(reader, LatestFrame):
                reader.close()
            elif cap is not None:
                cap.release()
            con.close()


engine = Engine()
enroll_state = {"running": False, "output": ""}


# ====================================================================== helpers
def db():
    return open_db()


def rows(con, session):
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(
        "SELECT * FROM attendance WHERE session=? ORDER BY time", (session,))]


def enrolled_names():
    try:
        return Gallery().names
    except SystemExit:
        return []


def audit(con, session, name, action):
    con.execute("CREATE TABLE IF NOT EXISTS audit_log (time TEXT, session TEXT, name TEXT, action TEXT)")
    con.execute("INSERT INTO audit_log VALUES (?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session, name, action))
    con.commit()


def safe_student_name(name):
    name = re.sub(r"\s+", " ", (name or "").strip())
    if not re.fullmatch(r"[A-Za-z0-9 ._-]{1,40}", name) or name.startswith("."):
        abort(400, "Name: letters, numbers, spaces, . _ - only (max 40)")
    return name


# ====================================================================== request safety
LOCAL_NAMES = {"127.0.0.1", "localhost"}


def _hostname(hostport):
    return hostport.rsplit(":", 1)[0] if hostport.count(":") == 1 else hostport


@app.before_request
def block_cross_site():
    """The teacher app is only for this laptop. Without this, ANY website open in the
    browser could silently POST to 127.0.0.1:5000 (CSRF) and e.g. delete all records,
    and DNS-rebinding pages could read it. Reject other Hosts and cross-site writes."""
    if _hostname(request.host) not in LOCAL_NAMES:
        abort(403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
        if origin:
            host = origin.split("://", 1)[-1].split("/", 1)[0]
            if host != request.host:           # must come from this very app (same host:port)
                abort(403)
        if request.mimetype not in ("application/json", "multipart/form-data"):
            abort(415)                     # cross-site forms can't send JSON without permission


# ====================================================================== pages
@app.get("/")
def index():
    return send_from_directory(config.ROOT / "web", "index.html")


@app.get("/video")
def video():
    def gen():
        last = None
        while True:
            frame = engine.jpeg
            if frame is not None and frame is not last:
                last = frame
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if not engine.running and frame is last:
                time.sleep(0.5)
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.get("/proof/<path:fname>")
def proof(fname):
    return send_from_directory(config.SNAPSHOT_DIR, fname)


@app.get("/student_photo/<name>")
def student_photo(name):
    folder = config.DATASET_DIR / safe_student_name(name)
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return send_file(p)
    abort(404)


# ====================================================================== live API
@app.post("/api/start")
def api_start():
    d = request.get_json(force=True)
    session = (d.get("session") or datetime.now().strftime("%Y-%m-%d")).strip()[:60]
    try:
        engine.start(d.get("source", config.CAMERA_SOURCE), session, bool(d.get("autozoom")))
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True)


@app.post("/api/autozoom")
def api_autozoom():
    try:
        engine.set_autozoom(bool(request.get_json(force=True).get("on")))
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True)


@app.post("/api/stop")
def api_stop():
    engine.stop()
    return jsonify(ok=True)


@app.get("/api/status")
def api_status():
    session = request.args.get("session") or engine.session
    present = []
    if session:
        with contextlib.closing(db()) as con:
            present = rows(con, session)
            sent = students_db.notification_status(con, session)
        for r in present:
            r["notified"] = sent.get(r["name"], "pending" if bot.token else "no bot")
    names = engine.enrolled if engine.running else enrolled_names()
    marked = {r["name"] for r in present}
    az = getattr(engine, "autozoom", None) if engine.running else None
    return jsonify(autozoom=(az.label() if az else None), autozoom_error=(az.error if az else None),
                   autozoom_state=(az.scan.state if az else None),
                   camera_state=(az.camera_state() if az else None),
                   running=engine.running, session=session, source=engine.source,
                   fps=round(engine.fps, 1), error=engine.error, timing=engine.timing,
                   enrolled=len(names), present=present,
                   absent=sorted(set(names) - marked), events=list(engine.events))


@app.post("/api/manual")
def api_manual():
    d = request.get_json(force=True)
    session, name, action = d.get("session"), d.get("name"), d.get("action")
    if not session or not name or action not in {"add", "remove"}:
        abort(400)
    with contextlib.closing(db()) as con:
        if action == "add":
            if mark_manual(con, session, name):
                bot.notify(name, session, "manual")
            if engine.running and engine.session == session:
                engine.manual_q.append(("add", name))
        else:
            con.execute("DELETE FROM attendance WHERE session=? AND name=?", (session, name))
            con.commit()
            if engine.running and engine.session == session:
                engine.manual_q.append(("remove", name))
        audit(con, session, name, f"manual {action}")
    return jsonify(ok=True)


# ====================================================================== records
@app.get("/api/sessions")
def api_sessions():
    with contextlib.closing(db()) as con:
        data = con.execute("SELECT session, COUNT(*), MIN(time) FROM attendance "
                           "GROUP BY session ORDER BY MIN(time) DESC").fetchall()
    total = len(enrolled_names())
    return jsonify([{"session": s, "present": n, "total": total, "started": t} for s, n, t in data])


@app.post("/api/delete")
def api_delete():
    """Delete recorded attendance: one session, or everything ({"all": true}).
    Removes attendance rows, their proof photos, Telegram status and zoom log.
    Students / dataset / enrollment / PINs are NOT touched."""
    d = request.get_json(force=True)
    every, session = bool(d.get("all")), d.get("session")
    if not every and not session:
        abort(400)
    if engine.running and (every or engine.session == session):
        return jsonify(ok=False, error="Stop the camera first."), 400
    where, args = ("", ()) if every else (" WHERE session=?", (session,))
    with contextlib.closing(db()) as con:
        files = [f for r in con.execute(f"SELECT snapshot, frame_snapshot FROM attendance{where}", args)
                 for f in r if f]
        n = con.execute(f"DELETE FROM attendance{where}", args).rowcount
        for table in ("notifications", "zoom_events"):
            try:
                con.execute(f"DELETE FROM {table}{where}", args)
            except sqlite3.OperationalError:
                pass                                   # table not created yet
        con.commit()
        audit(con, "ALL" if every else session, "-", f"deleted {n} attendance records")
    removed = 0
    for f in files:
        p = config.SNAPSHOT_DIR / Path(f).name             # never follow paths out of snapshots/
        if p.exists():
            p.unlink()
            removed += 1
    return jsonify(ok=True, records=n, photos=removed)


@app.get("/api/zoomlog")
def api_zoomlog():
    """Every auto-zoom move of a session (when, zoom, where, why, who was recognised)."""
    with contextlib.closing(db()) as con:
        con.row_factory = sqlite3.Row
        rows_ = [dict(r) for r in con.execute(
            "SELECT * FROM zoom_events WHERE session=? ORDER BY time", (request.args.get("session"),))]
    return jsonify(rows_)


@app.get("/api/export")
def api_export():
    session = request.args.get("session")
    if not session:
        return jsonify(ok=False, error="Choose a session to export."), 400
    with contextlib.closing(db()) as con:
        recs = rows(con, session)
    marked = {r["name"] for r in recs}
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["session", "name", "status", "time", "similarity", "face_px", "votes",
                "method", "face_proof", "frame_proof"])
    for r in recs:
        w.writerow([session, r["name"], "Present", r["time"], r["similarity"], r["face_px"],
                    r.get("votes"), r.get("method"), r["snapshot"], r.get("frame_snapshot")])
    for n in sorted(set(enrolled_names()) - marked):
        w.writerow([session, n, "Absent", "", "", "", "", "", "", ""])
    fname = re.sub(r"[^A-Za-z0-9_-]", "_", session or "attendance")
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=attendance_{fname}.csv"})


# ====================================================================== students
@app.get("/api/students")
def api_students():
    names = set(enrolled_names())
    out = []
    if config.DATASET_DIR.exists():
        for d in sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()):
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
            out.append({"name": d.name, "photos": n, "enrolled": d.name in names})
    return jsonify(students=out, model=config.MODEL_NAME, enrolling=enroll_state["running"],
                   enroll_output=enroll_state["output"])


@app.post("/api/students")
def api_add_student():
    name = safe_student_name(request.form.get("name"))
    files = request.files.getlist("photos")
    folder = config.DATASET_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext in IMG_EXT:
            f.save(folder / f"upload_{int(time.time() * 1000)}_{saved}{ext}")
            saved += 1
    return jsonify(ok=True, saved=saved)


@app.post("/api/enroll")
def api_enroll():
    if engine.running:
        return jsonify(ok=False, error="Stop the camera first."), 400
    if enroll_state["running"]:
        return jsonify(ok=False, error="Already enrolling."), 400

    def job():
        import enroll
        buf = io.StringIO()
        enroll_state.update(running=True, output="Enrolling... (about 1 minute)")
        try:
            with contextlib.redirect_stdout(buf):
                enroll.main()
        except SystemExit as e:
            buf.write(f"\n{e}")
        except Exception as e:
            buf.write(f"\nError: {e}")
        enroll_state.update(running=False, output=buf.getvalue())

    threading.Thread(target=job, daemon=True).start()
    return jsonify(ok=True)


# ====================================================================== accounts / notifications
def lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))      # no packet is sent; just picks the Wi-Fi interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


@app.get("/api/roster")
def api_roster():
    with contextlib.closing(db()) as con:
        r = students_db.roster(con, set(enrolled_names()))
    return jsonify(students=r, telegram=bot.status, bot=bot.username,
                   portal=f"http://{lan_ip()}:{PORTAL_PORT}")


@app.post("/api/roster")
def api_set_roll():
    d = request.get_json(force=True)
    with contextlib.closing(db()) as con:
        try:
            students_db.set_roll(con, d["name"], d.get("roll"))
        except ValueError as e:
            return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True)


@app.post("/api/roster/pin")
def api_new_pin():
    d = request.get_json(force=True)
    with contextlib.closing(db()) as con:
        pin = students_db.new_pin(con, d["name"])
    return jsonify(ok=True, pin=pin)


@app.post("/api/telegram")
def api_telegram():
    token = (request.get_json(force=True).get("token") or "").strip()
    if not re.fullmatch(r"\d+:[A-Za-z0-9_-]{30,}", token):
        return jsonify(ok=False, error="That doesn't look like a BotFather token."), 400
    ok = bot.set_token(token)
    return jsonify(ok=ok, status=bot.status), (200 if ok else 400)


@app.post("/api/roster/test")
def api_test_message():
    name = request.get_json(force=True).get("name")
    with contextlib.closing(db()) as con:
        chat = students_db.chat_of(con, name)
    if not bot.token:
        return jsonify(ok=False, error="Connect the Telegram bot first."), 400
    if chat is None:
        return jsonify(ok=False, error=f"{name} has not linked Telegram yet (/link roll PIN)."), 400
    try:
        bot.send(chat, f"🔔 Test message for {name} from Face Attendance. Notifications are working.")
    except Exception as e:
        return jsonify(ok=False, error=f"Telegram error: {e}"), 502
    return jsonify(ok=True)


# ---------------------------------------------------------------- phone camera (IP Webcam app)
def _phone_base():
    src = str(engine.source or request.get_json(force=True).get("source") or "")
    # phone stream may be http://ip:8080/video (MJPEG) or rtsp://ip:8080/h264_ulaw.sdp (H.264);
    # the control API is always http://ip:8080
    m = re.match(r"^(?:https?|rtsp)://(?:[^@/]*@)?([^/]+)", src)
    if not m:
        abort(400, "Camera is not a phone (IP Webcam) URL")
    return "http://" + m.group(1)


def _phone_get(base, path):
    import urllib.request
    with urllib.request.urlopen(base + path, timeout=4) as r:
        return r.status, r.read(2000)


def _phone_status(base):
    import urllib.request
    with urllib.request.urlopen(base + "/status.json?show_avail=1", timeout=4) as r:
        return json.loads(r.read())


@app.post("/api/phone/status")
def api_phone_status():
    try:
        d = _phone_status(_phone_base())
    except Exception as e:
        return jsonify(ok=False, error=f"Phone not reachable ({e}). Same Wi-Fi? IP Webcam running?"), 502
    c, a = d.get("curvals", {}), d.get("avail", {})
    zooms = [int(z) for z in a.get("zoom", ["100"])]
    return jsonify(ok=True, zoom=int(c.get("zoom", 100)) / 100, max_zoom=max(zooms) / 100,
                   pan_x=int(c.get("crop_x", 50)), pan_y=int(c.get("crop_y", 50)),
                   size=c.get("video_size"), quality=c.get("quality"),
                   sizes=[s for s in a.get("video_size", []) if s in ("3840x2160", "1920x1080", "1280x720")],
                   battery=d.get("deviceInfo", {}).get("batteryPercent"))


@app.post("/api/phone/zoom")
def api_phone_zoom():
    """zoom: 1.0 - 8.0 (IP Webcam uses 100-800). pan_x / pan_y: 0-100 (50 = centre)."""
    d = request.get_json(force=True)
    if engine.autozoom_active():          # one authority moves the camera at a time
        return jsonify(ok=False, error="Auto-zoom is controlling the camera. "
                                       "Turn auto-zoom off to zoom/pan by hand."), 409
    base, cmds = _phone_base(), []
    if "zoom" in d:
        from autozoom import phone_zoom_steps, snap_zoom      # phone ignores values not in its list
        cmds.append(f"/settings/zoom?set={snap_zoom(float(d['zoom']), phone_zoom_steps(base))}")
    for k, key in (("pan_x", "crop_x"), ("pan_y", "crop_y")):
        if k in d:
            cmds.append(f"/settings/{key}?set={max(0, min(100, int(d[k])))}")
    try:
        for c in cmds:
            _phone_get(base, c)
    except Exception as e:
        return jsonify(ok=False, error=f"Phone did not accept zoom ({e})"), 502
    return jsonify(ok=True)


@app.post("/api/phone/optimize")
def api_phone_optimize():
    """Best settings for recognition: resolution, high JPEG quality, face-priority exposure."""
    size = request.get_json(force=True).get("size", "1920x1080")
    if size not in ("3840x2160", "1920x1080", "1280x720"):
        abort(400)
    base, done, errors = _phone_base(), [], []
    for path in [f"/settings/video_size?set={size}", "/settings/quality?set=85",
                 "/settings/scenemode?set=face_priority", "/settings/focusmode?set=continuous-video"]:
        try:
            s, _ = _phone_get(base, path)
            (done if s == 200 else errors).append(path)
        except Exception as e:
            errors.append(f"{path} ({e})")
    return jsonify(ok=not errors, done=done, errors=errors,
                   note="Stop and Start attendance again so the new resolution is used.")


def start_student_portal():
    from werkzeug.serving import make_server
    from student_portal import portal
    srv = make_server("0.0.0.0", PORTAL_PORT, portal, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Student portal: http://{lan_ip()}:{PORTAL_PORT}  (phones on the same Wi-Fi)")


# ====================================================================== main
if __name__ == "__main__":
    print("Loading face models...")
    get_app()
    start_student_portal()
    url = "http://127.0.0.1:5000"
    print(f"Teacher app: {url}   (only this laptop; Ctrl+C to quit)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, threaded=True)
