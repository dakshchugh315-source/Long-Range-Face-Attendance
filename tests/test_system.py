"""End-to-end system test (no real phone / Telegram needed).

python tests/test_system.py [path/to/clip_of_an_enrolled_student.mov] [expected name]

Starts: a fake phone (IP Webcam-style MJPEG stream + /ptz zoom + /settings),
a fake Telegram API, the teacher app and the student portal, all on
localhost with a throw-away database. Then checks everything a real class
would do. Your real attendance.db / snapshots are not touched.
"""
import http.server
import json
import shutil
import socketserver
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from time import sleep as real_sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="fa_test_"))
config.DB_FILE = TMP / "att.db"
config.SNAPSHOT_DIR = TMP / "snaps"
import notify  # noqa: E402

notify.SECRETS = TMP / "secrets.json"

CLIP = Path(sys.argv[1]) if len(sys.argv) > 1 else None
EXPECT = sys.argv[2] if len(sys.argv) > 2 else None
APP, PORTAL, PHONE = 5065, 5066, 8099

# ------------------------------------------------------------------ fake Telegram
SENT, INBOX, FAIL_NEXT = [], [], [0]


def fake_call(self, method, timeout=35, **p):
    if method == "getMe":
        return {"username": "FakeBot"}
    if method == "sendMessage":
        if FAIL_NEXT[0] > 0:
            FAIL_NEXT[0] -= 1
            raise urllib.error.URLError("simulated network down")
        SENT.append(p)
        return {}
    if method == "getUpdates":
        real_sleep(0.2)
        out = [{"update_id": int(time.time() * 1000) + i,
                "message": {"chat": {"id": c, "type": "private"}, "text": t}}
               for i, (c, t) in enumerate(INBOX)]
        INBOX.clear()
        return out


notify.TelegramBot._call = fake_call
notify.time.sleep = lambda s: real_sleep(min(s, 0.05))       # fast retries in tests

# ------------------------------------------------------------------ fake phone
PHONE_LOG = []


def load_frames():
    cap = cv2.VideoCapture(str(CLIP))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    return frames


class Phone(http.server.BaseHTTPRequestHandler):
    frames = []
    drop_after = 20          # first stream connection is cut after N frames (Wi-Fi drop)

    def log_message(self, *a):
        pass

    def do_GET(self):
        PHONE_LOG.append(self.path)
        if self.path.startswith("/video"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            limit, Phone.drop_after = Phone.drop_after, None
            try:
                i = 0
                while limit is None or i < limit:
                    f = self.frames[i % len(self.frames)]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(f)}\r\n\r\n".encode() + f + b"\r\n")
                    i += 1
                    real_sleep(1 / 15)
            except (BrokenPipeError, ConnectionError, OSError):
                return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


# ------------------------------------------------------------------ helpers
results = []


def check(name, cond, info=""):
    results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info and not cond else ""))


def req(url, data=None, opener=None, method=None):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, method=method or ("POST" if data is not None else "GET"),
                               headers={"Content-Type": "application/json"})
    try:
        with (opener or urllib.request.build_opener()).open(r, timeout=20) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw[:1] in b"{[" else raw)
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw[:1] in b"{[" else raw)


def wait(cond, timeout):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        real_sleep(0.5)
    return False


def main():
    import app as appmod
    from werkzeug.serving import make_server
    from student_portal import portal

    A, P = f"http://127.0.0.1:{APP}", f"http://127.0.0.1:{PORTAL}"
    appmod.get_app()
    names = appmod.enrolled_names()
    student = EXPECT or names[0]
    other = next(n for n in names if n != student)
    for srv in (make_server("127.0.0.1", APP, appmod.app, threaded=True),
                make_server("127.0.0.1", PORTAL, portal, threaded=True)):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    if CLIP:
        Phone.frames = load_frames()
        threading.Thread(target=ThreadedServer(("127.0.0.1", PHONE), Phone).serve_forever,
                         daemon=True).start()

    print("\n1. Telegram bot setup")
    s, r = req(A + "/api/telegram", {"token": "bad"})
    check("bad token rejected", s == 400)
    s, r = req(A + "/api/telegram", {"token": "123456789:AAFakeFakeFakeFakeFakeFakeFakeFake12"})
    check("bot connects", s == 200 and r["ok"], r)

    print("\n2. Roll numbers and PINs")
    req(A + "/api/roster")
    s, _ = req(A + "/api/roster", {"name": student, "roll": "test001"})
    check("roll saved", s == 200)
    s, r = req(A + "/api/roster", {"name": other, "roll": "TEST001"})
    check("duplicate roll rejected", s == 400)
    s, r = req(A + "/api/roster/pin", {"name": student})
    pin = r["pin"]
    check("6-digit PIN generated", len(pin) == 6 and pin.isdigit())
    con = appmod.open_db()
    stored = con.execute("SELECT pin_hash FROM students WHERE name=?", (student,)).fetchone()[0]
    con.close()
    check("PIN stored only as hash", pin not in stored and len(stored) == 64)

    print("\n3. Student links Telegram")
    wrong = "000000" if pin != "000000" else "111111"
    INBOX.append((555, f"/link TEST001 {wrong}"))
    INBOX.append((555, f"/link test001 {pin}"))
    ok = wait(lambda: any("Linked as" in m["text"] for m in SENT), 10)
    check("wrong PIN refused, right PIN links", ok and any("wrong" in m["text"] for m in SENT))
    for _ in range(6):
        INBOX.append((999, "/link TEST001 123123"))
    wait(lambda: any("Too many" in m["text"] for m in SENT if m["chat_id"] == 999), 10)
    check("bot blocks PIN guessing after 5 tries",
          any("Too many" in m["text"] for m in SENT if m["chat_id"] == 999))
    s, _ = req(A + "/api/roster/test", {"name": student})
    check("teacher 'Test msg' delivered", s == 200 and "Test message" in SENT[-1]["text"])

    if CLIP:
        print("\n4. Phone camera over Wi-Fi (fake IP Webcam)")
        src = f"http://127.0.0.1:{PHONE}/video"
        s, r = req(A + "/api/phone/zoom", {"zoom": 2.4, "pan_x": 30, "pan_y": 50, "source": src})
        check("zoom + pan commands reach phone", s == 200 and "/settings/zoom?set=240" in PHONE_LOG
              and "/settings/crop_x?set=30" in PHONE_LOG, PHONE_LOG[-3:])
        s, r = req(A + "/api/phone/optimize", {"size": "1920x1080", "source": src})
        check("optimize command reaches phone",
              any("video_size?set=1920x1080" in p for p in PHONE_LOG)
              and any("quality?set=85" in p for p in PHONE_LOG), PHONE_LOG[-4:])
        s, r = req(A + "/api/start", {"source": src, "session": "System Test"})
        check("attendance starts on phone stream", s == 200, r)
        s, _ = req(A + "/api/start", {"source": src, "session": "System Test"})
        check("second start refused while running", s == 400)
        t0 = time.time()
        marked = wait(lambda: any(p["name"] == student for p in
                                  req(A + "/api/status")[1]["present"]), 90)
        check(f"{student} recognised from live stream", marked, f"{time.time() - t0:.0f}s")
        st = req(A + "/api/status")[1]
        texts = [e["text"] for e in st["events"]]
        check("survives Wi-Fi drop (auto reconnect)",
              st["running"] and "Camera reconnected" in texts, texts[:4])
        check("still running after drop", st["running"], st.get("error"))
        check("FPS reported", st["fps"] > 0, st["fps"])
        wrong_people = [p["name"] for p in st["present"] if p["name"] != student]
        check("nobody else wrongly marked", not wrong_people, wrong_people)
        rec = next((p for p in st["present"] if p["name"] == student), {})
        check("proof: face crop saved", (config.SNAPSHOT_DIR / str(rec.get("snapshot"))).exists())
        check("proof: full frame saved", (config.SNAPSHOT_DIR / str(rec.get("frame_snapshot"))).exists())
        sim, votes = rec.get("similarity") or 0, rec.get("votes") or 0
        need = config.VOTES_STRONG if sim >= config.STRONG_MATCH else config.VOTES_TO_MARK
        check("proof: votes + similarity recorded (3 votes if strong match, else 5)",
              votes >= need and sim >= config.MATCH_THRESHOLD, f"sim {sim} votes {votes} need {need}")
        ok = wait(lambda: any("marked through Face" in m["text"] for m in SENT), 15)
        check("Telegram 'marked through Face' message sent", ok)
        ok = wait(lambda: next((p for p in req(A + "/api/status")[1]["present"]
                                if p["name"] == student), {}).get("notified") == "sent", 10)
        check("delivery status 'sent' shown to teacher", ok)
        req(A + "/api/stop", {})
        check("stop works", not req(A + "/api/status")[1]["running"])
    else:
        print("\n4. (skipped: pass a clip to test the camera path)")
        req(A + "/api/manual", {"session": "System Test", "name": student, "action": "add"})

    print("\n5. Manual marking + failed delivery")
    third = next(n for n in names if n not in (student, other))
    s, _ = req(A + "/api/manual", {"session": "System Test", "name": other, "action": "add"})
    ok = wait(lambda: next((p for p in req(A + "/api/status?session=System%20Test")[1]["present"]
                            if p["name"] == other), {}).get("notified") == "not linked", 10)
    check("unlinked student -> status 'not linked'", ok)
    req(A + "/api/roster", {"name": third, "roll": "TEST003"})
    pin3 = req(A + "/api/roster/pin", {"name": third})[1]["pin"]
    INBOX.append((777, f"/link TEST003 {pin3}"))
    wait(lambda: any(m["chat_id"] == 777 for m in SENT), 10)
    FAIL_NEXT[0] = 2                                  # Wi-Fi drops twice, then recovers
    req(A + "/api/manual", {"session": "System Test", "name": third, "action": "add"})
    ok = wait(lambda: next((p for p in req(A + "/api/status?session=System%20Test")[1]["present"]
                            if p["name"] == third), {}).get("notified") == "sent", 15)
    check("message retried after network drop and delivered", ok)
    FAIL_NEXT[0] = 99                                 # Telegram completely down
    fourth = next(n for n in names if n not in (student, other, third))
    req(A + "/api/roster", {"name": fourth, "roll": "TEST004"})
    pin4 = req(A + "/api/roster/pin", {"name": fourth})[1]["pin"]
    FAIL_NEXT[0] = 0
    INBOX.append((888, f"/link TEST004 {pin4}"))
    wait(lambda: any(m["chat_id"] == 888 for m in SENT), 10)
    FAIL_NEXT[0] = 99
    req(A + "/api/manual", {"session": "System Test", "name": fourth, "action": "add"})
    ok = wait(lambda: next((p for p in req(A + "/api/status?session=System%20Test")[1]["present"]
                            if p["name"] == fourth), {}).get("notified") == "failed", 20)
    check("Telegram down -> teacher sees 'failed'", ok)
    FAIL_NEXT[0] = 0
    s, csv_text = req(A + "/api/export?session=System%20Test")
    csv_text = csv_text.decode() if isinstance(csv_text, bytes) else str(csv_text)
    check("CSV has present + absent rows", "Present" in csv_text and "Absent" in csv_text)

    print("\n6. Student portal")
    jar = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    s, _ = req(P + "/api/me", opener=jar)
    check("not logged in -> 401", s == 401)
    s, _ = req(P + "/login", {"roll": "TEST001", "pin": wrong}, opener=jar)
    check("wrong PIN refused", s == 401)
    s, _ = req(P + "/login", {"roll": "test001", "pin": pin}, opener=jar)
    check("right PIN logs in", s == 200)
    s, me = req(P + "/api/me", opener=jar)
    check("sees own attendance", s == 200 and me["name"] == student and me["records"])
    check("record says Face", (not CLIP) or me["records"][0]["method"] == "auto")
    if CLIP:
        s, _ = req(P + "/myface/" + me["records"][0]["snapshot"], opener=jar)
        check("own proof photo visible", s == 200)
    others = [p.name for p in config.SNAPSHOT_DIR.glob("*.jpg") if student.replace(" ", "_") not in p.name]
    if others:
        s, _ = req(P + "/myface/" + others[0], opener=jar)
        check("other student's photo blocked", s == 404)
    s, _ = req(P + "/myface/..%2Fatt.db", opener=jar)
    check("path trick blocked", s in (404, 400))
    for path in ["/api/status", "/api/roster", "/api/export?session=x"]:
        s, _ = req(P + path, opener=jar)
        check(f"teacher API {path} not on portal", s == 404)
    jar2 = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    codes = [req(P + "/login", {"roll": "TEST001", "pin": "999999"}, opener=jar2)[0] for _ in range(10)]
    check("portal blocks PIN guessing", 429 in codes, codes)

    print("\n7. Delete recorded attendance")
    req(A + "/api/manual", {"session": "Keep Me", "name": student, "action": "add"})
    photos_before = len(list(config.SNAPSHOT_DIR.glob("*.jpg"))) if config.SNAPSHOT_DIR.exists() else 0
    s, r = req(A + "/api/delete", {"session": "System Test"})
    check("delete one session", s == 200 and r["records"] >= 3, r)
    sess = [x["session"] for x in req(A + "/api/sessions")[1]]
    check("that session is gone, other session kept", "System Test" not in sess and "Keep Me" in sess, sess)
    if CLIP:
        check("its proof photos deleted", r["photos"] >= 2 and
              len(list(config.SNAPSHOT_DIR.glob("*.jpg"))) < photos_before, r)
    roster = req(A + "/api/roster")[1]["students"]
    check("students / roll numbers / PINs untouched",
          any(x["name"] == student and x["roll"] == "TEST001" and x["has_pin"] for x in roster))
    s, r = req(A + "/api/delete", {"all": True})
    check("delete ALL", s == 200 and req(A + "/api/sessions")[1] == [], r)
    s, _ = req(A + "/api/delete", {})
    check("empty delete request refused", s == 400)
    s, _ = req(P + "/api/delete", {"all": True}, opener=jar)
    check("students cannot delete (not on portal)", s in (404, 405))

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
