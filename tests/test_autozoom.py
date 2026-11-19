"""Auto-zoom end-to-end test with a simulated phone that really zooms.

python tests/test_autozoom.py [background.jpg]

A 4K "classroom" is built with enrolled students sitting so far away that
their faces are ~20 px at 1x (too small to recognise - like a big hall).
A fake IP Webcam serves the view: zoom 1-8x (only its 100,107,114.. steps,
like the real app) and pan with crop_x / crop_y. Then:
  1. auto-zoom OFF -> expected: nobody can be marked
  2. auto-zoom ON  -> the software must zoom by itself and mark them,
                      and log every move in zoom_events
Real attendance.db is not touched.
"""
import http.server
import json
import socketserver
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from time import sleep as real_sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="fa_az_"))
config.DB_FILE = TMP / "att.db"
config.SNAPSHOT_DIR = TMP / "snaps"
import notify  # noqa: E402

notify.SECRETS = TMP / "secrets.json"
notify.TelegramBot._call = lambda self, method, timeout=35, **p: (real_sleep(1), [])[1]

import face_engine  # noqa: E402

W4, H4 = 3840, 2160                 # the "sensor"
OUT_W, OUT_H = 1920, 1080           # what the phone streams
FACE_4K = 42                        # face width on the sensor -> ~21 px at 1x
STEPS = list(range(100, 801, 7))
APP, PHONE = 5075, 8080


# ------------------------------------------------------------------ build the classroom
def build_room(bg_path, people):
    bg = cv2.imread(str(bg_path)) if bg_path else None
    room = cv2.resize(bg, (W4, H4)) if bg is not None else np.full((H4, W4, 3), 190, np.uint8)
    seats = [(0.18, 0.42), (0.40, 0.38), (0.62, 0.44), (0.84, 0.40), (0.30, 0.62), (0.72, 0.64)]
    placed = []
    for (name, img, face), (sx, sy) in zip(people, seats):
        s = FACE_4K / face_engine.face_width(face)
        x1, y1, x2, y2 = face.bbox
        pad = (x2 - x1) * 1.2
        crop = img[int(max(0, y1 - pad)):int(min(img.shape[0], y2 + pad * 2.5)),
                   int(max(0, x1 - pad)):int(min(img.shape[1], x2 + pad))]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = crop.shape[:2]
        cx, cy = int(sx * W4), int(sy * H4)
        y0, x0 = max(0, cy - h // 3), max(0, cx - w // 2)
        room[y0:y0 + h, x0:x0 + w] = crop[:H4 - y0, :W4 - x0]
        placed.append(name)
    return room, placed


def pick_people(n):
    out = []
    for d in sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.jpg")):
            img = cv2.imread(str(f))
            fs = face_engine.detect(img, tiles=False)
            if fs:
                face = max(fs, key=face_engine.face_width)
                if not face_engine.is_turned_away(face):          # a frontal photo
                    out.append((d.name, img, face))
                    break
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------------ fake phone (digital zoom)
class Phone(http.server.BaseHTTPRequestHandler):
    room = None
    zoom, crop_x, crop_y = 100, 50, 50
    log = []

    def log_message(self, *a):
        pass

    @classmethod
    def view(cls):
        z = cls.zoom / 100
        ww, wh = W4 / z, H4 / z
        cx = ww / 2 + (cls.crop_x / 100) * (W4 - ww)          # same "free range" mapping as the real app
        cy = wh / 2 + (cls.crop_y / 100) * (H4 - wh)
        x0, y0 = int(cx - ww / 2), int(cy - wh / 2)
        win = cls.room[y0:y0 + int(wh), x0:x0 + int(ww)]
        return cv2.resize(win, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA if z < 2 else cv2.INTER_LINEAR)

    def _ok(self, body=b"<result>Ok</result>", ctype="text/xml"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = cv2.imencode(".jpg", Phone.view(), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                    real_sleep(1 / 8)
            except OSError:
                return
        if u.path == "/shot.jpg":
            return self._ok(cv2.imencode(".jpg", Phone.view())[1].tobytes(), "image/jpeg")
        if u.path == "/status.json":
            return self._ok(json.dumps({"curvals": {"zoom": str(Phone.zoom), "crop_x": str(Phone.crop_x),
                                                    "crop_y": str(Phone.crop_y), "video_size": "1920x1080",
                                                    "quality": "85"},
                                        "avail": {"zoom": [str(s) for s in STEPS]}}).encode(), "application/json")
        if u.path.startswith("/settings/"):
            key, val = u.path.rsplit("/", 1)[1], q.get("set", "")
            Phone.log.append((key, val))
            if key == "zoom" and val.isdigit() and int(val) in STEPS:   # real phone ignores others
                Phone.zoom = int(val)
            elif key in ("crop_x", "crop_y") and val.isdigit():
                setattr(Phone, key, max(0, min(100, int(val))))
        return self._ok()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def req(url, data=None):
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                               headers={"Content-Type": "application/json"},
                               method="POST" if data is not None else "GET")
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read())


def run_session(A, name, autozoom, seconds):
    req(A + "/api/start", {"source": f"http://127.0.0.1:{PHONE}/video", "session": name,
                           "autozoom": autozoom})
    t0, marked = time.time(), {}
    while time.time() - t0 < seconds:
        real_sleep(2)
        st = req(A + "/api/status")
        for p in st["present"]:
            marked.setdefault(p["name"], (round(time.time() - t0), p.get("zoom")))
        if not st["running"]:
            break
    fps = st["fps"]
    req(A + "/api/stop", {})
    return marked, fps


def main():
    import app as appmod
    from werkzeug.serving import make_server

    appmod.get_app()
    people = pick_people(6)
    Phone.room, placed = build_room(sys.argv[1] if len(sys.argv) > 1 else None, people)
    cv2.imwrite(str(TMP / "room_4k.jpg"), Phone.room)
    wide = Phone.view()
    sizes = [face_engine.face_width(f) for f in face_engine.detect(wide)]
    print(f"classroom: {placed}")
    print(f"faces at 1x (px): {sorted(sizes)}  (recognition needs >= {config.MIN_FACE_PX})")

    threading.Thread(target=Server(("127.0.0.1", PHONE), Phone).serve_forever, daemon=True).start()
    srv = make_server("127.0.0.1", APP, appmod.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    A = f"http://127.0.0.1:{APP}"

    print("\n1. auto-zoom OFF (45 s)")
    off, fps = run_session(A, "AZ off", False, 45)
    print(f"   marked: {off or 'nobody'} | fps {fps}")

    print("\n2. auto-zoom ON (up to 240 s)")
    Phone.log.clear()
    on, fps = run_session(A, "AZ on", True, 240)
    print(f"   marked: {on} | fps {fps}")
    log = req(A + "/api/zoomlog?session=AZ%20on")
    print(f"   zoom moves logged: {len(log)}")
    for z in log[:12]:
        print(f"     {z['time'][11:]} {z['action']:6} {z['zoom']}x at ({z['x']},{z['y']}) "
              f"{z['reason']} -> {z['recognised'] or '-'}")
    bad = [k for k, v in Phone.log if k == "zoom" and not (v.isdigit() and int(v) in STEPS)]
    wrong = [n for n in on if n not in placed]

    print("\nRESULT")
    checks = [("without auto-zoom nobody (or almost nobody) is marked", len(off) <= 1),
              ("auto-zoom marks at least 4 of the 6 far students", len([n for n in on if n in placed]) >= 4),
              ("nobody wrongly marked", not wrong),
              ("marks were made zoomed in (zoom > 1 saved)", any((v[1] or 1) > 1.01 for v in on.values())),
              ("zoom moves are logged in the database", len(log) >= 1),
              ("only zoom values the phone accepts were sent", not bad),
              ("phone is back at 1x after stop", Phone.zoom == 100)]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed   (room image: {TMP / 'room_4k.jpg'})")


if __name__ == "__main__":
    main()
