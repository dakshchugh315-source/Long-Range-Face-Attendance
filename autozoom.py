"""Auto-zoom ("AI PTZ") for a phone running IP Webcam.

The software drives the phone's zoom + pan by itself:
  WIDE   - look at the whole room at 1x, list faces that are small and not yet marked
  TARGET - zoom onto one such face, dwell a few seconds so it gets recognised, return
  SWEEP  - if students are still absent, zoom and scan the room in a grid
           (finds faces too small to even detect at 1x)
Every move is written to the zoom_events table.

Pan mapping: IP Webcam takes crop_x / crop_y (0-100). calibrate() measures how
that maps to the visible window by template-matching zoomed shots against a
wide shot, so targeting is accurate on the real phone.
"""
import json
import threading
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

import config


# ------------------------------------------------------------------ phone I/O
def phone_get(base, path, timeout=4):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read()


def phone_shot(base):
    buf = phone_get(base, "/shot.jpg", timeout=6)
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def phone_zoom_steps(base):
    """Zoom values the phone accepts (e.g. 100, 107, 114 ... 800). It silently
    ignores anything not in this list, e.g. 200."""
    try:
        d = json.loads(phone_get(base, "/status.json?show_avail=1").decode("utf-8-sig"))
        return sorted(int(z) for z in d["avail"]["zoom"])
    except Exception:
        return list(range(100, 801, 7))


def is_ip_webcam(base):
    """True only if `base` really is the IP Webcam app (its status lists zoom steps)."""
    try:
        d = json.loads(phone_get(base, "/status.json?show_avail=1", timeout=3).decode("utf-8-sig"))
        return bool(d.get("avail", {}).get("zoom"))
    except Exception:
        return False


def snap_zoom(z, steps):
    """1.0-8.0 -> nearest value the phone accepts."""
    return min(steps, key=lambda s: abs(s - z * 100))


def phone_max_zoom(base):
    return max(phone_zoom_steps(base)) / 100


# ------------------------------------------------------------------ pan mapping
def window_centre(crop, z, mode):
    """Centre (0-1) of the visible window for crop value 0-100 at zoom z."""
    if mode == "free_range":          # crop moves the window inside the frame
        return 1 / (2 * z) + (crop / 100) * (1 - 1 / z)
    c = crop / 100                    # "centre": crop is the window centre
    return min(max(c, 1 / (2 * z)), 1 - 1 / (2 * z))


def crop_for(centre, z, mode):
    """crop value 0-100 that puts the window centre at `centre` (0-1)."""
    if z <= 1.001:
        return 50
    if mode == "free_range":
        v = (centre - 1 / (2 * z)) / (1 - 1 / z)
    else:
        v = centre
    return int(round(min(max(v, 0.0), 1.0) * 100))


def _locate(wide, zoomed, z):
    """Where (normalised centre) does a zoomed shot sit inside the wide shot?"""
    gw = cv2.cvtColor(wide, cv2.COLOR_BGR2GRAY)
    gz = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
    h, w = gw.shape
    tmpl = cv2.resize(gz, (max(8, int(w / z)), max(8, int(h / z))), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(gw, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    return (x + tmpl.shape[1] / 2) / w, (y + tmpl.shape[0] / 2) / h, score


def calibrate(base, settle=3.0):
    """Measure the crop -> window mapping on the real phone.
    Returns dict(mode=..., error=..., score=...) or raises."""
    phone_get(base, "/settings/zoom?set=100")
    phone_get(base, "/settings/crop_x?set=50")
    phone_get(base, "/settings/crop_y?set=50")
    time.sleep(settle)
    wide = phone_shot(base)
    z = snap_zoom(2.0, phone_zoom_steps(base)) / 100      # e.g. 1.98 - phone ignores 2.00
    obs = []
    for cx, cy in ((0, 0), (100, 100), (50, 50)):
        phone_get(base, f"/settings/zoom?set={int(z * 100)}")
        phone_get(base, f"/settings/crop_x?set={cx}")
        phone_get(base, f"/settings/crop_y?set={cy}")
        time.sleep(settle)
        mx, my, score = _locate(wide, phone_shot(base), z)
        obs.append((cx, cy, mx, my, score))
    phone_get(base, "/settings/zoom?set=100")
    phone_get(base, "/settings/crop_x?set=50")
    phone_get(base, "/settings/crop_y?set=50")
    best = None
    for mode in ("free_range", "centre"):
        err = np.mean([abs(window_centre(cx, z, mode) - mx) + abs(window_centre(cy, z, mode) - my)
                       for cx, cy, mx, my, _ in obs])
        if best is None or err < best["error"]:
            best = {"mode": mode, "error": round(float(err), 3)}
    best["score"] = round(float(min(o[4] for o in obs)), 2)
    best["ok"] = best["error"] < 0.08 and best["score"] > 0.4
    best["observations"] = [[round(float(v), 3) for v in o] for o in obs]
    return best


# ------------------------------------------------------------------ camera drivers
class PhoneDriver:
    """IP Webcam app (Android). Digital zoom 1-8x + pan inside the sensor image."""
    name = "phone"

    def __init__(self, base, mode="free_range"):
        self.base, self.mode = base, mode
        self.steps = phone_zoom_steps(base)
        self.max_zoom = max(self.steps) / 100

    def set_view(self, z, cx, cy):
        z = snap_zoom(z, self.steps) / 100
        for c in (f"/settings/zoom?set={int(z * 100)}",
                  f"/settings/crop_x?set={crop_for(cx, z, self.mode)}",
                  f"/settings/crop_y?set={crop_for(cy, z, self.mode)}"):
            phone_get(self.base, c)

    def reset(self):
        self.set_view(1.0, 0.5, 0.5)


class OnvifDriver:
    """PTZ / motorised-zoom CCTV via ONVIF (Hikvision, CP Plus, Dahua, ...).

    Zoom uses AbsoluteMove in the camera's normalised zoom space. Pan/tilt to a
    target uses RelativeMove scaled by the current field of view. EXPERIMENTAL:
    verify with camera_check.py --ptz-test on the real camera first."""
    name = "onvif"

    def __init__(self, host, port, user, password, max_zoom=None):
        from onvif import ONVIFCamera                     # pip install onvif-zeep
        cam = ONVIFCamera(host, port, user, password)
        self.media, self.ptz = cam.create_media_service(), cam.create_ptz_service()
        self.token = self.media.GetProfiles()[0].token
        self.max_zoom = max_zoom or config.ONVIF_OPTICAL_ZOOM
        self.zoom = 1.0

    def _zoom_space(self, z):            # 1x .. max_zoom  ->  0 .. 1
        return max(0.0, min(1.0, (z - 1) / max(self.max_zoom - 1, 1e-6)))

    def set_view(self, z, cx, cy):
        # pan/tilt by the target's offset from centre, in units of the current view
        fov_h = config.ONVIF_HFOV_DEG / self.zoom
        pan = (cx - 0.5) * fov_h / 180.0                  # ONVIF generic space: -1..1 = -180..180 deg
        tilt = -(cy - 0.5) * fov_h * 9 / 16 / 90.0
        if abs(pan) > 0.002 or abs(tilt) > 0.002:
            self.ptz.RelativeMove({"ProfileToken": self.token,
                                   "Translation": {"PanTilt": {"x": pan, "y": tilt}}})
        self.ptz.AbsoluteMove({"ProfileToken": self.token,
                               "Position": {"Zoom": {"x": self._zoom_space(z)}}})
        self.zoom = z

    def reset(self):
        try:
            self.ptz.GotoHomePosition({"ProfileToken": self.token})
        except Exception:
            self.ptz.AbsoluteMove({"ProfileToken": self.token, "Position": {"Zoom": {"x": 0.0}}})
        self.zoom = 1.0


# ------------------------------------------------------------------ scan state
SEARCHING_FOR_FACE = "SEARCHING_FOR_FACE"
FACE_DETECTED = "FACE_DETECTED"          # recognisable face in view, not identified yet
FACE_SCANNING = "FACE_SCANNING"          # face identified, collecting votes (1/5 .. 4/5)
SCAN_COMPLETED = "SCAN_COMPLETED"        # the face just reached 5/5 and was marked
FACE_LOST = "FACE_LOST"                  # face vanished; waiting to confirm before moving
HOLD_STATES = (FACE_DETECTED, FACE_SCANNING, FACE_LOST)


class ScanMonitor:
    """Knows whether a face is being detected/scanned in the current view.

    While it is in a HOLD state the camera must not sweep, advance a sweep or
    zoom out. A face only holds while it makes progress: one that gains no vote
    for AUTOZOOM_SCAN_STALL_S (a stranger, someone permanently turned away) or
    has held for AUTOZOOM_MAX_HOLD_S stops holding, so the camera can't get stuck.
    """

    def __init__(self):
        self.state = SEARCHING_FOR_FACE
        self.entries = []           # one per face: x, y, view, first, last_seen, last_progress, votes, label
        self.holding = []           # [(x, y, label, votes)] faces that hold the camera this frame
        self.last_holding = []      # where they were when last seen (used while FACE_LOST)
        self.last_hold_seen = float("-inf")
        self.completed_at = float("-inf")
        self.completed_name = None
        self._scanning_names = set()

    def _entry(self, x, y, view):
        for e in self.entries:
            if e["view"] == view and abs(e["x"] - x) < 0.06 and abs(e["y"] - y) < 0.08:
                return e
        return None

    def update(self, faces, w, h, present, view, now):
        present = set(present)
        holding = []
        # a name that was being scanned and is now marked -> SCAN_COMPLETED
        done = self._scanning_names & present
        if done:
            self.completed_at, self.completed_name = now, sorted(done)[0]
        scanning_names = set()
        for f in faces:
            if f["px"] < config.MIN_FACE_PX or f["label"] in present:
                continue                      # too small to scan (an auto-zoom target) / already marked
            x1, y1, x2, y2 = f["box"]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            label = f["label"]
            votes = int(f.get("votes") or 0) if label != "Unknown" else 0
            e = self._entry(cx, cy, view)
            if e is None:
                e = {"x": cx, "y": cy, "view": view, "first": now, "last_seen": now,
                     "last_progress": now, "votes": votes, "label": label}
                self.entries.append(e)
            elif votes > e["votes"] or (label != "Unknown" and e["label"] == "Unknown"):
                e["last_progress"] = now      # scan moved forward (e.g. 1/5 -> 2/5)
            e.update(x=cx, y=cy, last_seen=now, votes=votes, label=label)
            stalled = (now - e["last_progress"] > config.AUTOZOOM_SCAN_STALL_S
                       or now - e["first"] > config.AUTOZOOM_MAX_HOLD_S)
            if not stalled:
                holding.append((cx, cy, label, votes, int(f.get("need") or config.VOTES_TO_MARK)))
                if label != "Unknown":
                    scanning_names.add(label)
        self.entries = [e for e in self.entries if now - e["last_seen"] < 60]
        if holding:
            self._scanning_names = scanning_names
        elif now - self.last_hold_seen >= config.AUTOZOOM_FACE_LOSS_S:
            self._scanning_names = set()      # genuinely lost: forget who was being scanned
        self.holding = holding
        if holding:
            self.last_holding = holding
            self.last_hold_seen = now
            self.state = FACE_SCANNING if any(hf[2] != "Unknown" for hf in holding) else FACE_DETECTED
        elif now - self.last_hold_seen < config.AUTOZOOM_FACE_LOSS_S:
            self.state = FACE_LOST            # brief dropout (blink, head turn, missed frame): wait
        elif now - self.completed_at < 2.0:
            self.state = SCAN_COMPLETED
        else:
            self.state = SEARCHING_FOR_FACE
        if done and not holding:              # the scanned face just finished: no need to wait
            self.state = SCAN_COMPLETED
            self.last_hold_seen = float("-inf")
        return self.state

    def hold(self):
        return self.state in HOLD_STATES

    def held_positions(self):
        """Faces the camera must keep in view: the ones seen now, or - while a face is
        briefly lost - where it was last seen (it will most likely reappear there)."""
        if self.holding:
            return self.holding
        return self.last_holding if self.state == FACE_LOST else []

    def describe(self):
        if self.state == FACE_SCANNING:
            known = [hf for hf in self.holding if hf[2] != "Unknown"]
            top = max(known, key=lambda hf: hf[3]) if known else (0, 0, "", 0, config.VOTES_TO_MARK)
            return f"{self.state} {top[2]} {top[3]}/{top[4]}"
        if self.state == SCAN_COMPLETED:
            return f"{self.state} {self.completed_name or ''}"
        return self.state


# ------------------------------------------------------------------ controller
class AutoZoom:
    """The ONE authority that moves the camera while auto-zoom is on.

    Movement rule: while ScanMonitor holds (a face is detected / being scanned /
    just lost) the camera may not sweep, advance a sweep or zoom out. The only
    move allowed during a hold is a zoom-in whose window still contains every
    face under scan (it keeps them in view, just bigger).
    """

    def __init__(self, driver, log_event, async_cmds=True, clock=time.time):
        self.driver = driver
        self.log_event = log_event            # callable(dict) -> writes zoom_events row
        self.clock = clock
        self.async_cmds = async_cmds
        self.max_zoom = min(driver.max_zoom, config.AUTOZOOM_MAX)
        self.state = "WIDE"                   # camera motion: WIDE / TARGET / SWEEP
        self.scan = ScanMonitor()             # face state: SEARCHING / DETECTED / SCANNING / ...
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5
        self.since = self.clock()
        self.view_id = 0                      # changes whenever the view moves
        self.spots = []                       # places zoomed on: dict(x, y, last, visits, done_until)
        self.last_sweep = self.clock()
        self.sweep_queue = []
        self.fruitless_sweeps = 0             # sweeps in a row that found nobody new
        self.sweep_marked_at_start = 0
        self.current = None                   # dict describing the running action
        self.busy = False
        self.error = None
        self.cmd_lock = threading.Lock()      # one command sequence at a time
        self.gen = 0                          # newest command wins; stale ones are dropped
        self.stopped = False
        self._set(1.0, 0.5, 0.5)

    # -- camera commands: serialised, newest wins, nothing after stop()
    def _set(self, z, cx, cy):
        if self.stopped:
            return
        self.zoom, self.cx, self.cy = z, cx, cy
        self.since = self.clock()
        self.view_id += 1
        self.busy = True
        self.gen += 1
        my_gen = self.gen

        def send():
            with self.cmd_lock:
                if my_gen != self.gen or self.stopped:
                    return                        # superseded by a newer command / stop()
                try:
                    self.driver.set_view(z, cx, cy)
                    self.error = None
                except Exception as e:
                    self.error = f"{self.driver.name} zoom failed: {e}"
                finally:
                    if my_gen == self.gen:
                        self.since = self.clock()  # settle time counts from when the move finished
                        self.busy = False
        if self.async_cmds:
            threading.Thread(target=send, daemon=True).start()
        else:
            send()

    def settling(self):
        return self.busy or self.clock() - self.since < config.AUTOZOOM_SETTLE_S

    def camera_state(self):
        """One explicit state for the UI / logs, derived from the controller:
        REPOSITIONING  camera command in flight          STABILIZING  waiting for focus/H.264 after a move
        RECOGNIZING    a face in view is being scanned   TARGET_LOST  face vanished, waiting to confirm
        ATTENDANCE_RECORDED  a scan just completed        TARGET_LOCKED  zoomed on a target
        SEARCHING      sweeping the room                 COOLDOWN     wide view, waiting before next move
        IDLE           all visible faces done; no sweeping until a new face appears"""
        if self.busy:
            return "REPOSITIONING"
        if self.clock() - self.since < config.AUTOZOOM_SETTLE_S:
            return "STABILIZING"
        s = self.scan.state
        if s in (FACE_SCANNING, FACE_DETECTED):
            return "RECOGNIZING"
        if s == FACE_LOST:
            return "TARGET_LOST"
        if s == SCAN_COMPLETED:
            return "ATTENDANCE_RECORDED"
        if self.state == "TARGET":
            return "TARGET_LOCKED"
        if self.state == "SWEEP":
            return "SEARCHING"
        if self.fruitless_sweeps >= config.AUTOZOOM_MAX_FRUITLESS:
            return "IDLE"
        return "COOLDOWN"

    def label(self):
        scan = self.scan.describe()
        hold = " (holding camera)" if self.scan.hold() else ""
        cs = self.camera_state()
        if self.state == "WIDE":
            return f"AUTO-ZOOM [{cs}] wide 1.0x | {scan}{hold}"
        what = self.current.get("reason", "") if self.current else ""
        return f"AUTO-ZOOM [{cs}] {self.zoom:.1f}x {what} | {scan}{hold}"

    def _window_keeps(self, z, cx, cy, holding):
        """Would a zoom to (z, cx, cy) keep every held face inside the picture?"""
        half = 1 / (2 * z)
        wx = min(max(cx, half), 1 - half)
        wy = min(max(cy, half), 1 - half)
        return all(abs(hx - wx) <= half * 0.85 and abs(hy - wy) <= half * 0.85
                   for hx, hy, *_ in holding)

    # -- called once per processed frame (only from the camera loop thread)
    def step(self, faces, frame_shape, present, absent_count, recognised_now):
        """faces: list of dict(box, px, label, votes). Returns True if the view moved."""
        if self.stopped:
            return False
        now = self.clock()
        h, w = frame_shape[:2]
        if self.current is not None:
            self.current["seen"].update(recognised_now)
        self.scan.update(faces, w, h, present, self.view_id, now)
        hold = self.scan.hold()
        if hold:
            self.last_sweep = now             # sweep timer only runs while genuinely searching
            if self.current is not None:
                self.current["held"] = True
        if self.settling():
            return False

        if self.state in ("TARGET", "SWEEP"):
            c = self.current or {}
            if self.state == "TARGET":
                # which face in the zoomed view is the one we came for? (where the target
                # must appear given the zoom window) - neighbours don't count
                z = self.zoom
                wx = min(max(c["x"], 1 / (2 * z)), 1 - 1 / (2 * z))
                wy = min(max(c["y"], 1 / (2 * z)), 1 - 1 / (2 * z))
                ex, ey = (0.5 + (c["x"] - wx) * z) * w, (0.5 + (c["y"] - wy) * z) * h
                best = None
                for f in faces:
                    x1, y1, x2, y2 = f["box"]
                    d = np.hypot((x1 + x2) / 2 - ex, (y1 + y2) / 2 - ey)
                    if d < 0.2 * w and (best is None or d < best[0]):
                        best = (d, f["label"])
                if best and best[1] != "Unknown":
                    c["target_names"].add(best[1])
            tn = c.get("target_names", set())
            # leave early once the person we came for is marked - no need to stare longer
            done_early = self.state == "TARGET" and tn and tn <= set(present)
            if now - self.since < config.AUTOZOOM_DWELL_S and not done_early:
                return False
            if hold:
                return False                  # a face here is being scanned: stay until done / lost
            self._finish(now, present)
            if self.state == "SWEEP" and self.sweep_queue:
                return self._go_sweep_next()
            if self.state == "SWEEP":         # whole sweep done: timer starts NOW, not at its start
                self.last_sweep = now
                found = len(present) > self.sweep_marked_at_start
                self.fruitless_sweeps = 0 if found else self.fruitless_sweeps + 1
            self.state = "WIDE"
            self._set(1.0, 0.5, 0.5)
            return True

        # WIDE
        if now - self.since < config.AUTOZOOM_WIDE_S or absent_count == 0:
            return False
        cands = []
        for f in faces:
            if f["label"] in present or f["px"] >= config.AUTOZOOM_TARGET_PX:
                continue
            if f["px"] * self.max_zoom < config.MIN_FACE_PX:
                continue                                   # too small even at max zoom: zooming can't help
            x1, y1, x2, y2 = f["box"]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            spot = self._spot(cx, cy)
            if spot and (spot["done_until"] > now or now - spot["last"] < config.AUTOZOOM_REVISIT_S):
                continue                                   # identified there already / just looked
            # never-visited places first, then the one looked at longest ago, then smallest face
            cands.append((spot is not None, spot["last"] if spot else 0, f["px"], cx, cy, f["label"]))
        for _, _, px, cx, cy, label in sorted(cands):
            z = min(self.max_zoom, max(1.5, config.AUTOZOOM_GOAL_PX / max(px, 1)))
            if hold and not self._window_keeps(z, cx, cy, self.scan.held_positions()):
                continue                      # this zoom would push a face under scan out of view
            spot = self._spot(cx, cy)
            if spot is None:
                spot = {"x": cx, "y": cy, "last": now, "visits": 0, "done_until": 0}
                self.spots.append(spot)
            spot["last"] = now
            spot["visits"] += 1
            self.state = "TARGET"
            self.current = {"action": "target", "zoom": round(z, 2), "x": round(cx, 3),
                            "y": round(cy, 3), "face_px": int(px), "spot": spot,
                            "reason": f"small face {int(px)}px ({label})", "seen": set(),
                            "target_names": set(), "start": now}
            self._set(z, cx, cy)
            return True
        if hold:
            return False                      # ACTIVE FACE SCANNING = SWEEP DISABLED
        if self.fruitless_sweeps >= config.AUTOZOOM_MAX_FRUITLESS:
            # IDLE: repeated sweeps found nobody new. No more sweeping (a stranger in view must not
            # keep the camera sweeping). New faces in view are still zoomed on by the targeting
            # above; a sweep that finds someone, or switching auto-zoom off/on, leaves IDLE.
            return False
        if self.scan.state == SEARCHING_FOR_FACE and now - self.last_sweep > self.sweep_interval():
            self.last_sweep = now
            self.sweep_marked_at_start = len(present)
            z = min(self.max_zoom, config.AUTOZOOM_SWEEP_ZOOM)
            # windows are 1/z of the frame wide: ceil(z) columns overlap and cover
            # every seat; one row fewer because faces sit in the middle band
            cols = max(2, int(np.ceil(z)))
            rows = max(1, int(np.ceil(z)) - 1)
            self.sweep_queue = [((i + 0.5) / cols, (j + 0.5) / rows)
                                for j in range(rows) for i in range(cols)]
            self.state = "SWEEP"
            return self._go_sweep_next()
        return False

    def sweep_interval(self):
        """30 s, doubling after every sweep that found nobody new (absent students are
        usually just not in the room), capped at AUTOZOOM_SWEEP_MAX_S."""
        return min(config.AUTOZOOM_SWEEP_MAX_S,
                   config.AUTOZOOM_SWEEP_EVERY_S * 2 ** self.fruitless_sweeps)

    def _go_sweep_next(self):
        cx, cy = self.sweep_queue.pop(0)
        z = min(self.max_zoom, config.AUTOZOOM_SWEEP_ZOOM)
        self.current = {"action": "sweep", "zoom": round(z, 2), "x": round(cx, 3), "y": round(cy, 3),
                        "face_px": None, "reason": f"sweep ({len(self.sweep_queue)} left)",
                        "seen": set(), "start": self.clock()}
        self._set(z, cx, cy)
        return True

    def _spot(self, cx, cy):
        for sp in self.spots:
            if abs(cx - sp["x"]) < 0.05 and abs(cy - sp["y"]) < 0.05:
                return sp
        return None

    def _finish(self, now, present=()):
        c = self.current
        if not c:
            return
        sp = c.get("spot")
        if sp is not None:
            tn = c.get("target_names", set())
            if tn and tn <= set(present):
                sp["done_until"] = float("inf")          # person there identified + marked: never again
            elif not tn and sp["visits"] >= 3:
                # 3 tries, nobody identified (turned away / not a student): wait 2, 4, 8 ... min
                sp["done_until"] = now + 120 * 2 ** (sp["visits"] - 3)
        reason = c["reason"] + (" | held for face scan" if c.get("held") else "")
        self.log_event({"time": datetime.fromtimestamp(c["start"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "action": c["action"], "zoom": c["zoom"], "x": c["x"], "y": c["y"],
                        "face_px": c["face_px"], "reason": reason,
                        "seconds": round(now - c["start"], 1),
                        "recognised": ", ".join(sorted(c["seen"]))})
        self.current = None

    def stop(self):
        """Back to 1x. Cancels any pending move; no command can follow it."""
        try:
            if self.current:
                self._finish(self.clock())
        except Exception:
            pass
        self.stopped = True
        self.gen += 1                                   # pending moves become stale
        with self.cmd_lock:
            try:
                self.driver.reset()
            except Exception:
                pass
