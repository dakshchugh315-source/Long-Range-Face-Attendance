"""Full-classroom coverage: which parts of the room were inspected, who is verified,
which faces stay unresolved, and an honest end-of-session verdict.

The room is a 3 x 2 grid of regions in WIDE-view coordinates (the auto-zoom sweep at
2.5x visits exactly these 6 cells). Every face seen - at 1x or zoomed in - is mapped
back to its region.

  region status   not_inspected  -> watched with a still camera for < INSPECT_S seconds
                  needs_review   -> an open unresolved face, or faces too small to recognise
                                    that no zoomed look has covered yet
                  completed      -> inspected, nothing left open
  unresolved face a face big enough to recognise that stayed Unknown after
                  UNRESOLVED_AFTER attempts. The same person seen again (new track ID,
                  after a camera move...) is merged by face embedding, so one stranger
                  is one entry. If that person is later recognised, the entry resolves itself.

"Attendance Verification Completed" is shown ONLY when every enrolled student is confirmed.
"""
import time
from pathlib import Path

import cv2
import numpy as np

import config

GX, GY = 3, 2
INSPECT_S = 3.0
UNRESOLVED_AFTER = 8
MERGE_SIM = 0.45


def init(con):
    con.execute("""CREATE TABLE IF NOT EXISTS coverage (
        session TEXT, region INTEGER, seen_s REAL, best_zoom REAL, faces INTEGER,
        verified INTEGER, tiny INTEGER, status TEXT, PRIMARY KEY (session, region))""")
    con.execute("""CREATE TABLE IF NOT EXISTS unresolved (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, uid INTEGER, first_seen TEXT,
        last_seen TEXT, attempts INTEGER, best_px INTEGER, reason TEXT, region INTEGER,
        snapshot TEXT, status TEXT DEFAULT 'open', resolved_as TEXT, UNIQUE (session, uid))""")
    con.commit()


def region_name(i):
    return f"{['Back', 'Front'][i // GX]}-{['left', 'centre', 'right'][i % GX]}"


def _ts(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


class Coverage:
    def __init__(self, session, enrolled):
        self.session, self.enrolled = session, list(enrolled)
        self.regions = [{"id": i, "seen_s": 0.0, "best_zoom": 1.0, "tracks": set(),
                         "verified": set(), "tiny": set()} for i in range(GX * GY)]
        self.tracks = {}
        self.unresolved = []          # dicts: uid, emb, attempts, best_px, reason, region, crop, status...
        self._next_uid = 1
        self._last = None

    # ---------------------------------------------------------------- geometry
    @staticmethod
    def _window(view):
        z, cx, cy = view if view else (1.0, 0.5, 0.5)
        half = 1 / (2 * z)
        return z, min(max(cx, half), 1 - half), min(max(cy, half), 1 - half), half

    def to_wide(self, nx, ny, view):
        """Point (0-1) in the current (maybe zoomed) view -> point in the wide view."""
        z, wx, wy, half = self._window(view)
        return wx - half + nx / z, wy - half + ny / z

    @staticmethod
    def region_of(wx, wy):
        return min(max(int(wx * GX), 0), GX - 1) + GX * min(max(int(wy * GY), 0), GY - 1)

    def visible_regions(self, view):
        z, wx, wy, half = self._window(view)
        out = []
        for r in self.regions:
            rx0, ry0 = (r["id"] % GX) / GX, (r["id"] // GX) / GY
            ox = min(wx + half, rx0 + 1 / GX) - max(wx - half, rx0)
            oy = min(wy + half, ry0 + 1 / GY) - max(wy - half, ry0)
            if ox > 0.5 / GX * 0.99 and oy > 0.5 / GY * 0.99:      # at least half the region visible
                out.append(r)
        return out

    # ---------------------------------------------------------------- per frame
    def update(self, info, shape, view, moving, present, now=None):
        now = now or time.time()
        dt = min(now - self._last, 1.0) if self._last else 0.0
        self._last = now
        z = view[0] if view else 1.0
        if not moving:
            for r in self.visible_regions(view):
                r["seen_s"] += dt
                r["best_zoom"] = max(r["best_zoom"], z)
        if moving:
            return
        h, w = shape[:2]
        clean = info.get("clean")
        for f in info.get("faces", []):
            x1, y1, x2, y2 = f["box"]
            wx, wy = self.to_wide((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, view)
            r = self.regions[self.region_of(wx, wy)]
            tid = f.get("track")
            t = self.tracks.setdefault(tid, {"attempts": 0, "best_px": 0, "uid": None, "reason": None})
            r["tracks"].add(tid)
            label = f["label"]
            if label != "Unknown":
                r["tiny"].discard(tid)
                if label in present:
                    r["verified"].add(label)
                    if t["uid"] is not None:
                        self._resolve(t["uid"], label)
                continue
            if f["px"] < config.MIN_FACE_PX:
                r["tiny"].add(tid)
                continue
            r["tiny"].discard(tid)
            if not f.get("tried"):
                continue
            t["attempts"] += 1
            if f.get("status") in ("low_light", "not_clear", "turned_away"):
                t["reason"] = f["status"]
            if f.get("emb") is not None and f["px"] >= t["best_px"]:
                t["best_px"], t["emb"] = f["px"], f["emb"]
                if clean is not None:
                    p = int((x2 - x1) * 0.5)
                    t["crop"] = clean[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p].copy()
            if t["uid"] is None and t["attempts"] >= UNRESOLVED_AFTER and t.get("emb") is not None:
                t["uid"] = self._add_unresolved(t, r["id"], now)
            elif t["uid"] is not None:
                self._touch(t["uid"], t, now)

    def _find(self, uid):
        return next(u for u in self.unresolved if u["uid"] == uid)

    def _add_unresolved(self, t, region, now):
        for u in self.unresolved:                          # same person again -> same entry
            if u["status"] == "open" and float(u["emb"] @ t["emb"]) >= MERGE_SIM:
                self._touch(u["uid"], t, now)
                return u["uid"]
        u = {"uid": self._next_uid, "emb": t["emb"], "attempts": t["attempts"], "best_px": t["best_px"],
             "reason": t["reason"] or "unknown", "region": region, "crop": t.get("crop"), "status": "open",
             "first": now, "last": now, "resolved_as": None, "dirty": True}
        self._next_uid += 1
        self.unresolved.append(u)
        return u["uid"]

    def _touch(self, uid, t, now):
        u = self._find(uid)
        u["attempts"] = max(u["attempts"], t["attempts"])
        u["last"], u["dirty"] = now, True
        if t["best_px"] > u["best_px"]:
            u["best_px"], u["emb"], u["crop"] = t["best_px"], t["emb"], t.get("crop")
        if t["reason"]:
            u["reason"] = t["reason"]

    def _resolve(self, uid, name):
        u = self._find(uid)
        if u["status"] == "open":
            u["status"], u["resolved_as"], u["dirty"] = "resolved", name, True

    # ---------------------------------------------------------------- state
    def region_status(self, r, open_regions):
        if r["seen_s"] < INSPECT_S:
            return "not_inspected"
        tiny_unzoomed = r["tiny"] and r["best_zoom"] < 2.0
        if r["id"] in open_regions or tiny_unzoomed:
            return "needs_review"
        return "completed"

    def regions_to_visit(self):
        """Wide-view centres of the regions still worth a zoomed look, most urgent first."""
        open_regions = {u["region"] for u in self.unresolved if u["status"] == "open"}
        order = {"not_inspected": 0, "needs_review": 1}
        todo = [(order[s], r["id"]) for r in self.regions
                for s in [self.region_status(r, open_regions)] if s in order
                and not (s == "not_inspected" and r["best_zoom"] >= 2.0)]
        return [((i % GX + 0.5) / GX, (i // GX + 0.5) / GY) for _, i in sorted(todo)]

    # ---------------------------------------------------------------- persistence
    def flush(self, con):
        open_regions = {u["region"] for u in self.unresolved if u["status"] == "open"}
        for r in self.regions:
            con.execute("INSERT OR REPLACE INTO coverage VALUES (?,?,?,?,?,?,?,?)",
                        (self.session, r["id"], round(r["seen_s"], 1), r["best_zoom"], len(r["tracks"]),
                         len(r["verified"]), len(r["tiny"]), self.region_status(r, open_regions)))
        for u in self.unresolved:
            if not u.pop("dirty", False):
                continue
            snap = None
            if u.get("crop") is not None and u["crop"].size:
                config.SNAPSHOT_DIR.mkdir(exist_ok=True)
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.session)
                snap = f"unresolved_{safe}_{u['uid']}.jpg"
                cv2.imwrite(str(config.SNAPSHOT_DIR / snap), u["crop"])
            # teacher decisions (assigned / ignored) are never overwritten by the camera
            con.execute("""INSERT INTO unresolved (session, uid, first_seen, last_seen, attempts, best_px,
                               reason, region, snapshot, status, resolved_as) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(session, uid) DO UPDATE SET last_seen=excluded.last_seen,
                               attempts=excluded.attempts, best_px=excluded.best_px, reason=excluded.reason,
                               snapshot=COALESCE(excluded.snapshot, snapshot),
                               status=CASE WHEN status='open' THEN excluded.status ELSE status END,
                               resolved_as=CASE WHEN status='open' THEN excluded.resolved_as ELSE resolved_as END""",
                        (self.session, u["uid"], _ts(u["first"]), _ts(u["last"]), u["attempts"], u["best_px"],
                         u["reason"], u["region"], snap, u["status"], u["resolved_as"]))
        con.commit()


# -------------------------------------------------------------------- report (from the DB)
def report(con, session, enrolled):
    present = {r[0] for r in con.execute("SELECT name FROM attendance WHERE session=?", (session,))}
    enrolled = list(enrolled)
    regions = [{"id": i, "name": region_name(i), "seen_s": s, "best_zoom": z, "faces": f, "verified": v,
                "tiny": t, "status": st}
               for i, s, z, f, v, t, st in con.execute(
                   "SELECT region, seen_s, best_zoom, faces, verified, tiny, status FROM coverage "
                   "WHERE session=? ORDER BY region", (session,))]
    cols = ["id", "uid", "first_seen", "last_seen", "attempts", "best_px", "reason", "region", "snapshot",
            "status", "resolved_as"]
    unres = [dict(zip(cols, r)) for r in con.execute(
        f"SELECT {', '.join(cols)} FROM unresolved WHERE session=? ORDER BY last_seen DESC", (session,))]
    for u in unres:
        u["region_name"] = region_name(u["region"]) if u["region"] is not None else "-"
    confirmed = sorted(n for n in enrolled if n in present)
    not_verified = sorted(n for n in enrolled if n not in present)
    open_u = [u for u in unres if u["status"] == "open"]
    weak = [r for r in regions if r["status"] != "completed"]
    if not not_verified:
        verdict = "Attendance Verification Completed"
    elif open_u or weak:
        verdict = "Coverage Review Required"
    elif not regions:
        verdict = "Scanning not started"
    else:
        verdict = "Recognition Pending"
    return {"session": session, "enrolled": len(enrolled), "confirmed": len(confirmed),
            "not_verified": not_verified, "unknown_individuals": len(open_u),
            "unresolved_attempts": sum(u["attempts"] or 0 for u in open_u),
            "regions": regions, "weak_regions": [r["name"] for r in weak], "unresolved": unres,
            "verdict": verdict,
            "headline": f"{len(confirmed)} Confirmed Present / {len(not_verified)} Not Yet Verified"}


def classroom_state(az_state, report_, scanning):
    """One classroom-level state for the UI, derived from the camera controller + coverage."""
    if report_ and not report_["not_verified"]:
        return "SESSION_COMPLETED"
    if not scanning:
        return "COVERAGE_VERIFICATION"
    m = {"REPOSITIONING": "REGION_SELECTED", "STABILIZING": "REGION_STABILIZING",
         "RECOGNIZING": "RECOGNITION_PROCESSING", "TARGET_LOST": "MULTI_FACE_TRACKING",
         "ATTENDANCE_RECORDED": "ATTENDANCE_CONFIRMATION", "TARGET_LOCKED": "UNRESOLVED_FACE_RETRY",
         "SEARCHING": "NEXT_REGION_SELECTION", "IDLE": "COVERAGE_VERIFICATION"}
    if az_state in m:
        return m[az_state]
    if report_ and report_["regions"] and all(r["status"] == "not_inspected" for r in report_["regions"]):
        return "CLASSROOM_DISCOVERY"
    return "MULTI_FACE_DETECTION"
