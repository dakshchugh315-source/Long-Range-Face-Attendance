"""Live attendance from webcam / CCTV.

python attend.py                       # session named after today's date
python attend.py --session "DL-Lecture"
python attend.py --source "rtsp://..." # override camera

Every face is tracked across frames. A student is marked present only after
VOTES_TO_MARK frames agree on the same name, so one bad frame can't mark
the wrong person. The face width in pixels is shown so you can see how far
the camera can reach.  Q = quit.
"""
import argparse
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
import classroom
import sessions_db
import students_db
from face_engine import Gallery, detect, embed, embed_batch, ensure_bgr, face_width, is_turned_away


# ---------------------------------------------------------------- database
def open_db(path=None):
    con = sqlite3.connect(path or config.DB_FILE, check_same_thread=False, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT NOT NULL,
        name TEXT NOT NULL,
        time TEXT NOT NULL,
        similarity REAL,
        face_px INTEGER,
        snapshot TEXT,
        UNIQUE(session, name))""")
    # proof columns (added later; migrate old databases)
    cols = {r[1] for r in con.execute("PRAGMA table_info(attendance)")}
    for col, typ in [("frame_snapshot", "TEXT"), ("votes", "INTEGER"),
                     ("method", "TEXT DEFAULT 'auto'"), ("camera", "TEXT"), ("zoom", "REAL")]:
        if col not in cols:
            con.execute(f"ALTER TABLE attendance ADD COLUMN {col} {typ}")
    # every auto-zoom move: when, how much, where, why, who was recognised during it
    con.execute("""CREATE TABLE IF NOT EXISTS zoom_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT, time TEXT, action TEXT, zoom REAL, x REAL, y REAL,
        face_px INTEGER, reason TEXT, seconds REAL, recognised TEXT)""")
    con.commit()
    students_db.init(con)
    sessions_db.init(con)
    classroom.init(con)
    return con


def _safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def mark_present(con, session, name, sim, px, frame, bbox, votes=None, camera=None, zoom=None):
    """Save proof (face crop + full annotated frame) and insert the record."""
    config.SNAPSHOT_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    x1, y1, x2, y2 = bbox
    pad = int((x2 - x1) * 0.6)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    base = f"{_safe(session)}_{_safe(name)}_{now:%H%M%S}"
    snap, full = f"{base}_face.jpg", f"{base}_frame.jpg"
    cv2.imwrite(str(config.SNAPSHOT_DIR / snap), crop)

    proof = frame.copy()
    cv2.rectangle(proof, (x1, y1), (x2, y2), (0, 220, 0), 3)
    cv2.putText(proof, name, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    banner = (f"{name} | sim {sim:.2f} | {px}px | {votes or '-'} votes | "
              + (f"zoom {zoom:.1f}x | " if zoom and zoom > 1.01 else "")
              + f"{now:%Y-%m-%d %H:%M:%S} | session {session}")
    cv2.rectangle(proof, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.putText(proof, banner, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(config.SNAPSHOT_DIR / full), proof)

    try:
        cur = con.execute(
            "INSERT OR IGNORE INTO attendance(session, name, time, similarity, face_px, snapshot, "
            "frame_snapshot, votes, method, camera, zoom) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (session, name, now.strftime("%Y-%m-%d %H:%M:%S"), round(float(sim), 3), int(px),
             snap, full, votes, "auto", str(camera) if camera is not None else None,
             round(float(zoom), 2) if zoom else 1.0))
        con.commit()
    except Exception:
        con.rollback()                       # DB failed: no record -> remove the proof photos too,
        for f in (snap, full):               # the caller retries on the next frame (no duplicate,
            (config.SNAPSHOT_DIR / f).unlink(missing_ok=True)   # no false success, no orphans)
        raise
    if cur.rowcount != 1:                    # already recorded (duplicate attempt): keep no extra proof
        for f in (snap, full):
            (config.SNAPSHOT_DIR / f).unlink(missing_ok=True)
    return cur.rowcount == 1


def mark_manual(con, session, name, by="teacher"):
    cur = con.execute(
        "INSERT OR IGNORE INTO attendance(session, name, time, method) VALUES (?,?,?,?)",
        (session, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"manual ({by})"))
    con.commit()
    return cur.rowcount == 1


# ---------------------------------------------------------------- tracker
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Track:
    """One face followed across frames. Its name comes from RECENT recognitions only
    (not an all-time majority), so if the box moves onto another person the name
    follows the new face instead of sticking to the old one."""
    _next = 0

    def __init__(self, bbox):
        self.id = Track._next
        Track._next += 1
        self.bbox = bbox
        self.missed = 0
        self.name = "Unknown"         # last confident recognition on this track
        self.since_rec = 10 ** 6      # frames since that recognition
        self.since_check = 10 ** 6    # frames since recognition last ran on this track
        self.v = (0.0, 0.0)           # centre velocity in px/frame (smoothed)
        self.crowded = False          # another face is close: identities could swap
        self.reacquired = True        # not continuously followed by box overlap -> must re-identify

    def predicted(self):
        """Where the box should be in the next frame (constant-velocity prediction), so two
        people crossing each other are not swapped by plain nearest-box matching."""
        dx, dy = self.v
        x1, y1, x2, y2 = self.bbox
        k = 1 + min(self.missed, 3)
        return (x1 + dx * k, y1 + dy * k, x2 + dx * k, y2 + dy * k)

    def move_to(self, b):
        ox, oy = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
        nx, ny = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        steps = 1 + self.missed
        self.v = (0.5 * self.v[0] + 0.5 * (nx - ox) / steps, 0.5 * self.v[1] + 0.5 * (ny - oy) / steps)
        self.bbox = b

    def label(self):
        if self.name != "Unknown" and self.since_rec <= config.LABEL_HOLD_FRAMES:
            return self.name, 1
        return "Unknown", 0


def _centre_close(a, b):
    """Low-FPS fallback when boxes no longer overlap: same size, centre moved < 0.75 face."""
    wa, wb = a[2] - a[0], b[2] - b[0]
    if wa <= 0 or wb <= 0 or not (0.6 < wa / wb < 1.67):
        return None
    d = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    return d if d < 0.75 * max(wa, wb) else None


def update_tracks(tracks, boxes, max_missed=15):
    """Greedy IoU matching, then centre-distance matching for what is left.
    Returns the track for each box."""
    assigned = [None] * len(boxes)
    free = set(range(len(tracks)))
    pred = [t.predicted() for t in tracks]
    pairs = sorted(((iou(pred[ti], b), ti, bi) for ti, t in enumerate(tracks)
                    for bi, b in enumerate(boxes)), reverse=True)
    for score, ti, bi in pairs:
        if score < 0.3:
            break
        if ti in free and assigned[bi] is None:
            assigned[bi] = tracks[ti]
            assigned[bi].reacquired = tracks[ti].missed > 0     # came back after a gap
            free.discard(ti)
    near = sorted((d, ti, bi) for ti in free for bi, b in enumerate(boxes) if assigned[bi] is None
                  for d in [_centre_close(pred[ti], b)] if d is not None)
    for d, ti, bi in near:
        if ti in free and assigned[bi] is None:
            assigned[bi] = tracks[ti]
            assigned[bi].reacquired = True                      # only "close by": could be someone else
            free.discard(ti)
    for ti in free:
        tracks[ti].missed += 1
    for bi, b in enumerate(boxes):
        if assigned[bi] is None:
            t = Track(b)
            tracks.append(t)
            assigned[bi] = t
        else:
            assigned[bi].move_to(b)
        assigned[bi].missed = 0
    # faces close to each other (crossing, sitting side by side) are re-identified every
    # frame instead of every REVERIFY_EVERY frames, so a swap can never stick
    for i, ti in enumerate(assigned):
        a = ti.bbox
        wa = a[2] - a[0]
        ti.crowded = any(
            j != i and np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
            < 1.6 * max(wa, b[2] - b[0])
            for j, b in enumerate(boxes))
    tracks[:] = [t for t in tracks if t.missed <= max_missed]
    return assigned


# ---------------------------------------------------------------- camera
def open_camera(src):
    """Open a webcam index, a video file or an RTSP/HTTP stream.

    Laptop webcams on Windows: the default Media Foundation backend often opens the camera
    but then delivers no frames (error -1072875772, e.g. when another app such as Teams
    touched it). DirectShow works on those machines, so a webcam index tries DirectShow
    first (MJPG, configured size) and falls back to the default backend. Each candidate
    must deliver a real frame before it is accepted."""
    if isinstance(src, str) and src.strip().isdigit():
        src = int(src)
    if isinstance(src, int):
        import sys
        backends = ([cv2.CAP_DSHOW] if sys.platform == "win32" else []) + [cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(src, be)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            for _ in range(10):                      # first frames can be empty while it warms up
                ok, _f = cap.read()
                if ok:
                    return cap
            cap.release()
        return cv2.VideoCapture()                    # not opened -> caller reports it
    return cv2.VideoCapture(src)


class LatestFrame:
    """Reads a live stream in a background thread and keeps only the newest
    frame, so a slow CPU never falls seconds behind the phone/CCTV."""

    def __init__(self, cap):
        self.cap, self.frame, self.ok = cap, None, True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.ok:
            try:
                ok, f = self.cap.read()
            except cv2.error:              # broken stream can raise instead of returning False
                ok, f = False, None
            with self.lock:
                self.ok, self.frame = (self.ok and ok), (f if ok else self.frame)

    def close(self):
        """Stop the reader thread, then release the camera (never release mid-read)."""
        self.ok = False
        self.thread.join(timeout=3)
        self.cap.release()

    def read(self):
        while self.ok and self.frame is None:
            time.sleep(0.01)
        with self.lock:
            f, self.frame = self.frame, None
        return (f is not None), f


# ---------------------------------------------------------------- voting
class Voter:
    """Counts recognitions per NAME in a sliding time window.

    Tracks break easily at low FPS (person moves between two frames -> new
    track -> votes lost). Counting per name keeps the evidence. Keeps the
    best-matching frame of each person as proof."""

    def __init__(self):
        self.hits = {}            # name -> [timestamps]
        self.best = {}            # name -> (sim, px, clean_frame, box)
        self.best_t = {}          # name -> when best was taken
        self.marked_t = {}        # name -> when it was marked during this run
        self.frame_no = 0         # frames processed in this run

    def add(self, name, sim, px, clean, box, now=None, zoom=1.0):
        now = now or time.time()
        self.hits.setdefault(name, []).append(now)
        b = self.best.get(name)
        if b is None or sim > b[0] or now - self.best_t[name] > config.VOTE_WINDOW_S:
            self.best[name] = (sim, px, clean, box, zoom)
            self.best_t[name] = now

    def count(self, name, now=None):
        now = now or time.time()
        h = [t for t in self.hits.get(name, []) if now - t <= config.VOTE_WINDOW_S]
        self.hits[name] = h
        return len(h)


def face_quality(frame, face):
    """(sharpness, brightness) of a face. Sharpness is the Laplacian variance of the
    contrast-normalised face at 112x112, divided by the face width - so it is comparable
    across face sizes and lighting (dark faces aren't called blurry)."""
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    h, w = frame.shape[:2]
    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return 0.0, 0.0
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = float(g.mean())
    g = cv2.normalize(cv2.resize(g, (112, 112)), None, 0, 255, cv2.NORM_MINMAX)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) / max(face_width(face), 1), bright


def votes_needed(voter, label):
    """3 checks when the best match is clearly strong, otherwise 5 (weak/blurry matches
    need more evidence so a look-alike is never marked)."""
    best = voter.best.get(label)
    if best is not None and best[0] >= config.STRONG_MATCH:
        return config.VOTES_STRONG
    return config.VOTES_TO_MARK


# ---------------------------------------------------------------- per frame
def process_frame(frame, tracks, gallery, present, con, session, camera=None, voter=None,
                  zoom=1.0, info=None, camera_moving=False):
    """Detect, track, recognise, vote, mark. Draws boxes on `frame` in place.
    Returns [(name, similarity, face_px)] for students newly marked present.
    If `info` (dict) is given it is filled with faces=[{box, px, label, votes}] and
    recognised={names matched in this frame} for the auto-zoom controller.
    camera_moving=True (zoom/pan in progress): faces are shown but NOT recognised and give
    no votes - frames grabbed while the camera moves are blurred and unreliable."""
    if voter is None:                     # callers should keep one Voter per session
        voter = process_frame.__dict__.setdefault("_voter", Voter())
    clean = None                          # proof image without boxes (copied only if needed)
    newly = []
    t0 = time.time()
    # far/small faces need the tiled scan; it is ~90% of detection time, so it runs every
    # TILE_EVERY-th frame (near faces are found by the whole-frame pass every frame)
    use_tiles = config.DET_TILES and voter.frame_no % max(1, config.TILE_EVERY) == 0
    voter.frame_no += 1
    faces = [f for f in detect(frame, tiles=use_tiles) if face_width(f) >= config.MIN_FACE_PX // 2]
    t_det = time.time() - t0
    boxes = [tuple(int(v) for v in f.bbox) for f in faces]
    owners = update_tracks(tracks, boxes)

    # 1) recognise. A track already marked present is re-checked every REVERIFY_EVERY frames,
    #    so if its box has moved onto someone else that person still gets recognised.
    results = {}                                   # face index -> (name, sim)
    todo = []                                      # faces to recognise this frame
    for i, (face, tr) in enumerate(zip(faces, owners)):
        tr.since_rec += 1
        tr.since_check += 1
        px = face_width(face)
        if px < config.MIN_FACE_PX or camera_moving:   # too small / camera moving: no recognition
            continue
        marked_track = tr.label()[0] in present
        if (marked_track and tr.since_check < config.REVERIFY_EVERY and not tr.crowded
                and not tr.reacquired):
            continue                                # same face, followed frame to frame: skip work
        tr.since_check = 0
        todo.append(i)
    embed_batch(frame, [faces[i] for i in todo])    # ALL faces of the frame in one model call
    for i in todo:
        face, tr = faces[i], owners[i]
        name, sim = gallery.match(face.normed_embedding, face_width(face))
        if name != "Unknown":
            results[i] = (name, sim)
        elif tr.label()[0] in present:
            tr.since_rec = 0       # re-check inconclusive (head turned): keep the name; only a
                                   # DIFFERENT confident name may replace it
    if info is not None:
        info["t_detect"], info["t_recog"] = t_det, time.time() - t0 - t_det
    # 2) one name can only be ONE face per frame: if two faces match the same student,
    #    keep the better match; the other is treated as unknown (no vote, no label).
    best_for = {}
    for i, (name, sim) in results.items():
        if name not in best_for or sim > results[best_for[name]][1]:
            best_for[name] = i
    for i, (name, sim) in list(results.items()):
        if best_for[name] != i:
            del results[i]
            owners[i].name, owners[i].since_rec = "Unknown", 10 ** 6
    tried = set(todo)
    if info is not None and any(i not in results for i in tried):
        if clean is None:
            clean = frame.copy()                     # unannotated frame for unresolved-face crops
        info["clean"] = clean
    # 3) vote, mark, draw
    for i, (face, box, tr) in enumerate(zip(faces, boxes, owners)):
        px = face_width(face)
        if i in results:
            name, sim = results[i]
            tr.name, tr.since_rec = name, 0
            if clean is None:
                clean = frame.copy()
            voter.add(name, sim, px, clean, box, zoom=zoom)
            if info is not None:
                info.setdefault("recognised", set()).add(name)
        label, _ = tr.label()
        n = voter.count(label) if label != "Unknown" else 0
        need = votes_needed(voter, label) if label != "Unknown" else config.VOTES_TO_MARK
        if info is not None:
            # votes = scan progress shown as "k/5"; the auto-zoom controller must know it so it
            # never moves the camera away from a face that is being scanned
            info.setdefault("faces", []).append({
                "box": box, "px": px, "label": label, "votes": n, "need": need, "track": tr.id,
                "tried": i in tried,
                "emb": face.normed_embedding if (i in tried and i not in results) else None})

        if label != "Unknown" and n >= need and label not in present:
            bsim, bpx, bframe, bbox, bzoom = voter.best[label]
            if mark_present(con, session, label, bsim, bpx, bframe, bbox,
                            votes=n, camera=camera, zoom=bzoom):
                newly.append((label, bsim, bpx))
                voter.marked_t[label] = time.time()
            present.add(label)

        # status shown on the box = the real internal state of this face
        if label in present:
            fresh = time.time() - voter.marked_t.get(label, 0) < config.MARKED_SHOW_S
            status, color, text = ("marked", (0, 200, 0), f"{label} - Marked") if fresh else \
                ("already_marked", (0, 160, 0), f"{label} - Already marked")
        elif camera_moving:
            status, color, text = "stabilizing", (160, 160, 160), "Stabilizing..."
        elif label != "Unknown":
            status, color, text = "recognizing", (0, 200, 255), f"{label} {n}/{need}"
        elif px < config.MIN_FACE_PX:
            status, color, text = "too_far", (128, 128, 128), "Too far"
        elif tr.since_check > 10 ** 5:
            status, color, text = "detecting", (200, 200, 200), "Detecting..."
        else:
            sharp, bright = face_quality(frame, face)
            if bright < config.LOW_LIGHT_LEVEL:
                status, color, text = "low_light", (0, 140, 255), "Low light"
            elif is_turned_away(face):
                status, color, text = "turned_away", (200, 160, 60), "Turned away"
            elif sharp < config.BLUR_MIN:
                status, color, text = "not_clear", (0, 140, 255), "Face not clear"
            else:
                status, color, text = "unknown", (0, 0, 255), "Unknown"
        if info is not None:
            info["faces"][-1]["status"] = status
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{text} [{px}px]", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return newly


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--source", default=None, help="camera index or RTSP/HTTP url")
    args = ap.parse_args()

    gallery = Gallery()
    con = open_db()
    present = {r[0] for r in con.execute(
        "SELECT name FROM attendance WHERE session=?", (args.session,))}

    src = config.CAMERA_SOURCE if args.source is None else args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = open_camera(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera: {src}")
    is_file = isinstance(src, str) and Path(src).is_file()
    reader = cap if is_file else LatestFrame(cap)   # recorded video: process every frame

    print(f"Session '{args.session}' | enrolled: {len(gallery.names)} | already present: {len(present)}")
    tracks, fps, t_prev, voter = [], 0.0, time.time(), Voter()

    while True:
        ok, frame = reader.read()
        if not ok:
            print("Camera stream ended / failed.")
            break
        frame = ensure_bgr(frame)

        for name, sim, px in process_frame(frame, tracks, gallery, present, con, args.session,
                                           src, voter):
            print(f"[{datetime.now():%H:%M:%S}] PRESENT: {name} (sim {sim:.2f}, {px}px)")

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        cv2.putText(frame, f"Present {len(present)}/{len(gallery.names)}  |  {fps:.1f} FPS",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Attendance  (Q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    absent = sorted(set(gallery.names) - present)
    print(f"\nPresent ({len(present)}): {', '.join(sorted(present)) or '-'}")
    print(f"Absent  ({len(absent)}): {', '.join(absent) or '-'}")


if __name__ == "__main__":
    main()
