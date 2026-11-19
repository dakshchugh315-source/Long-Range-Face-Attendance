"""Tests for the audit fixes (multi-detection, tracking, IR frames, threads, enrolment, zoom mapping).

python tests/test_audit_fixes.py
Uses real photos from dataset/ and the real models; real attendance.db is not touched.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="fa_audit_"))
config.DB_FILE = TMP / "att.db"
config.SNAPSHOT_DIR = TMP / "snaps"
import notify  # noqa: E402

notify.SECRETS = TMP / "secrets.json"
notify.TelegramBot._call = lambda self, method, timeout=35, **p: [] if method == "getUpdates" else {}

import attend  # noqa: E402
import autozoom as az  # noqa: E402
import face_engine as fe  # noqa: E402

results = []


def check(name, ok, info=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   <- {info}" if not ok and info is not None else ""))


def frontal(name):
    for f in sorted((config.DATASET_DIR / name).glob("*.jpg")):
        img = cv2.imread(str(f))
        fs = fe.detect(img, tiles=False)
        if fs:
            face = max(fs, key=fe.face_width)
            if not fe.is_turned_away(face) and fe.face_width(face) > 80:
                return img, face
    raise RuntimeError(f"no frontal photo for {name}")


def paste(canvas, img, face, cx, cy, size):
    """Put `img` scaled so its face is `size` px wide with the face centred at (cx, cy)."""
    s = size / fe.face_width(face)
    patch = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    fx, fy = int((face.bbox[0] + face.bbox[2]) / 2 * s), int((face.bbox[1] + face.bbox[3]) / 2 * s)
    H, W = canvas.shape[:2]
    pw = ph = int(size * 1.8)                         # only the head area, so neighbours don't overlap
    x0, y0 = cx - pw // 2, cy - ph // 2
    sx, sy = fx - pw // 2, fy - ph // 2
    for dy in range(ph):
        yy, sy2 = y0 + dy, sy + dy
        if 0 <= yy < H and 0 <= sy2 < patch.shape[0]:
            xa, xb = max(0, x0), min(W, x0 + pw)
            sa = sx + (xa - x0)
            if xb > xa and 0 <= sa and sa + (xb - xa) <= patch.shape[1]:
                canvas[yy, xa:xb] = patch[sy2, sa:sa + (xb - xa)]
    return canvas


def blank():
    return np.full((1080, 1920, 3), 185, np.uint8)


def run_frames(frames, present=None, tracks=None, voter=None):
    con = attend.open_db()
    g = fe.Gallery()
    present = set() if present is None else present
    tracks = [] if tracks is None else tracks
    voter = voter or attend.Voter()
    infos = []
    for fr in frames:
        info = {}
        attend.process_frame(fr.copy(), tracks, g, present, con, "audit", None, voter, info=info)
        infos.append(info)
    con.close()
    return infos, present, tracks, voter


def t1_one_name_one_face():
    print("\n1. The same student can't be two faces in one frame (no double votes)")
    img, face = frontal("Daksh Chugh")
    fr = paste(paste(blank(), img, face, 600, 500, 150), img, face, 1300, 500, 150)   # twin
    infos, _, _, voter = run_frames([fr])
    labels = [f["label"] for f in infos[0]["faces"]]
    check("two faces found", len(labels) == 2, labels)
    check("only ONE of them gets the name", labels.count("Daksh Chugh") == 1, labels)
    check("only one vote was added this frame", voter.count("Daksh Chugh") == 1, voter.count("Daksh Chugh"))


def t2_swap_on_marked_track():
    print("\n2. Box moves from a marked student onto another one -> the new one is recognised")
    a, fa = frontal("Daksh Chugh")
    b, fb = frontal("Divyansh")
    frames_a = [paste(blank(), a, fa, 900, 500, 160) for _ in range(4)]
    infos, present, tracks, voter = run_frames(frames_a)
    check("first student marked (precondition)", "Daksh Chugh" in present, present)
    frames_b = [paste(blank(), b, fb, 905, 505, 160) for _ in range(12)]          # same spot
    infos, present, tracks, voter = run_frames(frames_b, present, tracks, voter)
    shown = [i["faces"][0]["label"] for i in infos]
    check("same track kept (the box really 'moved' onto the new person)", len(tracks) == 1, len(tracks))
    check("new student gets recognised within REVERIFY_EVERY+1 frames",
          "Divyansh" in shown[:config.REVERIFY_EVERY + 1], shown)
    check("and is marked present", "Divyansh" in present, present)
    check("no longer shown as the old student", shown[-1] == "Divyansh", shown[-3:])


def t3_low_fps_tracking():
    print("\n3. Low FPS: face jumps far between frames -> keeps the same track (no flicker)")
    tracks = []
    w = 120
    ids = []
    for k in range(6):
        x = 600 + k * 70                                  # 0.6 face width per frame, IoU < 0.3
        ids.append(attend.update_tracks(tracks, [(x, 400, x + w, 400 + w)])[0].id)
    check("one track id the whole time", len(set(ids)) == 1, ids)
    two = attend.update_tracks(tracks, [(1020, 400, 1140, 520), (300, 400, 420, 520)])   # next step + new face
    check("a second, far away face gets its own track", two[0].id == ids[0] and two[1].id != ids[0],
          [t.id for t in two])


def t4_ir_grayscale():
    print("\n4. IR / grayscale (1-channel) and 4-channel frames don't crash")
    img, face = frontal("Kanika")
    fr = paste(blank(), img, face, 960, 500, 160)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    try:
        fs = fe.detect(gray)
        name, sim = fe.Gallery().match(fe.embed(gray, fs[0]))
        check("grayscale frame: face found + recognised", name == "Kanika", (name, sim))
    except Exception as e:
        check("grayscale frame: face found + recognised", False, repr(e))
    try:
        fs = fe.detect(cv2.cvtColor(fr, cv2.COLOR_BGR2BGRA))
        check("BGRA frame works", len(fs) == 1)
    except Exception as e:
        check("BGRA frame works", False, repr(e))


def t5_threads_and_api():
    print("\n5. Teacher add/remove never touches the live set from another thread")
    import app as appmod
    client = appmod.app.test_client()
    eng = appmod.engine
    eng.running, eng.session, eng.present = True, "live", {"Yatri"}
    eng.manual_q.clear()
    r = client.post("/api/manual", json={"session": "live", "name": "Kanika", "action": "add"})
    r2 = client.post("/api/manual", json={"session": "live", "name": "Yatri", "action": "remove"})
    check("requests accepted", r.status_code == 200 and r2.status_code == 200)
    check("live set unchanged by the web thread", eng.present == {"Yatri"}, eng.present)
    check("changes queued for the camera thread", list(eng.manual_q) == [("add", "Kanika"), ("remove", "Yatri")],
          list(eng.manual_q))
    eng.running, eng.session, eng.present = False, None, set()
    eng.manual_q.clear()
    r = client.get("/api/export")
    check("CSV export without a session -> clear 400 error", r.status_code == 400, r.status_code)

    print("\n6. One bad frame does not end the class")
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not clip:
        print("  (skipped: pass a video clip of an enrolled student)")
        return
    import time
    real = appmod.process_frame
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("simulated decode glitch")
        return real(*a, **k)
    appmod.process_frame = flaky
    try:
        eng.start(str(clip), "flaky")
        t0 = time.time()
        while eng.running and time.time() - t0 < 120:
            time.sleep(0.5)
        texts = [e["text"] for e in eng.events]
        check("glitch was logged as 'Frame skipped'", any("Frame skipped" in t for t in texts), texts[:3])
        check("session kept running to the end of the video", eng.error == "Video file finished.", eng.error)
        check("stage timings reported", {"wait", "detect", "recognise", "encode"} <= set(eng.timing), eng.timing)
    finally:
        appmod.process_frame = real
        eng.stop()


def t7_enrolment():
    print("\n7. Enrolment: HEIC photos are read, big photos are fast, embeddings unchanged")
    import enroll
    try:
        import pillow_heif
        from PIL import Image
        src = sorted((config.DATASET_DIR / "Kashish").glob("*.jpg"))[0]
        folder = TMP / "heic_test"
        folder.mkdir()
        pillow_heif.register_heif_opener()
        Image.open(src).save(folder / "photo.heic", format="HEIF")
        imgs = list(enroll.read_images(folder))
        check("HEIC photo is read (was silently skipped)", len(imgs) == 1 and imgs[0].ndim == 3)
    except Exception as e:
        check("HEIC photo is read (was silently skipped)", False, repr(e))
    old = np.load(config.EMBEDDINGS_FILE)
    out = TMP / "emb.npz"
    real_file, config.EMBEDDINGS_FILE = config.EMBEDDINGS_FILE, out
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            enroll.main()
        new = np.load(out)
        same = list(new["names"]) == list(old["names"])
        sims = (new["centroids"] * old["centroids"]).sum(1) if same else [0]
        check("same 9 students enrolled", same, list(new["names"]))
        check("embeddings practically identical (cos >= 0.98)", float(min(sims)) >= 0.98, str(np.round(sims, 3)))
    finally:
        config.EMBEDDINGS_FILE = real_file


def t8_zoom_mapping():
    print("\n8. After a zoom the target face is exactly where the controller expects it")
    W4, H4 = 3840, 2160
    img, face = frontal("Abhrojet")
    worst = 0.0
    for cx, cy in [(0.2, 0.3), (0.5, 0.5), (0.85, 0.7), (0.05, 0.5), (0.95, 0.9)]:
        room = np.full((H4, W4, 3), 185, np.uint8)
        paste(room, img, face, int(cx * W4), int(cy * H4), 40)
        for z in (1.5, 2.0, 3.0):
            steps = list(range(100, 801, 7))
            zs = az.snap_zoom(z, steps) / 100
            crop_x, crop_y = az.crop_for(cx, zs, "free_range"), az.crop_for(cy, zs, "free_range")
            ww, wh = W4 / zs, H4 / zs                        # what the phone shows (free-range pan)
            wcx = ww / 2 + crop_x / 100 * (W4 - ww)
            wcy = wh / 2 + crop_y / 100 * (H4 - wh)
            x0, y0 = int(wcx - ww / 2), int(wcy - wh / 2)
            view = cv2.resize(room[y0:y0 + int(wh), x0:x0 + int(ww)], (1920, 1080),
                              interpolation=cv2.INTER_LINEAR)
            fs = fe.detect(view)
            if not fs:
                check(f"face visible after zoom {zs}x at ({cx},{cy})", False)
                continue
            f = max(fs, key=fe.face_width)
            got = ((f.bbox[0] + f.bbox[2]) / 2 / 1920, (f.bbox[1] + f.bbox[3]) / 2 / 1080)
            wx = min(max(cx, 1 / (2 * zs)), 1 - 1 / (2 * zs))   # the controller's own formula
            wy = min(max(cy, 1 / (2 * zs)), 1 - 1 / (2 * zs))
            exp = (0.5 + (cx - wx) * zs, 0.5 + (cy - wy) * zs)
            worst = max(worst, abs(got[0] - exp[0]), abs(got[1] - exp[1]))
    check("predicted position within 3% of the frame at every zoom/position", worst < 0.03, f"{worst:.3f}")


def main():
    t1_one_name_one_face()
    t2_swap_on_marked_track()
    t3_low_fps_tracking()
    t4_ir_grayscale()
    t5_threads_and_api()
    t7_enrolment()
    t8_zoom_mapping()
    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
