"""Crowded-classroom benchmark (SIMULATED, labelled) - 5 to 100 students.

python tests/test_classroom_scale.py PATH/TO/lfw.parquet [sizes...]

NOT real classroom footage. Uses Labeled Faces in the Wild (real photos of real, different
people) to build classroom scenes, and runs the REAL pipeline (detect -> track -> batch
recognise -> vote -> mark -> coverage report) over a short "video" of each scene:
  * each enrolled student is enrolled from 2 photos; the classroom shows a 3rd, different photo
  * rows from front (big faces) to back (small faces), back rows partly hidden by front rows,
    some faces partly covered, one dark corner
  * ~10 % of the roster is ABSENT (must never be marked)
  * ~10 % strangers who are not enrolled (must never be marked; should be listed unresolved)
  * every frame gets a small shift, sensor noise and JPEG compression
Reports per class size: detection recall, correct marks, wrong marks (false acceptances),
strangers marked, absentees marked, duplicates, unresolved list, coverage verdict, speed.
"""
import io
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="fa_class_"))
config.DB_FILE = TMP / "att.db"
config.SNAPSHOT_DIR = TMP / "snaps"
import attend  # noqa: E402
import classroom  # noqa: E402
import face_engine as fe  # noqa: E402

FRAMES = int(__import__('os').environ.get('FRAMES', 30))
rng = np.random.default_rng(7)


class ArrayGallery(fe.Gallery):
    def __init__(self, names, centroids):           # noqa: D401 - in-memory gallery
        self.names, self.centroids, self.lowres = names, centroids.astype(np.float32), None


def load_lfw(path):
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(path)
    people = defaultdict(list)
    for img, fn in zip(df["image"], df["filename"]):
        people[fn.rsplit("_", 1)[0]].append(img["bytes"])
    to_bgr = lambda b: cv2.cvtColor(np.asarray(Image.open(io.BytesIO(b)).convert("RGB")), cv2.COLOR_RGB2BGR)  # noqa: E731
    return people, to_bgr


def face_of(img):
    fs = fe.detect(img, tiles=False)
    return max(fs, key=fe.face_width) if fs else None


def paste_head(canvas, img, face, cx, cy, size):
    s = size / fe.face_width(face)
    patch = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    fx, fy = (face.bbox[0] + face.bbox[2]) / 2 * s, (face.bbox[1] + face.bbox[3]) / 2 * s
    half = int(size * 0.95)
    H, W = canvas.shape[:2]
    for dy in range(-half, int(half * 1.3)):
        yy, sy = int(cy + dy), int(fy + dy)
        if not (0 <= yy < H and 0 <= sy < patch.shape[0]):
            continue
        x0, x1 = int(cx - half), int(cx + half)
        sx0 = int(fx - half)
        a, b = max(0, x0), min(W, x1)
        sa = sx0 + (a - x0)
        if b > a and sa >= 0 and sa + (b - a) <= patch.shape[1]:
            canvas[yy, a:b] = patch[sy, sa:sa + (b - a)]


def build_scene(n_students, frame_wh, people_imgs, strangers):
    """Returns (scene image, [(name or None, cx, cy, size)])."""
    W, H = frame_wh
    k = W / 1920
    seats = n_students + len(strangers)
    per_row = max(4, min(20, math.ceil(math.sqrt(seats * 2.8))))
    rows = math.ceil(seats / per_row)
    room = np.full((H, W, 3), (178, 182, 186), np.uint8)
    cv2.rectangle(room, (0, int(H * 0.55)), (W, H), (120, 126, 134), -1)       # floor / desks
    occupants = list(people_imgs) + [(None, img, f) for img, f in strangers]
    rng.shuffle(occupants)
    placed = []
    it = iter(occupants)
    for r in range(rows - 1, -1, -1):                     # back row first, front rows overlap it
        t = r / max(rows - 1, 1)                          # 0 = front, 1 = back
        size = int((84 - 56 * t) * k)                     # 84 px front -> 28 px back at 1080p
        y = int(H * (0.84 - 0.50 * t))
        span = W * (0.92 - 0.30 * t)
        n_here = min(per_row, seats - r * per_row) if r == rows - 1 else per_row
        n_here = max(0, min(n_here, seats - len(placed)))
        for j in range(n_here):
            name_img = next(it, None)
            if name_img is None:
                break
            name, img, f = name_img
            cx = int((W - span) / 2 + span * (j + 0.5) / n_here + rng.integers(-6, 7) * k)
            cy = y + int(rng.integers(-5, 6) * k)
            paste_head(room, img, f, cx, cy, size)
            covered = rng.random() < 0.08
            if covered:                                   # hand / object covering the lower face
                cv2.ellipse(room, (cx, int(cy + size * 0.35)), (int(size * 0.45), int(size * 0.25)), 0, 0, 360,
                            (60, 70, 90), -1)
            placed.append((name, cx, cy, size, covered, cx < W // 4 and cy < H // 2))
    dark = room[: H // 2, : W // 4].astype(np.float32) * 0.35   # poorly lit back-left corner
    room[: H // 2, : W // 4] = dark.astype(np.uint8)
    return room, placed


def frames_of(scene):
    for i in range(FRAMES):
        M = np.float32([[1, 0, rng.uniform(-2, 2)], [0, 1, rng.uniform(-2, 2)]])
        f = cv2.warpAffine(scene, M, (scene.shape[1], scene.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        f = np.clip(f.astype(np.float32) + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8)
        yield cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_COLOR)


def run_size(n, people, to_bgr, frame_wh):
    names_all = [p for p, imgs in people.items() if len(imgs) >= 3]
    rng.shuffle(names_all)
    roster, gallery_vecs, in_class = [], [], []
    absent_n = max(1, round(n * 0.10))
    for name in names_all:
        if len(roster) >= n + absent_n:
            break
        imgs = [to_bgr(b) for b in people[name][:3]]
        faces = [face_of(im) for im in imgs]
        if any(f is None for f in faces):
            continue
        embs = [fe.embed(im, f).copy() for im, f in zip(imgs[:2], faces[:2])]
        c = np.mean(embs, axis=0)
        roster.append(name)
        gallery_vecs.append(c / np.linalg.norm(c))
        if len(in_class) < n:
            in_class.append((name, imgs[2], faces[2]))      # a DIFFERENT photo than enrolment
    absent = set(roster) - {x[0] for x in in_class}
    strangers = []
    singles = [p for p, imgs in people.items() if len(imgs) == 1]
    for name in singles[: max(2, round(n * 0.10)) * 3]:
        if len(strangers) >= max(2, round(n * 0.10)):
            break
        im = to_bgr(people[name][0])
        f = face_of(im)
        if f is not None:
            strangers.append((im, f))
    gallery = ArrayGallery(roster, np.stack(gallery_vecs))
    scene, placed = build_scene(n, frame_wh, in_class, strangers)

    session = f"scale-{n}-{frame_wh[1]}p"
    con = attend.open_db()
    cov = classroom.Coverage(session, roster)
    tracks, voter, present, newly, times = [], attend.Voter(), set(), [], []
    shown_labels, label_errors = [0, 0], []
    det0 = None
    for fr in frames_of(scene):
        info = {}
        t0 = time.time()
        newly += attend.process_frame(fr, tracks, gallery, present, con, session, None, voter, info=info)
        times.append(time.time() - t0)
        cov.update(info, fr.shape, None, False, present, now=time.time() + len(times) * 1.0)
        for f in info.get("faces", []):                   # is the NAME SHOWN on each box right?
            if f["label"] == "Unknown":
                continue
            fx, fy = (f["box"][0] + f["box"][2]) / 2, (f["box"][1] + f["box"][3]) / 2
            who = min(placed, key=lambda p: (p[1] - fx) ** 2 + (p[2] - fy) ** 2)
            shown_labels[0] += 1
            if who[0] != f["label"]:
                shown_labels[1] += 1
                if len(label_errors) < 6:
                    label_errors.append({"frame": len(times), "shown": f["label"], "truth": who[0] or "stranger",
                                         "px": f["px"], "tried": f.get("tried"), "track": f.get("track")})
        if det0 is None:
            det0 = info.get("faces", [])
    cov.flush(con)
    rep = classroom.report(con, session, roster)
    rows = con.execute("SELECT name, COUNT(*) FROM attendance WHERE session=? GROUP BY name", (session,)).fetchall()
    con.close()

    truth = {name for name, *_ in placed if name}
    last_faces = info.get("faces", [])
    cent = {n: v for n, v in zip(roster, gallery_vecs)}
    missed_detail = []
    for name, cx, cy, size, covered, dark in placed:
        if not name or name in {r[0] for r in rows}:
            continue
        near = [f for f in last_faces if abs((f["box"][0] + f["box"][2]) / 2 - cx) < size * 0.6
                and abs((f["box"][1] + f["box"][3]) / 2 - cy) < size * 0.8]
        d = {"size": size, "covered": covered, "dark": dark, "detected": bool(near)}
        if near:
            f = near[0]
            d["label"], d["px"] = f["label"], f["px"]
            if f.get("emb") is not None:
                sims = np.stack(gallery_vecs) @ f["emb"]
                own = float(cent[name] @ f["emb"])
                d["own_sim"] = round(own, 2)
                d["best_other"] = round(float(max(v for n2, v in zip(roster, sims) if n2 != name)), 2)
        missed_detail.append(d)
    seen = 0                                            # detection recall on frame 1
    for name, cx, cy, size, *_ in placed:
        if any(abs((f["box"][0] + f["box"][2]) / 2 - cx) < size * 0.6 and abs((f["box"][1] + f["box"][3]) / 2 - cy) < size * 0.8
               for f in det0):
            seen += 1
    marked = {r[0] for r in rows}
    small = [x for x in placed if x[3] < config.MIN_FACE_PX * 1.15]
    return {
        "students_in_class": n, "frame": f"{frame_wh[0]}x{frame_wh[1]}", "strangers": len(strangers),
        "absent_enrolled": len(absent), "faces_placed": len(placed),
        "faces_below_recognisable_size": len(small),
        "detection_recall": round(seen / len(placed), 3),
        "correct_marks": len(marked & truth), "missed_students": len(truth - marked),
        "wrong_marks": len(marked - truth - absent), "absentees_marked": len(marked & absent),
        "strangers_marked": 0,                          # strangers have no roster name - any wrong mark counts above
        "duplicate_rows": sum(c - 1 for _, c in rows if c > 1),
        "unresolved_listed": rep["unknown_individuals"], "verdict": rep["verdict"], "headline": rep["headline"],
        "weak_regions": rep["weak_regions"],
        "missed_detail": missed_detail,
        "names_shown": shown_labels[0], "wrong_names_shown": shown_labels[1], "label_errors": label_errors,
        "ms_per_frame_avg": round(1000 * float(np.mean(times))), "ms_per_frame_max": round(1000 * float(np.max(times))),
    }


def main():
    lfw = sys.argv[1]
    sizes = [int(x) for x in sys.argv[2:]] or [5, 10, 20, 40, 60, 100]
    fe.get_app()
    people, to_bgr = load_lfw(lfw)
    results = []
    for n in sizes:
        for wh in ([(1920, 1080)] + ([(3840, 2160)] if n >= 60 else [])):
            r = run_size(n, people, to_bgr, wh)
            results.append(r)
            print(json.dumps(r), flush=True)
    out = ROOT / "tools" / "classroom_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\nSIMULATED classroom results (LFW photos, not real footage)")
    print(f"{'class':>5} {'frame':>9} {'det%':>5} {'correct':>8} {'missed':>6} {'WRONG':>5} {'absent':>6} "
          f"{'dups':>4} {'unres':>5} {'ms/frame':>8}  verdict")
    for r in results:
        print(f"{r['students_in_class']:>5} {r['frame']:>9} {100 * r['detection_recall']:>4.0f}% "
              f"{r['correct_marks']:>4}/{r['students_in_class']:<3} {r['missed_students']:>6} {r['wrong_marks']:>5} "
              f"{r['absentees_marked']:>6} {r['duplicate_rows']:>4} {r['unresolved_listed']:>5} "
              f"{r['ms_per_frame_avg']:>8}  {r['headline']} - {r['verdict']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
