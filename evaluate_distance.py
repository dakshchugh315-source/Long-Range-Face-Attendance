"""How far can the system recognise people?  (synthetic long-range test)

python evaluate_distance.py [--background some_frame.jpg] [--tiles off|on|both]

For every enrolled photo it:
  1. shrinks the person so the face is N pixels wide (N = 80 ... 12),
  2. adds CCTV-like blur + JPEG compression,
  3. pastes it at a random spot in a 1920x1080 "classroom" frame,
  4. runs the real pipeline (detect -> embed -> match) on the full frame,
  5. compares with a gallery built WITHOUT that photo (leave-one-out).
Prints detection rate / correct / wrong per face size, and converts face size
to metres for a Samsung M31 streaming 1080p (1x and 2x zoom).
"""
import argparse
import time

import cv2
import numpy as np

import config
import face_engine
from enroll import read_images

SIZES = [80, 60, 45, 35, 28, 22, 18, 15, 12]
W, H = 1920, 1080
# M31 main camera ~26 mm equiv -> ~65 deg horizontal FOV in 16:9 video
FOCAL_PX_1080 = (W / 2) / np.tan(np.radians(65 / 2))
FACE_M = 0.15            # typical face width incl. cheeks


def metres(px, zoom=1.0):
    return FOCAL_PX_1080 * zoom * FACE_M / px


def load_people():
    people = []
    for d in sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()):
        for img in read_images(d):
            fs = face_engine.detect(img, tiles=False)
            if not fs:
                continue
            f = max(fs, key=face_engine.face_width)
            people.append((d.name, img, f, face_engine.embed(img, f).copy(),
                           face_engine.lowres_embedding(img, f)))
    return people


def degrade(img, scale, rng):
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        small = cv2.GaussianBlur(small, (3, 3), 0.6)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 80))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def run(people, bg, tiles, rng, lowres):
    names = sorted({p[0] for p in people})
    E = np.stack([p[3] for p in people])
    L = np.stack([p[4] for p in people])
    Y = np.array([names.index(p[0]) for p in people])
    res = {s: dict(n=0, det=0, ok=0, wrong=0, t=0.0) for s in SIZES}
    for i, (name, img, face, _, _) in enumerate(people):
        m = np.ones(len(E), bool)
        m[i] = False
        C = np.stack([E[m & (Y == k)].mean(0) for k in range(len(names))])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        CL = np.stack([L[m & (Y == k)].mean(0) for k in range(len(names))])
        CL /= np.linalg.norm(CL, axis=1, keepdims=True)
        for s in SIZES:
            scale = s / face_engine.face_width(face)
            patch = degrade(img, scale, rng)
            ph, pw = patch.shape[:2]
            if ph >= H or pw >= W:
                continue
            canvas = bg.copy()
            y, x = int(rng.integers(0, H - ph)), int(rng.integers(0, W - pw))
            canvas[y:y + ph, x:x + pw] = patch
            # where the face now is
            fx1, fy1, fx2, fy2 = face.bbox * scale + [x, y, x, y]
            t0 = time.time()
            found = face_engine.detect(canvas, tiles=tiles)
            res[s]["t"] += time.time() - t0
            res[s]["n"] += 1
            hit = None
            for f in found:
                cx, cy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
                if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                    hit = f
                    break
            if hit is None:
                continue
            res[s]["det"] += 1
            e = face_engine.embed(canvas, hit)
            sims = C @ e
            if lowres and face_engine.face_width(hit) < config.LOWRES_BELOW_PX:
                sims = np.maximum(sims, CL @ e)
            o = np.argsort(sims)[::-1]
            if sims[o[0]] < config.MATCH_THRESHOLD or sims[o[0]] - sims[o[1]] < config.MARGIN:
                continue
            if names[o[0]] == name:
                res[s]["ok"] += 1
            else:
                res[s]["wrong"] += 1
    return res


def report(title, res):
    print(f"\n{title}")
    print(f"{'face px':>8} {'M31 1x':>8} {'M31 2x':>8} {'detected':>9} {'correct':>8} "
          f"{'WRONG':>6} {'det ms':>7}")
    for s in SIZES:
        r = res[s]
        if not r["n"]:
            continue
        n = r["n"]
        print(f"{s:>8} {metres(s):>7.1f}m {metres(s, 2):>7.1f}m {r['det'] / n:>9.0%} "
              f"{r['ok'] / n:>8.0%} {r['wrong'] / n:>6.0%} {1000 * r['t'] / n:>7.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", help="image used as the classroom background")
    ap.add_argument("--tiles", default="both", choices=["off", "on", "both"])
    ap.add_argument("--limit", type=int, default=0, help="use only N photos (faster)")
    ap.add_argument("--lowres", default="both", choices=["off", "on", "both"],
                    help="also match far faces against a blurred/small gallery")
    a = ap.parse_args()

    if a.background:
        bg = cv2.resize(cv2.imread(a.background), (W, H))
    else:
        bg = np.full((H, W, 3), 200, np.uint8)
    print(f"model {config.MODEL_NAME}, threshold {config.MATCH_THRESHOLD}")
    people = load_people()
    if a.limit:
        idx = np.random.default_rng(0).permutation(len(people))[:a.limit]
        people = [people[i] for i in sorted(idx)]
    print(f"{len(people)} test photos")
    modes = {"off": [False], "on": [True], "both": [False, True]}[a.tiles]
    lr_modes = {"off": [False], "on": [True], "both": [False, True]}[a.lowres]
    for tiles in modes:
        for lr in lr_modes:
            res = run(people, bg, tiles, np.random.default_rng(42), lr)
            report(f"=== tiling {'ON' if tiles else 'OFF'} | low-res gallery "
                   f"{'ON' if lr else 'OFF'} ===", res)


if __name__ == "__main__":
    main()
