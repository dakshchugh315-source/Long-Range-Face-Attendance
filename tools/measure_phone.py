"""Measure zoom quality and low-light options on the REAL phone (IP Webcam).

python tools/measure_phone.py http://PHONE_IP:8080            # zoom + low light
python tools/measure_phone.py http://PHONE_IP:8080 --only zoom
python tools/measure_phone.py http://PHONE_IP:8080 --only light

Put the phone on a stand, one enrolled student sitting still 3-8 m away (facing it).
For each setting it grabs several frames and measures what matters:
  face px, sharpness of the face (Laplacian variance), recognition match score.
ZOOM : phone zoom (crops the 64 MP sensor BEFORE the 1080p stream) versus software zoom
       (crop the 1x frame + LANCZOS upscale) at 1x..5x -> which is better, and where real
       detail stops (match score no longer improves).
LIGHT: phone settings (normal, night vision, higher ISO/longer exposure) and the software
       enhancer (CLAHE+gamma) on the same frames -> keep only what really raises the match.
All phone settings are restored at the end. Results: tools/measure_result.json + .jpg
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import autozoom as az  # noqa: E402
import face_engine as fe  # noqa: E402

FRAMES = 4
SETTLE = 3.0


def status(base):
    return json.loads(az.phone_get(base, "/status.json?show_avail=1").decode("utf-8-sig"))


def setv(base, key, val):
    az.phone_get(base, f"/settings/{key}?set={val}")


def shots(base, n=FRAMES):
    out = []
    for _ in range(n):
        out.append(az.phone_shot(base))
        time.sleep(0.3)
    return out


def measure(frames, gallery, enhance=False, near=None):
    """Mean face px, sharpness and best match over frames (largest / nearest face)."""
    px, sharp, sim, names = [], [], [], []
    for fr in frames:
        fs = fe.detect(fr)
        if not fs:
            continue
        if near is not None:
            h, w = fr.shape[:2]
            f = min(fs, key=lambda f: np.hypot((f.bbox[0] + f.bbox[2]) / 2 / w - near[0],
                                               (f.bbox[1] + f.bbox[3]) / 2 / h - near[1]))
        else:
            f = max(fs, key=fe.face_width)
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        crop = cv2.cvtColor(fr[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (112, 112))               # same scale for every setting
        sharp.append(float(cv2.Laplacian(crop, cv2.CV_64F).var()))
        s = gallery.centroids @ fe.embed(fr, f, enhance=enhance)
        sim.append(float(s.max()))
        names.append(gallery.names[int(s.argmax())])
        px.append(fe.face_width(f))
    if not px:
        return {"found": 0}
    return {"found": len(px), "px": round(float(np.mean(px))), "sharp": round(float(np.mean(sharp)), 1),
            "match": round(float(np.mean(sim)), 3), "who": max(set(names), key=names.count)}


def zoom_test(base, gallery, report):
    print("\n=== ZOOM: phone zoom vs software crop+LANCZOS ===")
    steps = az.phone_zoom_steps(base)
    setv(base, "zoom", 100), setv(base, "crop_x", 50), setv(base, "crop_y", 50)
    time.sleep(SETTLE)
    wide = shots(base)
    fs = fe.detect(wide[0])
    if not fs:
        raise SystemExit("No face visible at 1x - sit facing the phone and retry.")
    f = max(fs, key=fe.face_width)
    h, w = wide[0].shape[:2]
    cx, cy = (f.bbox[0] + f.bbox[2]) / 2 / w, (f.bbox[1] + f.bbox[3]) / 2 / h
    print(f"target face at ({cx:.2f},{cy:.2f}), {fe.face_width(f)} px at 1x")
    rows = []
    for z in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        zs = az.snap_zoom(z, steps) / 100
        setv(base, "zoom", int(zs * 100))
        setv(base, "crop_x", az.crop_for(cx, zs, "free_range"))
        setv(base, "crop_y", az.crop_for(cy, zs, "free_range"))
        time.sleep(SETTLE)
        phone = measure(shots(base), gallery)
        soft_frames = []                                    # software zoom from the 1x frames
        for fr in wide:
            ww, wh = w / zs, h / zs
            x0 = int(min(max(cx * w - ww / 2, 0), w - ww))
            y0 = int(min(max(cy * h - wh / 2, 0), h - wh))
            soft_frames.append(cv2.resize(fr[y0:y0 + int(wh), x0:x0 + int(ww)], (w, h),
                                          interpolation=cv2.INTER_LANCZOS4))
        soft = measure(soft_frames, gallery)
        rows.append({"zoom": zs, "phone": phone, "software": soft})
        print(f"{zs:4.2f}x | phone: {phone} | software: {soft}")
    setv(base, "zoom", 100), setv(base, "crop_x", 50), setv(base, "crop_y", 50)
    best = max((r for r in rows if r["phone"].get("found")), key=lambda r: r["phone"]["match"], default=None)
    limit = None
    for a, b in zip(rows, rows[1:]):                        # where the match stops improving
        if a["phone"].get("found") and b["phone"].get("found") and b["phone"]["match"] < a["phone"]["match"] + 0.02:
            limit = a["zoom"]
            break
    phone_wins = sum(1 for r in rows[1:] if r["phone"].get("match", 0) > r["software"].get("match", 0))
    report["zoom"] = {"rows": rows, "best_zoom": best and best["zoom"], "detail_limit": limit,
                      "phone_better_in": f"{phone_wins}/{len(rows) - 1}"}
    print(f"-> phone zoom better than software in {phone_wins}/{len(rows) - 1} levels; "
          f"match stops improving after ~{limit}x (suggest AUTOZOOM_MAX = {limit})")


def light_test(base, gallery, report):
    print("\n=== LOW LIGHT: phone settings vs software enhancer ===")
    cur = status(base)["curvals"]
    keep = {k: cur.get(k) for k in ("night_vision", "night_vision_gain", "manual_sensor", "iso",
                                    "exposure_ns", "scenemode")}
    trials = [("normal", {}),
              ("face_priority", {"scenemode": "face_priority"}),
              ("night_vision x2", {"night_vision": "on", "night_vision_gain": "2.0"}),
              ("night_vision x4", {"night_vision": "on", "night_vision_gain": "4.0"})]
    rows = []
    try:
        for name, settings in trials:
            for k, v in keep.items():                       # start each trial from the original
                if v is not None:
                    setv(base, k, v)
            for k, v in settings.items():
                setv(base, k, v)
            time.sleep(SETTLE)
            frames = shots(base)
            bright = round(float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames])), 1)
            plain, enh = measure(frames, gallery, False), measure(frames, gallery, True)
            rows.append({"setting": name, "brightness": bright, "plain": plain, "software_enhance": enh})
            print(f"{name:16} brightness {bright:5} | plain: {plain} | +enhance: {enh}")
    finally:
        for k, v in keep.items():
            if v is not None:
                setv(base, k, v)
    ok = [r for r in rows if r["plain"].get("found")]
    best = max(ok, key=lambda r: max(r["plain"]["match"], r["software_enhance"].get("match", 0)), default=None)
    gain = [r["software_enhance"].get("match", 0) - r["plain"]["match"] for r in ok]
    report["light"] = {"rows": rows, "best_setting": best and best["setting"],
                       "software_enhance_avg_gain": round(float(np.mean(gain)), 3) if gain else None}
    print(f"-> best setting: {best and best['setting']}; software enhancer average gain "
          f"{report['light']['software_enhance_avg_gain']} (turn ENHANCE_DARK_FACES on only if clearly > 0.02)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="http://PHONE_IP:8080")
    ap.add_argument("--only", choices=["zoom", "light"])
    a = ap.parse_args()
    base = a.base.rstrip("/")
    if not az.is_ip_webcam(base):
        raise SystemExit(f"{base} is not reachable as IP Webcam (same Wi-Fi? server started?)")
    gallery = fe.Gallery()
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "phone": base}
    if a.only != "light":
        zoom_test(base, gallery, report)
    if a.only != "zoom":
        light_test(base, gallery, report)
    out = Path(__file__).with_name("measure_result.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
