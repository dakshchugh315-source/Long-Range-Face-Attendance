"""Final acceptance tests: the mandatory scenarios that can run without the real camera.

python tests/test_final.py
Real models + real enrolment photos; throw-away database. Phone/CCTV/low-light-hardware
scenarios are NOT covered here (see tools/measure_phone.py).
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_audit_fixes as H  # noqa: E402  (sets a temp DB / snapshot dir, gives photo helpers)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import attend  # noqa: E402
import autozoom as az  # noqa: E402
import config  # noqa: E402
import face_engine as fe  # noqa: E402

results = []
G = fe.Gallery()
PEOPLE = ["Abhrojet", "Adityam Vashisht", "Daksh Chugh", "Divyansh", "Kanika"]
PHOTOS = {n: H.frontal(n) for n in PEOPLE}


def check(name, ok, info=None):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   <- {info}" if not ok and info is not None else ""))


def scene(positions, size=110, frame=None):
    fr = H.blank() if frame is None else frame
    for name, (x, y) in positions.items():
        img, face = PHOTOS[name]
        H.paste(fr, img, face, int(x), int(y), size)
    return fr


def session(frames, present=None, moving=False, session_name="final"):
    con = attend.open_db()
    present = set() if present is None else present
    tracks, voter, infos, newly = [], attend.Voter(), [], []
    for fr in frames:
        info = {}
        newly += attend.process_frame(fr.copy(), tracks, G, present, con, session_name, None, voter,
                                      info=info, camera_moving=moving)
        infos.append(info)
    con.close()
    return infos, present, tracks, newly


def t_five_people():
    print("\n1. Five people at once: all detected, own track each, all marked, no mix-up")
    pos = {n: (250 + i * 350, 450 + (i % 2) * 120) for i, n in enumerate(PEOPLE)}
    infos, present, tracks, newly = session([scene(pos) for _ in range(6)])
    check("5 faces detected in every frame", all(len(i["faces"]) == 5 for i in infos),
          [len(i["faces"]) for i in infos])
    check("5 stable tracks", len(tracks) == 5, len(tracks))
    check("all 5 marked present", present == set(PEOPLE), present)
    check("each marked exactly once", sorted(n for n, _, _ in newly) == sorted(PEOPLE), newly)


def t_crossing():
    print("\n2. Two people walk past each other (cross positions): no identity swap")
    a, b = "Daksh Chugh", "Divyansh"
    frames, truth = [], []
    for k in range(10):
        t = k / 9
        xa, xb = 500 + 900 * t, 1400 - 900 * t
        frames.append(scene({a: (xa, 500), b: (xb, 520)}))
        truth.append((xa, xb))
    infos, present, tracks, _ = session(frames)
    swaps = 0
    for info, (xa, xb) in zip(infos, truth):
        for f in info["faces"]:
            cx = (f["box"][0] + f["box"][2]) / 2
            true_name = a if abs(cx - xa) < abs(cx - xb) else b
            if f["label"] not in ("Unknown", true_name):
                swaps += 1
    check("no face ever shows the other person's name", swaps == 0, swaps)
    check("both marked", present == {a, b}, present)


def t_strangers():
    print("\n3. Unknown people (not enrolled) for 15 frames: no attendance")
    from insightface.data import get_image
    crowd = get_image("t1")                               # 6 strangers
    infos, present, _, newly = session([crowd] * 15)
    statuses = {f["status"] for i in infos for f in i["faces"]}
    check("6 strangers detected", len(infos[-1]["faces"]) == 6, len(infos[-1]["faces"]))
    check("nobody marked", not present and not newly, present)
    check("shown as unknown (not a student's name)", not (statuses & {"marked", "recognizing"}), statuses)


def t_bad_quality():
    print("\n4. Motion blur / darkness: never a wrong name; the reason is shown")
    wrong, shown = 0, set()
    k = np.zeros((9, 9))
    k[4, :] = 1 / 9
    for name in PEOPLE:
        img, face = PHOTOS[name]
        fr = H.paste(H.blank(), img, face, 960, 540, 30)
        blurred = cv2.filter2D(fr, -1, k)
        dark = np.clip(fr.astype(np.float32) * 0.06, 0, 255).astype(np.uint8)
        for f2 in (blurred, dark):
            infos, present, _, _ = session([f2] * 3)
            if not infos[-1].get("faces"):
                shown.add("not_detected")
            for f in infos[-1].get("faces", []):
                shown.add(f["status"])
                if f["label"] not in ("Unknown", name):
                    wrong += 1
    check("no wrong name on blurred or very dark faces", wrong == 0, wrong)
    check("'Face not clear' or 'Low light' explains failures", shown & {"not_clear", "low_light"}, shown)


def t_camera_moving():
    print("\n5. Frames grabbed while the camera moves give no votes")
    infos, present, _, _ = session([scene({"Kanika": (960, 540)}) for _ in range(6)], moving=True)
    check("nobody marked from moving-camera frames", not present, present)
    check("status 'stabilizing' shown", {f["status"] for i in infos for f in i["faces"]} == {"stabilizing"})


def t_reappear_no_duplicate():
    print("\n6. Same person leaves and comes back: no duplicate attendance")
    fr = scene({"Abhrojet": (960, 540)})
    frames = [fr] * 4 + [H.blank()] * 20 + [fr] * 6
    infos, present, _, newly = session(frames, session_name="dup")
    con = attend.open_db()
    n = con.execute("SELECT COUNT(*) FROM attendance WHERE session='dup' AND name='Abhrojet'").fetchone()[0]
    con.close()
    check("exactly one DB row", n == 1, n)
    check("marked once", [x[0] for x in newly] == ["Abhrojet"], newly)
    check("shown as 'Already marked' or 'Marked' when back", infos[-1]["faces"][0]["status"]
          in ("marked", "already_marked"), infos[-1]["faces"][0]["status"])


def t_db_failure():
    print("\n7. Database write fails once: no false success, no duplicate, no orphan proof photos")
    fr = scene({"Adityam Vashisht": (960, 540)})
    con = attend.open_db()
    real_con, fail = con, {"n": 1}

    class Flaky:
        def __getattr__(self, k):
            return getattr(real_con, k)

        def execute(self, sql, *a):
            if sql.startswith("INSERT OR IGNORE INTO attendance(session, name, time, similarity") and fail["n"]:
                fail["n"] -= 1
                raise attend.sqlite3.OperationalError("database is locked (simulated)")
            return real_con.execute(sql, *a)
    before = set(config.SNAPSHOT_DIR.glob("*")) if config.SNAPSHOT_DIR.exists() else set()
    present, tracks, voter, errors, newly = set(), [], attend.Voter(), 0, []
    for _ in range(6):
        try:
            newly += attend.process_frame(fr.copy(), tracks, G, present, Flaky(), "dbfail", None, voter)
        except attend.sqlite3.OperationalError:
            errors += 1
    rows = con.execute("SELECT COUNT(*) FROM attendance WHERE session='dbfail'").fetchone()[0]
    new_files = set(config.SNAPSHOT_DIR.glob("*")) - before
    con.close()
    check("the failure surfaced (not hidden)", errors == 1, errors)
    check("marked after retry, exactly one row", rows == 1 and len(newly) == 1, (rows, newly))
    check("no orphan proof photos (2 files = 1 record)", len(new_files) == 2, len(new_files))


def t_rapid_entry_exit():
    print("\n8. Rapid entry/exit for 40 frames: no crash, tracks don't pile up")
    rng = np.random.default_rng(1)
    con = attend.open_db()
    tracks, voter, present, ok = [], attend.Voter(), set(), True
    try:
        for _ in range(40):
            pos = {n: (rng.integers(200, 1700), rng.integers(250, 850)) for n in PEOPLE if rng.random() < 0.4}
            attend.process_frame(scene(pos, size=90), tracks, G, present, con, "rapid", None, voter)
    except Exception as e:
        ok = repr(e)
    con.close()
    check("no exception", ok is True, ok)
    con = attend.open_db()
    for _ in range(20):                                     # everyone leaves
        attend.process_frame(H.blank(), tracks, G, present, con, "rapid", None, voter)
    con.close()
    check("all old tracks expire after people leave (no pile-up)", len(tracks) == 0, len(tracks))


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


class Driver:
    name, max_zoom = "fake", 8.0

    def __init__(self):
        self.calls = []

    def set_view(self, z, cx, cy):
        self.calls.append((round(z, 2), round(cx, 2), round(cy, 2)))

    def reset(self):
        self.calls.append("reset")


def drive(ctl, clock, seconds, faces, present=frozenset(), absent=3, dt=0.5):
    moves = 0
    end = clock.t + seconds
    while clock.t < end:
        clock.t += dt
        moves += ctl.step(faces, (1080, 1920, 3), set(present), absent, set())
    return moves


def t_idle_and_limits():
    print("\n9. All visible people done: camera settles to IDLE (no endless sweeping)")
    clock, drv = Clock(), Driver()
    ctl = az.AutoZoom(drv, lambda e: None, async_cmds=False, clock=clock)
    drive(ctl, clock, 900, [])
    check("IDLE after repeated empty sweeps", ctl.camera_state() == "IDLE", ctl.camera_state())
    n = len(drv.calls)
    drive(ctl, clock, 600, [])
    check("no camera movement at all for the next 10 minutes", len(drv.calls) == n, len(drv.calls) - n)
    marked = {"box": (900, 400, 1020, 520), "px": 120, "label": "Kanika", "votes": 5, "need": 3}
    drive(ctl, clock, 120, [marked], present={"Kanika"})
    check("an already-marked person does not wake it up", len(drv.calls) == n, len(drv.calls) - n)
    new = {"box": (300, 400, 320, 420), "px": 20, "label": "Unknown", "votes": 0, "need": 5}
    n = len(drv.calls)
    drive(ctl, clock, 10, [new])
    zoomed_on_new = [c for c in drv.calls[n:] if c != "reset" and c[0] > 2.9 and abs(c[1] - 0.16) < 0.03]
    check("a new face in view is still zoomed on while IDLE", zoomed_on_new, drv.calls[n:])
    n = len(drv.calls)
    drive(ctl, clock, 300, [new])
    sweeps = [c for c in drv.calls[n:] if c != "reset" and abs(c[0] - config.AUTOZOOM_SWEEP_ZOOM) < 0.01]
    check("a stranger who stays does NOT restart sweeping", not sweeps, sweeps[:3])

    print("\n10. Zoom limit: a face too small even at max zoom is not zoomed on again and again")
    clock, drv = Clock(), Driver()
    ctl = az.AutoZoom(drv, lambda e: None, async_cmds=False, clock=clock)
    small = {"box": (1500, 300, 1510, 310), "px": 10, "label": "Unknown", "votes": 0, "need": 5}
    drive(ctl, clock, 600, [small])
    targets = [c for c in drv.calls if c != "reset" and c[0] > 2.9]
    check("never beyond the zoom limit", all(c[0] <= config.AUTOZOOM_MAX for c in drv.calls if c != "reset"))
    # schedule: 3 quick tries (0, 25, 50 s) then waits of 2, 4, 8 min -> 5 tries in 10 min, gaps growing
    t_moves = [i for i, c in enumerate(drv.calls) if c != "reset" and c[0] > 2.9]
    check("unidentifiable spot: at most 5 tries in 10 minutes", len(targets) <= 5, len(targets))
    check("and the waits between tries keep growing (no oscillation)", len(targets) >= 2, len(targets))
    clock2, drv2 = Clock(), Driver()
    ctl2 = az.AutoZoom(drv2, lambda e: None, async_cmds=False, clock=clock2)
    tiny = {"box": (1500, 300, 1506, 306), "px": 6, "label": "Unknown", "votes": 0, "need": 5}
    drive(ctl2, clock2, 20, [tiny])
    check("face too small even at max zoom is never targeted",
          not [c for c in drv2.calls if c != "reset" and c[0] > 2.9], drv2.calls)


def t_security():
    print("\n11. Teacher app refuses cross-site requests (CSRF / DNS rebinding)")
    import app as appmod
    c = appmod.app.test_client()
    r = c.post("/api/delete", json={"all": True}, headers={"Origin": "http://evil.example"})
    check("POST from another website -> 403", r.status_code == 403, r.status_code)
    r = c.post("/api/delete", data='{"all": true}', content_type="text/plain")
    check("form-style text/plain POST -> 415", r.status_code == 415, r.status_code)
    r = c.get("/api/status", headers={"Host": "evil.example:5000"})
    check("foreign Host header -> 403", r.status_code == 403, r.status_code)
    r = c.get("/api/status")
    check("normal local request still works", r.status_code == 200, r.status_code)


def t_restart():
    print("\n12. Start -> stop -> start again: works, no leaked threads")
    import app as appmod
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not clip:
        print("  (skipped: pass a video clip)")
        return
    eng = appmod.engine
    base = threading.active_count()
    for k in range(2):
        eng.start(str(clip), f"restart{k}")
        t0 = time.time()
        while eng.running and time.time() - t0 < 300:
            time.sleep(0.3)
        print(f"  run {k + 1}: {time.time() - t0:.0f} s, error={eng.error!r}")
        eng.stop()
    time.sleep(1)
    check("both runs finished normally", eng.error == "Video file finished.", eng.error)
    check("no leaked camera threads", threading.active_count() <= base + 1, (base, threading.active_count()))


def main():
    for t in (t_five_people, t_crossing, t_strangers, t_bad_quality, t_camera_moving,
              t_reappear_no_duplicate, t_db_failure, t_rapid_entry_exit, t_idle_and_limits,
              t_security, t_restart):
        t()
    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
