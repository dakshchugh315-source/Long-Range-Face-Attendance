"""Face-scan priority tests for the auto-zoom controller (no camera needed).

python tests/test_scan_priority.py

Fake clock + fake camera driver: every camera command is recorded with its
time, so each scenario checks exactly whether / when the camera moved.
Rule under test: while a face is detected or being scanned (k/5) the camera
must not sweep, advance a sweep or zoom out.
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import autozoom as az  # noqa: E402

W, H = 1920, 1080
FRAME = (H, W, 3)
DT = 0.5                                   # one processed frame every 0.5 s (~2 FPS, like the laptop)
results = []


def check(name, ok, info=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   <- {info}" if info and not ok else ""))


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Driver:
    name = "fake"
    max_zoom = 8.0

    def __init__(self, clock, delay=0.0):
        self.clock, self.delay, self.calls = clock, delay, []

    def set_view(self, z, cx, cy):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(("view", round(z, 2), round(cx, 3), round(cy, 3), self.clock()))

    def reset(self):
        self.calls.append(("reset", 1.0, 0.5, 0.5, self.clock()))


def face(cx, cy, px, label="Unknown", votes=0):
    x, y = cx * W, cy * H
    return {"box": (int(x - px / 2), int(y - px / 2), int(x + px / 2), int(y + px / 2)),
            "px": px, "label": label, "votes": votes}


def make():
    clock = Clock()
    drv = Driver(clock)
    log = []
    ctl = az.AutoZoom(drv, log.append, async_cmds=False, clock=clock)
    drv.calls.clear()                        # ignore the initial "go to 1x"
    return ctl, drv, clock, log


def run(ctl, clock, seconds, faces_fn, present=(), absent=3):
    """Advance time frame by frame; returns list of (t, moved, scan_state, motion_state)."""
    trace = []
    end = clock.t + seconds
    while clock.t < end - 1e-9:
        clock.t += DT
        f = faces_fn(clock.t) if callable(faces_fn) else faces_fn
        moved = ctl.step(f, FRAME, set(present), absent, {x["label"] for x in f if x["label"] != "Unknown"})
        trace.append((clock.t, moved, ctl.scan.state, ctl.state))
    return trace


def moves(trace):
    return [t for t, m, _, _ in trace if m]


DAKSH = (0.5, 0.45)


# --------------------------------------------------------------------------- scenarios
def s1_detected_stops_sweep():
    print("\n1. Face detected -> sweep stops")
    ctl, drv, clock, _ = make()
    run(ctl, clock, 31, [])                                       # nobody: sweep starts after 30 s
    check("sweep started while searching (precondition)", ctl.state == "SWEEP")
    pos = ctl.current["x"], ctl.current["y"]
    n_calls = len(drv.calls)
    tr = run(ctl, clock, 10, [face(0.5, 0.5, 140)])               # a recognisable face appears
    check("sweep does NOT advance while a face is detected (10 s, dwell is 4 s)",
          not moves(tr) and len(drv.calls) == n_calls, drv.calls[n_calls:])
    check("state is FACE_DETECTED", tr[-1][2] == az.FACE_DETECTED, tr[-1][2])
    check("camera still on the same sweep position", (ctl.current["x"], ctl.current["y"]) == pos)


def s2_5_scanning_never_sweeps():
    for k in (1, 2, 3, 4):
        print(f"\n{k + 1}. Scanning at {k}/5 -> NO SWEEP")
        ctl, drv, clock, _ = make()
        run(ctl, clock, 26, [])                                   # sweep timer almost due (30 s)
        tr = run(ctl, clock, 3, [face(*DAKSH, 150, "Daksh Chugh", k)])   # crosses the 30 s mark
        check(f"no camera move while Daksh is at {k}/5 (sweep timer expired meanwhile)",
              not moves(tr) and not drv.calls, drv.calls)
        check("state is FACE_SCANNING", tr[-1][2] == az.FACE_SCANNING, tr[-1][2])
        check("label shows the progress", f"Daksh Chugh {k}/5" in ctl.label(), ctl.label())

    print("\n   2-5 continued: scan progressing 1/5 -> 4/5 over 16 s")
    ctl, drv, clock, _ = make()
    run(ctl, clock, 20, [])
    prog = lambda t: [face(*DAKSH, 150, "Daksh Chugh", min(4, 1 + int((t - 1020) // 4)))]  # noqa: E731
    tr = run(ctl, clock, 16, prog)
    check("no move during the whole 1/5..4/5 scan", not moves(tr) and not drv.calls, drv.calls)
    check("all frames in FACE_SCANNING", all(s == az.FACE_SCANNING for _, _, s, _ in tr))

    print("\n   2-5 while zoomed in on a target: scan in progress past the dwell time")
    ctl, drv, clock, _ = make()
    run(ctl, clock, 5, [face(0.3, 0.4, 21)])                      # far face -> auto-zoom target
    check("auto-zoom went to the far face (precondition)", ctl.state == "TARGET")
    n_calls = len(drv.calls)
    tr = run(ctl, clock, 12, [face(0.5, 0.5, 66, "Kanika", 2)])   # now 66 px, scanning 2/5
    check("does NOT return to wide while target is at 2/5 (12 s > 4 s dwell)",
          not moves(tr) and len(drv.calls) == n_calls and ctl.state == "TARGET", drv.calls[n_calls:])


def s6_completes():
    print("\n6. Scanning reaches 5/5 -> completes normally, then searching resumes")
    ctl, drv, clock, log = make()
    run(ctl, clock, 6, [face(*DAKSH, 150, "Daksh Chugh", 3)])
    run(ctl, clock, 2, [face(*DAKSH, 150, "Daksh Chugh", 4)])
    tr = run(ctl, clock, 1, [face(*DAKSH, 150, "Daksh Chugh", 5)], present={"Daksh Chugh"})
    check("state SCAN_COMPLETED right after 5/5", tr[-1][2] == az.SCAN_COMPLETED, tr[-1][2])
    check("hold released immediately (no FACE_LOST wait)", not ctl.scan.hold())
    tr = run(ctl, clock, 40, [face(*DAKSH, 150, "Daksh Chugh", 5)], present={"Daksh Chugh"})
    check("sweep resumes ~30 s after the scan finished", ctl.state == "SWEEP" and moves(tr)
          and 28 <= moves(tr)[0] - (clock.t - 40) <= 33, moves(tr)[:1])

    print("\n   6b. zoomed on a target that reaches 5/5 -> returns to wide early (auto-zoom unchanged)")
    ctl, drv, clock, log = make()
    run(ctl, clock, 5, [face(0.3, 0.4, 21)])
    run(ctl, clock, 2.5, [face(0.5, 0.5, 66, "Kanika", 4)])       # settle (2 s) + 4/5
    tr = run(ctl, clock, 1, [face(0.5, 0.5, 66, "Kanika", 5)], present={"Kanika"})
    check("back to wide as soon as the target is marked", ctl.state == "WIDE" and moves(tr))
    check("zoom move logged with recognised name", log and "Kanika" in log[-1]["recognised"], log[-1:])


def s7_brief_dropout():
    print("\n7. Face briefly disappears -> no sweep")
    ctl, drv, clock, _ = make()
    run(ctl, clock, 28, [])
    run(ctl, clock, 2, [face(*DAKSH, 150, "Daksh Chugh", 2)])
    tr_gap = run(ctl, clock, 2.5, [])                             # missed for 5 frames (detector blink)
    tr_back = run(ctl, clock, 3, [face(*DAKSH, 150, "Daksh Chugh", 3)])
    check("no move during the 2.5 s dropout", not moves(tr_gap) and not drv.calls, drv.calls)
    check("dropout shown as FACE_LOST", all(s == az.FACE_LOST for _, _, s, _ in tr_gap))
    check("scan continues after it reappears", tr_back[-1][2] == az.FACE_SCANNING and not moves(tr_back))


def s8_genuinely_lost():
    print("\n8. Face genuinely lost -> resume searching / sweeping safely")
    ctl, drv, clock, _ = make()
    run(ctl, clock, 10, [face(*DAKSH, 150, "Daksh Chugh", 2)])
    t_lost = clock.t
    tr = run(ctl, clock, 45, [])
    first = [t for t, _, s, _ in tr if s == az.SEARCHING_FOR_FACE][:1]
    check("waits FACE_LOSS_S (4 s) before calling it lost",
          first and abs(first[0] - t_lost - config.AUTOZOOM_FACE_LOSS_S) <= 0.6, first)
    check("no move while FACE_LOST", not [t for t, m, s, _ in tr if m and s == az.FACE_LOST])
    check("sweep resumes afterwards", ctl.state == "SWEEP" and moves(tr), moves(tr))

    print("   8b. lost face + a far face waiting -> auto-zoom target resumes right after loss is confirmed")
    ctl, drv, clock, _ = make()
    far = face(0.15, 0.4, 21)
    run(ctl, clock, 10, [face(*DAKSH, 150, "Daksh Chugh", 2), far])
    check("far face NOT zoomed while Daksh is scanning (would push him out of view)",
          ctl.state == "WIDE" and not drv.calls, drv.calls)
    t_lost = clock.t
    tr = run(ctl, clock, 8, [far])
    check("target move only after the loss is confirmed",
          moves(tr) and moves(tr)[0] - t_lost >= config.AUTOZOOM_FACE_LOSS_S - 0.01, moves(tr))

    print("   8c. stranger who never gets identified does not block the camera forever")
    ctl, drv, clock, _ = make()
    tr = run(ctl, clock, 60, [face(0.5, 0.5, 150)])
    first = moves(tr)[:1]
    check("held while the unknown face is new", not [t for t, m, _, _ in tr[:20] if m])
    check("after it stalls (12 s, no progress) sweeping resumes",
          first and first[0] - (clock.t - 60) >= config.AUTOZOOM_SCAN_STALL_S, first)


def s9_autozoom_unchanged():
    print("\n9. Auto-zoom works exactly as before")
    ctl, drv, clock, _ = make()
    tr = run(ctl, clock, 6, [face(0.18, 0.43, 20), face(0.84, 0.41, 21), face(0.40, 0.39, 22)])
    check("zooms onto a far face after the 4 s wide look", moves(tr) and ctl.state == "TARGET", moves(tr))
    check("smallest face first, 3.0x, centred on it",
          drv.calls and drv.calls[0][:4] == ("view", 3.0, 0.18, 0.43), drv.calls[:1])
    run(ctl, clock, 6.5, [])                                      # dwell, nobody visible there
    check("returns to wide after the 4 s dwell when nothing is being scanned", ctl.state == "WIDE")
    ctl2, drv2, clock2, _ = make()
    run(ctl2, clock2, 5, [face(0.3, 0.4, 21)])
    tr = run(ctl2, clock2, 20, [face(0.5, 0.5, 66)])              # an unidentifiable face at the target
    out = moves(tr)[:1]
    check("unknown face at the target: stays until it stalls (12 s), then returns",
          out and out[0] - (clock2.t - 20) >= config.AUTOZOOM_SCAN_STALL_S and ctl2.state != "TARGET", out)

    print("   9b. zoom-in allowed during a scan only if the scanned face stays in view")
    ctl, drv, clock, _ = make()
    run(ctl, clock, 6, [face(0.5, 0.5, 30, "Yatri", 1)])         # Yatri small (30 px) + scanning
    check("zooms onto the face being scanned (keeps it in view, bigger)",
          ctl.state == "TARGET" and drv.calls and drv.calls[0][2:4] == (0.5, 0.5), drv.calls[:1])


def s10_no_conflicting_commands():
    print("\n10. Rapid updates -> no conflicting camera commands / races")
    clock = time.time
    drv = Driver(clock, delay=0.15)                               # slow phone (Wi-Fi)
    ctl = az.AutoZoom(drv, lambda e: None, async_cmds=True)
    for i in range(6):                                            # 6 moves fired 20 ms apart
        ctl._set(1.5 + i * 0.2, 0.2 + i * 0.1, 0.5)
        time.sleep(0.02)
    ctl.stop()
    time.sleep(1.0)
    views = [c for c in drv.calls if c[0] == "view"]
    check("stale moves are dropped (newest wins)", len(views) <= 2, views)
    check("stop() is the LAST command - nothing moves the camera after it",
          drv.calls and drv.calls[-1][0] == "reset", drv.calls[-2:])
    n = len(drv.calls)
    ctl._set(2.0, 0.5, 0.5)
    ctl.step([face(0.2, 0.4, 20)], FRAME, set(), 3, set())
    time.sleep(0.4)
    check("after stop, step()/moves send nothing", len(drv.calls) == n, drv.calls[n:])

    print("   10b. manual zoom refused while auto-zoom controls the camera")
    import app as appmod
    client = appmod.app.test_client()
    appmod.engine.running, appmod.engine.autozoom = True, object()
    r = client.post("/api/phone/zoom", json={"zoom": 2.0, "source": "http://127.0.0.1:9/video"})
    check("POST /api/phone/zoom -> 409 while auto-zoom is on", r.status_code == 409, r.status_code)
    appmod.engine.running, appmod.engine.autozoom = False, None

    print("   10c. switch on/off requests are applied only by the camera loop")
    eng = appmod.Engine()
    eng.running, eng.autozoom, eng._az_request = True, None, None
    eng.con, eng.source, eng.session = None, "http://127.0.0.1:9/video", "test"
    fake = az.AutoZoom(Driver(time.time), lambda e: None, async_cmds=False)
    eng._make_autozoom = lambda *a: fake
    eng.set_autozoom(True)
    check("'on' only queues the controller (not live yet)", eng.autozoom is None and eng._az_request is fake)
    eng.set_autozoom(False)
    check("'off' before the loop applied it discards it, camera reset once",
          eng._az_request is None and fake.stopped)
    eng.running = False


def s11_sweep_spacing_and_backoff():
    print("\n11. Sweep is not continuous: gap counted from the END, backs off when it finds nobody")
    ctl, drv, clock, _ = make()
    tr = run(ctl, clock, 400, [])                                 # empty room, 3 students absent
    starts = [t for (t, m, _, st), prev in zip(tr, [None] + tr[:-1])
              if m and st == "SWEEP" and (prev is None or prev[3] != "SWEEP")]
    ends = [prev[0] for (t, m, _, st), prev in zip(tr[1:], tr[:-1]) if prev[3] == "SWEEP" and st == "WIDE"]
    gaps = [round(s - e) for s, e in zip(starts[1:], ends)]
    check("sweeps happen", len(starts) >= 3, starts)
    check("gap after each sweep >= 30 s and grows (30 -> 60 -> 120 ...)",
          len(gaps) >= 2 and gaps[0] >= 29 and all(b > a * 1.7 for a, b in zip(gaps, gaps[1:])), gaps)
    busy = sum(1 for _, _, _, st in tr if st != "WIDE") * DT
    check("camera is NOT moving most of the time (< 40 % of 400 s)", busy < 160, f"{busy:.0f} s moving")

    print("   11b. a sweep that finds someone resets the back-off")
    ctl.fruitless_sweeps = 3
    ctl.sweep_marked_at_start = 0
    ctl.state, ctl.sweep_queue = "SWEEP", []
    ctl.current = {"action": "sweep", "zoom": 2.5, "x": 0.5, "y": 0.5, "face_px": None,
                   "reason": "sweep", "seen": set(), "start": clock.t}
    clock.t += 10
    ctl.step([], FRAME, {"Kanika"}, 2, set())
    check("found someone -> next sweep after 30 s again", ctl.sweep_interval() == config.AUTOZOOM_SWEEP_EVERY_S,
          ctl.sweep_interval())


def s12_adaptive_votes():
    print("\n12. 3 checks for a strong match, 5 for a weak one")
    import attend
    v = attend.Voter()
    v.add("Daksh Chugh", 0.71, 200, None, (0, 0, 1, 1))
    v.add("Himanshu Rawat", 0.44, 54, None, (0, 0, 1, 1))
    check(f"strong match (0.71 >= {config.STRONG_MATCH}) needs {config.VOTES_STRONG}",
          attend.votes_needed(v, "Daksh Chugh") == config.VOTES_STRONG)
    check(f"weak match (0.44) still needs {config.VOTES_TO_MARK}",
          attend.votes_needed(v, "Himanshu Rawat") == config.VOTES_TO_MARK)
    ctl, drv, clock, _ = make()
    f = face(*DAKSH, 150, "Daksh Chugh", 2)
    f["need"] = 3
    run(ctl, clock, 2, [f])
    check("progress shown as k/3 for a strong match", "Daksh Chugh 2/3" in ctl.label(), ctl.label())


def s13_no_duplicate_boxes():
    print("\n13. One box per face (no duplicate from tile borders)")
    import cv2
    import numpy as np
    import face_engine as fe
    src_path = sorted((config.DATASET_DIR / "Daksh Chugh").glob("*.jpg"))[0]
    src = cv2.imread(str(src_path))
    f0 = max(fe.detect(src, tiles=False), key=fe.face_width)
    dup = n = 0
    for size in (110, 180, 260):
        for x in (491, 639, 972, 1120, 1268):                      # right on the tile borders
            s = size / fe.face_width(f0)
            patch = cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            bx, by = int((f0.bbox[0] + f0.bbox[2]) / 2 * s), int((f0.bbox[1] + f0.bbox[3]) / 2 * s)
            canvas = np.full((1080, 1920, 3), 200, np.uint8)
            ox, oy = x - bx, 500 - by
            ph, pw = patch.shape[:2]
            sx, sy = max(0, -ox), max(0, -oy)
            y0, y1, x0, x1 = max(0, oy), min(1080, oy + ph), max(0, ox), min(1920, ox + pw)
            canvas[y0:y1, x0:x1] = patch[sy:sy + y1 - y0, sx:sx + x1 - x0]
            n += 1
            dup += len(fe.detect(canvas, tiles=True)) != 1
    check(f"exactly one box in all {n} border cases (was 17/124 duplicates before)", dup == 0, f"{dup}/{n}")


def main():
    s1_detected_stops_sweep()
    s2_5_scanning_never_sweeps()
    s6_completes()
    s7_brief_dropout()
    s8_genuinely_lost()
    s9_autozoom_unchanged()
    s10_no_conflicting_commands()
    s11_sweep_spacing_and_backoff()
    s12_adaptive_votes()
    s13_no_duplicate_boxes()
    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
