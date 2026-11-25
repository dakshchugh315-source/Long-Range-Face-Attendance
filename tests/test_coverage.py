"""Classroom coverage logic (no models needed): regions, unresolved faces, verdicts, sweep order.

python tests/test_coverage.py
"""
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="fa_cov_"))
config.SNAPSHOT_DIR = TMP / "snaps"
import classroom  # noqa: E402

results = []
W, H = 1920, 1080
ROSTER = ["Ana", "Ben", "Cy", "Dee"]


def check(name, ok, info=None):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   <- {info}" if not ok and info is not None else ""))


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def face(x, y, px, label="Unknown", track=1, tried=True, emb=None):
    return {"box": (int(x - px / 2), int(y - px / 2), int(x + px / 2), int(y + px / 2)), "px": px,
            "label": label, "track": track, "tried": tried, "emb": emb, "status": "unknown"}


def db():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE attendance (session TEXT, name TEXT)")
    classroom.init(con)
    return con


def feed(cov, faces, n=1, view=None, present=(), t0=[1000.0]):
    for _ in range(n):
        t0[0] += 1.0
        frame = np.zeros((H, W, 3), np.uint8)
        cov.update({"faces": faces, "clean": frame}, frame.shape, view, False, set(present), now=t0[0])


def main():
    print("\n1. Regions: mapping, also when zoomed in")
    cov = classroom.Coverage("s", ROSTER)
    check("top-left of the wide view = Back-left", cov.region_of(*cov.to_wide(0.1, 0.1, None)) == 0)
    check("bottom-right of the wide view = Front-right", cov.region_of(*cov.to_wide(0.9, 0.9, None)) == 5)
    wx, wy = cov.to_wide(0.5, 0.5, (2.5, 0.833, 0.25))        # centre of a 2.5x view on the back-right
    check("centre of a 2.5x zoom on the back-right maps to Back-right", cov.region_of(wx, wy) == 2, (wx, wy))
    vis = [r["id"] for r in cov.visible_regions((2.5, 0.167, 0.25))]
    check("a 2.5x view only 'sees' its own region", vis == [0], vis)

    print("\n2. Unresolved faces: one entry per person, auto-resolve, teacher decisions kept")
    cov = classroom.Coverage("s", ROSTER)
    stranger = unit(np.r_[1, 0.2, np.zeros(510)])
    feed(cov, [face(300, 300, 60, track=1, emb=stranger)], n=classroom.UNRESOLVED_AFTER)
    check("listed after UNRESOLVED_AFTER failed attempts", len(cov.unresolved) == 1)
    feed(cov, [face(320, 310, 60, track=2, emb=unit(stranger + 0.05))], n=classroom.UNRESOLVED_AFTER)
    check("same person with a new track ID -> still ONE entry", len(cov.unresolved) == 1, len(cov.unresolved))
    other = unit(np.r_[0, 0, 1, np.zeros(509)])
    feed(cov, [face(900, 800, 60, track=3, emb=other)], n=classroom.UNRESOLVED_AFTER)
    check("a different person -> a second entry", len(cov.unresolved) == 2)
    feed(cov, [face(900, 800, 60, label="Ben", track=3)], present={"Ben"})
    check("recognised later -> entry resolves itself", cov.unresolved[1]["status"] == "resolved"
          and cov.unresolved[1]["resolved_as"] == "Ben")
    feed(cov, [face(300, 300, 60, track=4, tried=True, emb=stranger)], n=3)
    check("few tries are not listed (no noise)", all(u["attempts"] >= classroom.UNRESOLVED_AFTER for u in cov.unresolved))
    con = db()
    cov.flush(con)
    con.execute("UPDATE unresolved SET status='ignored' WHERE uid=1")
    con.commit()
    feed(cov, [face(300, 300, 60, track=1, emb=stranger)], n=2)
    cov.flush(con)
    st = con.execute("SELECT status FROM unresolved WHERE uid=1").fetchone()[0]
    check("teacher's 'ignored' is never overwritten by the camera", st == "ignored", st)
    check("crop photo saved for the teacher", (config.SNAPSHOT_DIR).exists() and any(config.SNAPSHOT_DIR.glob("unresolved_*")))

    print("\n3. Honest verdicts")
    con = db()
    cov = classroom.Coverage("v", ROSTER)
    rep = classroom.report(con, "v", ROSTER)
    check("nothing scanned -> not started", rep["verdict"] == "Scanning not started")
    feed(cov, [], n=5)
    cov.flush(con)
    for n in ROSTER[:3]:
        con.execute("INSERT INTO attendance VALUES ('v', ?)", (n,))
    rep = classroom.report(con, "v", ROSTER)
    check("3 of 4 confirmed -> '3 Confirmed Present / 1 Not Yet Verified'",
          rep["headline"] == "3 Confirmed Present / 1 Not Yet Verified", rep["headline"])
    check("...and NOT 'completed'", rep["verdict"] != "Attendance Verification Completed", rep["verdict"])
    con.execute("INSERT INTO attendance VALUES ('v', 'Dee')")
    rep = classroom.report(con, "v", ROSTER)
    check("all 4 confirmed -> Attendance Verification Completed", rep["verdict"] == "Attendance Verification Completed")

    print("\n4. Region status + sweep order")
    cov = classroom.Coverage("r", ROSTER)
    feed(cov, [face(1700, 300, 14, track=9, tried=False)], n=5)          # tiny face, back-right
    st = {r["id"]: cov.region_status(r, set()) for r in cov.regions}
    check("tiny faces never zoomed on -> that region needs review", st[2] == "needs_review", st)
    check("other regions completed after 5 s of watching", st[0] == "completed" and st[5] == "completed", st)
    todo = cov.regions_to_visit()
    check("sweep visits ONLY the region that needs it", todo == [(2.5 / 3, 0.25)], todo)
    for _ in range(5):
        cov.update({"faces": [face(960, 540, 60, label="Ana", track=10)]}, (H, W, 3), (2.5, 0.833, 0.25), False,
                   {"Ana"}, now=2000 + _)
    st = {r["id"]: cov.region_status(r, set()) for r in cov.regions}
    check("after a zoomed look at it, the region is completed", st[2] == "completed", st)
    check("nothing left to sweep", cov.regions_to_visit() == [])
    st0 = classroom.classroom_state("SEARCHING", {"not_verified": ["x"], "regions": []}, True)
    check("states are classroom-level names", st0 == "NEXT_REGION_SELECTION")

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
