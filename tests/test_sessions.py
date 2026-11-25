"""Sessions: create / edit / delete, schedule auto-start + auto-stop, late, bulk manual marking.

python tests/test_sessions.py path/to/clip_of_an_enrolled_student.mov
"""
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="fa_sess_"))
config.DB_FILE = TMP / "att.db"
config.SNAPSHOT_DIR = TMP / "snaps"
import notify  # noqa: E402

notify.SECRETS = TMP / "secrets.json"
notify.TelegramBot._call = lambda self, method, timeout=35, **p: [] if method == "getUpdates" else {}
import app as appmod  # noqa: E402
import sessions_db  # noqa: E402

results = []
C = appmod.app.test_client()
FMT = sessions_db.FMT


def check(name, ok, info=None):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   <- {info}" if not ok and info is not None else ""))


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else None
    now = datetime.now()
    print("\n1. Create / validate / edit / delete")
    r = C.post("/api/sessions", json={"name": "DL Lecture 1", "course": "Deep Learning",
                                      "start_at": (now + timedelta(hours=1)).strftime(FMT),
                                      "end_at": (now + timedelta(hours=2)).strftime(FMT)})
    check("create scheduled session", r.status_code == 200, r.get_json())
    sid = r.get_json()["session"]["id"]
    check("duplicate name refused", C.post("/api/sessions", json={"name": "DL Lecture 1"}).status_code == 400)
    bad = C.post("/api/sessions", json={"name": "X", "start_at": now.strftime(FMT),
                                        "end_at": (now - timedelta(hours=1)).strftime(FMT)})
    check("end before start refused", bad.status_code == 400)
    check("auto-start without camera refused", C.post("/api/sessions", json={
        "name": "Y", "auto_start": True, "start_at": now.strftime(FMT),
        "end_at": (now + timedelta(hours=1)).strftime(FMT)}).status_code == 400)
    r = C.post(f"/api/sessions/{sid}", json={"course": "DL (Sec A)", "late_after_min": 5})
    check("edit", r.status_code == 200 and r.get_json()["session"]["late_after_min"] == 5)
    lst = C.get("/api/sessions").get_json()
    check("listed as upcoming", any(s["name"] == "DL Lecture 1" and s["status"] == "scheduled" for s in lst))
    tmp = C.post("/api/sessions", json={"name": "To delete"}).get_json()["session"]["id"]
    check("delete a scheduled session", C.delete(f"/api/sessions/{tmp}").status_code == 200
          and not any(s["name"] == "To delete" for s in C.get("/api/sessions").get_json()))

    print("\n2. Bulk manual marking + late")
    late_sess = C.post("/api/sessions", json={"name": "Started 1h ago", "late_after_min": 10,
                                              "start_at": (now - timedelta(hours=1)).strftime(FMT),
                                              "end_at": (now + timedelta(hours=1)).strftime(FMT)}).get_json()
    names = appmod.enrolled_names()[:3]
    r = C.post("/api/manual/bulk", json={"session": "Started 1h ago", "names": names + ["Not A Student"],
                                         "action": "add"})
    check("3 students marked in one go (unknown name ignored)", r.get_json()["changed"] == 3, r.get_json())
    st = C.get("/api/status?session=Started%201h%20ago").get_json()
    check("marked 50 min after start -> Late", all(p["late"] for p in st["present"]), st["present"][:1])
    r = C.post("/api/manual/bulk", json={"session": "Started 1h ago", "names": names[:1], "action": "remove"})
    check("bulk remove", r.get_json()["changed"] == 1)
    prof = C.get("/api/student/" + names[1]).get_json()
    check("student profile: history + %", prof["attended"] == 1 and prof["total"] >= 1
          and prof["history"][0]["status"] in ("present", "absent"), prof)

    if not clip:
        print("\n3. (skipped: pass a video clip for start / schedule tests)")
    else:
        print("\n3. Start recording a session from the Sessions tab")
        r = C.post(f"/api/sessions/{sid}/start", json={"source": clip})
        check("starts", r.status_code == 200, r.get_json())
        live = [s for s in C.get("/api/sessions").get_json() if s["name"] == "DL Lecture 1"][0]
        check("status LIVE while recording", live["status"] == "live", live["status"])
        t0 = time.time()
        while appmod.engine.running and time.time() - t0 < 240:
            time.sleep(0.5)
        appmod.engine.stop()
        done = [s for s in C.get("/api/sessions").get_json() if s["name"] == "DL Lecture 1"][0]
        check("status DONE after it stops", done["status"] == "done", done["status"])

        print("\n4. Scheduler: auto-start at start time, auto-stop at end time")
        start = datetime.now().replace(second=0, microsecond=0)
        r = C.post("/api/sessions", json={"name": "Auto class", "camera": clip, "auto_start": True,
                                          "start_at": start.strftime(FMT),
                                          "end_at": (start + timedelta(minutes=30)).strftime(FMT)})
        check("auto-start session created", r.status_code == 200, r.get_json())
        appmod.scheduler_tick(start - timedelta(minutes=1))
        check("not started before its time", not appmod.engine.running)
        appmod.scheduler_tick(start + timedelta(seconds=30))
        check("started by itself at start time", appmod.engine.running and appmod.engine.session == "Auto class",
              (appmod.engine.running, appmod.engine.session))
        appmod.scheduler_tick(start + timedelta(minutes=31))
        time.sleep(1)
        check("stopped by itself at end time", not appmod.engine.running)
        missed = C.post("/api/sessions", json={"name": "Missed class", "camera": clip, "auto_start": True,
                                               "start_at": (start - timedelta(hours=3)).strftime(FMT),
                                               "end_at": (start - timedelta(hours=2)).strftime(FMT)})
        appmod.scheduler_tick(start)
        m = [s for s in C.get("/api/sessions").get_json() if s["name"] == "Missed class"][0]
        check("window already over -> marked missed, not started", m["status"] == "done"
              and "Missed" in (m["last_error"] or "") and not appmod.engine.running, m)

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
