"""Online dashboard + laptop sync, end to end on this laptop (no real server needed).

python tests/test_cloud.py
"""
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cloud"))
TMP = Path(tempfile.mkdtemp(prefix="fa_cloud_"))
os.environ.update(SYNC_TOKEN="test-token-123", TEACHER_PASSWORD="teach-pw", COOKIE_SECURE="0",
                  DATA_DIR=str(TMP / "cloud"), SECRET_KEY="x" * 64)

import config  # noqa: E402

config.DB_FILE = TMP / "laptop.db"
import notify  # noqa: E402

notify.SECRETS = TMP / "secrets.json"
import attend  # noqa: E402
import cloud_app  # noqa: E402
import cloud_sync  # noqa: E402
import students_db  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

PORT = 8091
URL = f"http://127.0.0.1:{PORT}"
results = []


def check(name, ok, info=None):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   <- {info}" if not ok and info is not None else ""))


def call(path, data=None, opener=None, method=None):
    r = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                               headers={"Content-Type": "application/json"},
                               method=method or ("POST" if data is not None else "GET"))
    try:
        with (opener or urllib.request.build_opener()).open(r, timeout=10) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body[:1] in b"{[" else body.decode())
    except urllib.error.HTTPError as e:
        return e.code, None


def main():
    threading.Thread(target=make_server("127.0.0.1", PORT, cloud_app.app, threaded=True).serve_forever,
                     daemon=True).start()
    con = attend.open_db()
    students_db.roster(con, {"Kanika", "Yatri"})
    students_db.set_roll(con, "Kanika", "R1")
    pin = students_db.new_pin(con, "Kanika")
    attend.mark_manual(con, "Lec 1", "Kanika")
    con.execute("INSERT INTO attendance(session, name, time, similarity, method, snapshot, frame_snapshot) "
                "VALUES ('Lec 1', 'Yatri', '2026-11-25 10:00:00', 0.71, 'auto', 'secret_face.jpg', 'secret_frame.jpg')")
    con.commit()
    sync = cloud_sync.CloudSync(lambda: {"running": True, "session": "Lec 1", "present": 2, "enrolled": 2})

    print("\n1. Sync from the laptop")
    check("not configured -> does nothing", sync.push_once() is False and sync.status == "Not configured")
    try:
        sync.configure("http://evil.example", "t")
        check("plain http to a remote server refused", False)
    except ValueError:
        check("plain http to a remote server refused", True)
    sync.configure(URL, "wrong-token")
    check("wrong token -> rejected, app keeps running", not sync.push_once(force=True)
          and "rejected" in sync.status, sync.status)
    sync.configure(URL, "test-token-123")
    check("right token -> synced", sync.push_once(force=True), sync.status)
    check("nothing changed -> not re-sent", sync.push_once() and sync._last_hash is not None)

    print("\n2. Teacher dashboard")
    check("data needs login", call("/api/overview")[0] == 401)
    jar = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    check("wrong password refused", call("/login", {"password": "nope"}, jar)[0] == 401)
    check("right password works", call("/login", {"password": "teach-pw"}, jar)[0] == 200)
    s, d = call("/api/overview", opener=jar)
    sess = d["sessions"][0] if d and d["sessions"] else {}
    check("session with both students present", {r["name"] for r in sess.get("present", [])} == {"Kanika", "Yatri"},
          sess)
    check("live status from the laptop shown", d and d["live"].get("running") is True and d["last_sync_s_ago"] < 5)
    s, csv_text = call("/api/export?session=Lec%201", opener=jar)
    check("CSV export", s == 200 and "Kanika" in csv_text and "Manual" in csv_text, csv_text)

    print("\n3. Privacy: no photos / face data online")
    import sqlite3
    cdb = sqlite3.connect(TMP / "cloud" / "cloud.db")
    cols = {r[1] for t in ("attendance", "students") for r in cdb.execute(f"PRAGMA table_info({t})")}
    dump = "\n".join(cdb.iterdump())
    cdb.close()
    check("no photo / snapshot columns online", not (cols & {"snapshot", "frame_snapshot", "embedding"}), cols)
    check("proof file names never sent", "secret_face" not in dump and "secret_frame" not in dump)
    check("PIN stored only as hash online", pin not in dump)

    print("\n4. Student portal")
    st = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    check("wrong PIN refused", call("/student/login", {"roll": "R1", "pin": "000000" if pin != "000000" else "1"}, st)[0] == 401)
    check("right roll + PIN works", call("/student/login", {"roll": "r1", "pin": pin}, st)[0] == 200)
    s, me = call("/api/me", opener=st)
    check("sees own record only", me and me["name"] == "Kanika" and len(me["records"]) == 1, me)
    check("student cannot open teacher data", call("/api/overview", opener=st)[0] == 401)

    print("\n5. Changes on the laptop reach the cloud")
    con.execute("DELETE FROM attendance WHERE name='Yatri'")
    con.commit()
    sync.push_once()
    s, d = call("/api/overview", opener=jar)
    check("record deleted on laptop disappears online", {r["name"] for r in d["sessions"][0]["present"]} == {"Kanika"})

    print("\n6. Robustness")
    check("sync with no token header -> 401", call("/api/sync", {"students": []})[0] == 401)
    sync.configure("http://127.0.0.1:9", "test-token-123")          # nothing listens there
    check("server offline -> laptop keeps working, says offline", not sync.push_once(force=True)
          and "Offline" in sync.status, sync.status)
    codes = [call("/student/login", {"roll": "R1", "pin": "999999"})[0] for _ in range(10)]
    check("PIN guessing is rate-limited", 429 in codes, codes)
    con.close()

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
