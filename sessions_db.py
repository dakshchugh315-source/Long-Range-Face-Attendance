"""Sessions (lectures): create, schedule, auto-start/stop, late marking.

A session is identified by its NAME everywhere (attendance.session = sessions.name),
so old attendance rows keep working. Times are local, "YYYY-MM-DDTHH:MM".
"""
from datetime import datetime, timedelta

FMT = "%Y-%m-%dT%H:%M"
STATUSES = ("scheduled", "live", "done")


def init(con):
    con.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,        -- same text as attendance.session
        course TEXT,
        start_at TEXT,                    -- local time YYYY-MM-DDTHH:MM (NULL = ad-hoc)
        end_at TEXT,
        camera TEXT,                      -- camera source used when it auto-starts
        autozoom INTEGER DEFAULT 0,
        auto_start INTEGER DEFAULT 0,     -- start/stop the camera by itself at start/end time
        late_after_min INTEGER DEFAULT 10,
        status TEXT DEFAULT 'scheduled',  -- scheduled / live / done
        last_error TEXT,
        created_at TEXT)""")
    con.commit()


def _parse(t):
    return datetime.strptime(t, FMT) if t else None


def _clean(d, partial=False):
    out = {}
    if "name" in d or not partial:
        name = " ".join(str(d.get("name") or "").split())[:60]
        if not name:
            raise ValueError("Give the session a name.")
        out["name"] = name
    for k in ("course", "camera"):
        if k in d:
            out[k] = (str(d[k]).strip() or None) if d[k] is not None else None
    for k in ("start_at", "end_at"):
        if k in d:
            v = (d[k] or "").strip()[:16] or None
            if v:
                _parse(v)                                   # raises ValueError if malformed
            out[k] = v
    for k in ("autozoom", "auto_start"):
        if k in d:
            out[k] = 1 if d[k] else 0
    if "late_after_min" in d:
        out["late_after_min"] = max(0, min(240, int(d["late_after_min"] or 0)))
    return out


def _check_times(row):
    s, e = _parse(row.get("start_at")), _parse(row.get("end_at"))
    if row.get("auto_start") and not (s and e):
        raise ValueError("Auto-start needs a start and an end time.")
    if s and e and e <= s:
        raise ValueError("End time must be after the start time.")
    if row.get("auto_start") and not row.get("camera"):
        raise ValueError("Auto-start needs a camera (set it on the Live tab first).")


def create(con, d):
    row = _clean(d)
    row.setdefault("late_after_min", 10)
    _check_times(row)
    row["status"] = "scheduled"
    row["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if con.execute("SELECT 1 FROM sessions WHERE name=?", (row["name"],)).fetchone() or \
            con.execute("SELECT 1 FROM attendance WHERE session=?", (row["name"],)).fetchone():
        raise ValueError(f'A session called "{row["name"]}" already exists.')
    cols = ", ".join(row)
    cur = con.execute(f"INSERT INTO sessions ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    con.commit()
    return cur.lastrowid


def get(con, sid):
    con.row_factory = None
    r = con.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if not r:
        return None
    cols = [c[0] for c in con.execute("SELECT * FROM sessions LIMIT 0").description]
    return dict(zip(cols, r))


def by_name(con, name):
    r = con.execute("SELECT id FROM sessions WHERE name=?", (name,)).fetchone()
    return get(con, r[0]) if r else None


def update(con, sid, d):
    cur = get(con, sid)
    if not cur:
        raise KeyError(sid)
    if cur["status"] == "live":
        raise ValueError("Stop the session before editing it.")
    changes = _clean(d, partial=True)
    if "name" in changes and changes["name"] != cur["name"]:
        if cur["status"] != "scheduled" or con.execute(
                "SELECT 1 FROM attendance WHERE session=?", (cur["name"],)).fetchone():
            raise ValueError("A session that already has attendance can't be renamed.")
        if con.execute("SELECT 1 FROM sessions WHERE name=? AND id<>?", (changes["name"], sid)).fetchone():
            raise ValueError(f'A session called "{changes["name"]}" already exists.')
    merged = {**cur, **changes}
    _check_times(merged)
    if changes:
        con.execute(f"UPDATE sessions SET {', '.join(k + '=?' for k in changes)} WHERE id=?",
                    [*changes.values(), sid])
        con.commit()
    return get(con, sid)


def delete(con, sid):
    cur = get(con, sid)
    if not cur:
        raise KeyError(sid)
    if cur["status"] == "live":
        raise ValueError("Stop the session first.")
    con.execute("DELETE FROM sessions WHERE id=?", (sid,))
    con.commit()
    return cur


def ensure(con, name, camera=None, autozoom=False):
    """Started from the Live tab without scheduling: create an ad-hoc session row."""
    s = by_name(con, name)
    if s:
        return s
    now = datetime.now().strftime(FMT)
    con.execute("INSERT INTO sessions (name, start_at, camera, autozoom, status, created_at) "
                "VALUES (?,?,?,?, 'scheduled', ?)",
                (name, now, str(camera) if camera is not None else None, 1 if autozoom else 0,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    return by_name(con, name)


def set_status(con, name, status, error=None):
    assert status in STATUSES
    con.execute("UPDATE sessions SET status=?, last_error=? WHERE name=?", (status, error, name))
    con.commit()


def set_error(con, sid, error):
    con.execute("UPDATE sessions SET last_error=? WHERE id=?", (error, sid))
    con.commit()


def due(con, now=None):
    """Scheduled auto-start sessions whose time window contains `now` (earliest first)."""
    now = (now or datetime.now()).strftime(FMT)
    con.row_factory = None
    ids = [r[0] for r in con.execute(
        "SELECT id FROM sessions WHERE auto_start=1 AND status='scheduled' AND start_at<=? AND end_at>? "
        "ORDER BY start_at", (now, now))]
    return [get(con, i) for i in ids]


def missed(con, now=None):
    """Auto-start sessions whose whole window passed without starting -> mark done."""
    now = (now or datetime.now()).strftime(FMT)
    names = [r[0] for r in con.execute(
        "SELECT name FROM sessions WHERE auto_start=1 AND status='scheduled' AND end_at<=?", (now,))]
    for n in names:
        set_status(con, n, "done", "Missed: the app was not running during the scheduled time.")
    return names


def is_late(session_row, mark_time):
    """mark_time 'YYYY-MM-DD HH:MM:SS'. Late = after start + late_after_min (scheduled sessions)."""
    if not session_row or not session_row.get("start_at") or not mark_time:
        return False
    start = _parse(session_row["start_at"])
    limit = start + timedelta(minutes=session_row.get("late_after_min") or 0)
    return datetime.strptime(mark_time[:19], "%Y-%m-%d %H:%M:%S") > limit


def list_all(con):
    """All sessions (scheduled ones without attendance too) with counts."""
    con.row_factory = None
    cols = [c[0] for c in con.execute("SELECT * FROM sessions LIMIT 0").description]
    known = {r[1]: dict(zip(cols, r)) for r in con.execute("SELECT * FROM sessions")}
    counts = {s: (n, t) for s, n, t in con.execute(
        "SELECT session, COUNT(*), MIN(time) FROM attendance GROUP BY session")}
    out = []
    for name in set(known) | set(counts):
        row = known.get(name) or {"id": None, "name": name, "status": "done", "course": None,
                                  "start_at": None, "end_at": None, "auto_start": 0, "late_after_min": 0}
        n, first = counts.get(name, (0, None))
        row["present"] = n
        row["started"] = first or (row.get("start_at") or "").replace("T", " ")
        late = 0
        if row.get("start_at") and n:
            late = sum(1 for (t,) in con.execute("SELECT time FROM attendance WHERE session=?", (name,))
                       if is_late(row, t))
        row["late"] = late
        out.append(row)
    live = [r for r in out if r["status"] == "live"]
    upcoming = sorted((r for r in out if r["status"] == "scheduled"), key=lambda r: r.get("start_at") or "")
    done = sorted((r for r in out if r["status"] not in ("live", "scheduled")),
                  key=lambda r: r["started"] or "", reverse=True)
    return live + upcoming + done
