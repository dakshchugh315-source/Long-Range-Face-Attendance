"""Student accounts: roll number, PIN (hashed), Telegram chat id."""
import hashlib
import hmac
import secrets
import sqlite3


def init(con):
    con.execute("""CREATE TABLE IF NOT EXISTS students (
        name TEXT PRIMARY KEY,          -- same as dataset/<name> folder
        roll TEXT UNIQUE,
        pin_hash TEXT,
        salt TEXT,
        tg_chat_id INTEGER UNIQUE)""")
    con.execute("""CREATE TABLE IF NOT EXISTS notifications (
        time TEXT, session TEXT, name TEXT, channel TEXT, status TEXT, detail TEXT)""")
    con.commit()


def log_notification(con, session, name, status, detail=""):
    from datetime import datetime
    con.execute("INSERT INTO notifications VALUES (?,?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session, name, "telegram",
                 status, str(detail)[:300]))
    con.commit()


def notification_status(con, session):
    """Latest Telegram delivery status per student for a session."""
    out = {}
    for name, status in con.execute(
            "SELECT name, status FROM notifications WHERE session=? ORDER BY time", (session,)):
        out[name] = status
    return out


def _hash(pin, salt):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), 120_000).hex()


def roster(con, names):
    """All enrolled students with account info (creates missing rows)."""
    for n in names:
        con.execute("INSERT OR IGNORE INTO students(name) VALUES (?)", (n,))
    con.commit()
    out = []
    for name, roll, pin_hash, chat in con.execute(
            "SELECT name, roll, pin_hash, tg_chat_id FROM students ORDER BY name"):
        if name in names:
            out.append({"name": name, "roll": roll or "", "has_pin": bool(pin_hash),
                        "telegram": chat is not None})
    return out


def set_roll(con, name, roll):
    roll = (roll or "").strip().upper() or None
    try:
        con.execute("UPDATE students SET roll=? WHERE name=?", (roll, name))
        con.commit()
    except sqlite3.IntegrityError:
        raise ValueError(f"Roll no {roll} is already used by another student")


def new_pin(con, name):
    """Generate a 6-digit PIN, store only its hash, return it once."""
    pin = f"{secrets.randbelow(10 ** 6):06d}"
    salt = secrets.token_hex(16)
    con.execute("UPDATE students SET pin_hash=?, salt=? WHERE name=?", (_hash(pin, salt), salt, name))
    con.commit()
    return pin


def verify(con, roll, pin):
    """Return student name if roll + PIN are correct, else None."""
    row = con.execute("SELECT name, pin_hash, salt FROM students WHERE roll=?",
                      ((roll or "").strip().upper(),)).fetchone()
    if not row or not row[1]:
        _hash(pin or "", "00" * 16)            # same work either way (timing)
        return None
    return row[0] if hmac.compare_digest(_hash(pin or "", row[2]), row[1]) else None


def link_telegram(con, name, chat_id):
    con.execute("UPDATE students SET tg_chat_id=NULL WHERE tg_chat_id=?", (chat_id,))
    con.execute("UPDATE students SET tg_chat_id=? WHERE name=?", (chat_id, name))
    con.commit()


def unlink_telegram(con, chat_id):
    con.execute("UPDATE students SET tg_chat_id=NULL WHERE tg_chat_id=?", (chat_id,))
    con.commit()


def chat_of(con, name):
    r = con.execute("SELECT tg_chat_id FROM students WHERE name=?", (name,)).fetchone()
    return r[0] if r else None


def name_of_chat(con, chat_id):
    r = con.execute("SELECT name FROM students WHERE tg_chat_id=?", (chat_id,)).fetchone()
    return r[0] if r else None


def history(con, name, limit=200):
    con.row_factory = None
    cols = ["session", "time", "method", "similarity", "snapshot"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM attendance WHERE name=? "
                       "ORDER BY time DESC LIMIT ?", (name, limit)).fetchall()
    return [dict(zip(cols, r)) for r in rows]
