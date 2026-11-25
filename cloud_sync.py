"""Push attendance records from this laptop to the online dashboard (cloud/cloud_app.py).

Sent: student names, roll numbers, PIN hashes (for the online portal login), attendance
rows (session, name, time, match score, Face/Manual) and a live status line.
NEVER sent: photos, proof images, face embeddings, Telegram data.

Settings live in secrets.json: cloud_url (https://...) and cloud_token (same as SYNC_TOKEN
on the server). Set them in the app (Students tab -> Online dashboard).
"""
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request

import notify
from attend import open_db

INTERVAL_S = 5
HEARTBEAT_S = 60


class CloudSync:
    def __init__(self, live_fn=lambda: {}):
        self.live_fn = live_fn               # returns {"running", "session", "present", "enrolled"}
        self.status = "Not configured"
        self.last_ok = None
        self._last_hash = None
        self._last_sent = 0.0
        self._stop = False
        self._wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def settings():
        s = notify.load_secrets()
        return (s.get("cloud_url") or "").rstrip("/"), s.get("cloud_token") or ""

    def configure(self, url, token):
        url = (url or "").strip().rstrip("/")
        if url and not url.startswith("https://") and not url.startswith("http://127.0.0.1"):
            raise ValueError("Use the https:// address of the online dashboard.")
        notify.save_secret("cloud_url", url)
        notify.save_secret("cloud_token", (token or "").strip())
        self._last_hash = None
        self._wake.set()

    def snapshot(self):
        con = open_db()
        try:
            students = [{"name": n, "roll": r, "pin_hash": h, "salt": s}
                        for n, r, h, s in con.execute("SELECT name, roll, pin_hash, salt FROM students")]
            rows = [{"session": a, "name": b, "time": c, "similarity": d, "method": e, "votes": f, "zoom": g}
                    for a, b, c, d, e, f, g in con.execute(
                        "SELECT session, name, time, similarity, method, votes, zoom FROM attendance")]
        finally:
            con.close()
        return {"students": students, "attendance": rows, "live": self.live_fn()}

    def push_once(self, force=False):
        url, token = self.settings()
        if not url or not token:
            self.status = "Not configured"
            return False
        snap = self.snapshot()
        h = hashlib.sha256(json.dumps(snap, sort_keys=True, default=str).encode()).hexdigest()
        if not force and h == self._last_hash and time.time() - self._last_sent < HEARTBEAT_S:
            return True                                  # nothing changed
        req = urllib.request.Request(url + "/api/sync", data=json.dumps(snap, default=str).encode(),
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                json.loads(r.read())
            self._last_hash, self._last_sent, self.last_ok = h, time.time(), time.strftime("%H:%M:%S")
            self.status = f"Synced {len(snap['attendance'])} records at {self.last_ok}"
            return True
        except urllib.error.HTTPError as e:
            self.status = "Token rejected by the server" if e.code == 401 else f"Server error {e.code}"
        except Exception as e:
            self.status = f"Offline - will retry ({type(e).__name__})"
        return False

    def _loop(self):
        while not self._stop:
            try:
                self.push_once()
            except Exception as e:                       # never let sync kill the app
                self.status = f"Sync error: {e}"
            self._wake.wait(INTERVAL_S)
            self._wake.clear()
