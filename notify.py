"""Telegram bot: students link once, then get a message whenever they're marked.

Teacher: create a bot with @BotFather in Telegram, paste its token in the app
(Students tab -> Telegram bot). Token is saved in secrets.json (keep it private).

Student, in the bot chat:
  /link <roll no> <PIN>   connect this Telegram account
  /today                  today's attendance
  /history                last 10 records
  /unlink                 disconnect
"""
import json
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime

import config
import students_db

SECRETS = config.ROOT / "secrets.json"


def load_secrets():
    try:
        return json.loads(SECRETS.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def save_secret(key, value):
    s = load_secrets()
    s[key] = value
    SECRETS.write_text(json.dumps(s, indent=2), encoding="utf-8")


class TelegramBot:
    def __init__(self, db_factory):
        self.db_factory = db_factory          # returns a new sqlite connection
        self.token = load_secrets().get("telegram_token")
        self.username = None
        self.status = "Not configured"
        self.q = queue.Queue()
        self.fails = defaultdict(list)        # chat_id -> failed /link timestamps
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._poller, daemon=True).start()
        if self.token:
            self._check()

    # ------------------------------------------------------------- api
    def _call(self, method, timeout=35, **params):
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        data = json.dumps(params).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["result"]

    def _check(self):
        try:
            me = self._call("getMe", timeout=10)
            self.username = me["username"]
            self.status = f"Connected as @{self.username}"
            return True
        except Exception as e:
            self.username = None
            self.status = f"Token not working ({type(e).__name__})"
            return False

    def set_token(self, token):
        old = self.token
        self.token = token.strip()
        if self._check():
            save_secret("telegram_token", self.token)
            return True
        self.token = old
        if old:
            self._check()
        return False

    def send(self, chat_id, text):
        self._call("sendMessage", timeout=15, chat_id=chat_id, text=text)

    # ------------------------------------------------------------- outgoing
    def notify(self, name, session, method, similarity=None):
        """Queue an 'attendance marked' message (never blocks the camera loop)."""
        self.q.put((name, session, method, similarity, datetime.now()))

    def _sender(self):
        while True:
            name, session, method, sim, when = self.q.get()
            con = self.db_factory()
            try:
                chat = students_db.chat_of(con, name)
                if not self.token:
                    students_db.log_notification(con, session, name, "no bot")
                    continue
                if chat is None:
                    students_db.log_notification(con, session, name, "not linked")
                    continue
                how = (f"through Face recognition (match {sim:.2f})" if method == "face"
                       else "by your teacher (manual)")
                text = (f"✅ Hi {name},\nyour attendance for \"{session}\" is marked {how}.\n"
                        f"🕒 {when:%d %b %Y, %I:%M %p}")
                err = None
                for attempt in range(5):            # survive short Wi-Fi drops
                    try:
                        self.send(chat, text)
                        err = None
                        break
                    except Exception as e:
                        err = e
                        time.sleep(3 * (attempt + 1))
                students_db.log_notification(con, session, name,
                                             "sent" if err is None else "failed", err or "")
            except Exception:
                pass
            finally:
                con.close()

    # ------------------------------------------------------------- incoming
    def _poller(self):
        offset = None
        while True:
            if not self.token:
                time.sleep(3)
                continue
            try:
                params = {"timeout": 25, "allowed_updates": ["message"]}
                if offset:
                    params["offset"] = offset
                for upd in self._call("getUpdates", **params):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    if msg.get("chat", {}).get("type") == "private" and msg.get("text"):
                        try:
                            self._handle(msg["chat"]["id"], msg["text"].strip())
                        except Exception:
                            pass
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    self.status = "Token rejected by Telegram"
                time.sleep(5)
            except Exception:
                time.sleep(5)

    def _handle(self, chat, text):
        cmd, *args = text.split()
        cmd = cmd.lower().split("@")[0]
        con = self.db_factory()
        try:
            me = students_db.name_of_chat(con, chat)
            if cmd == "/link":
                recent = [t for t in self.fails[chat] if time.time() - t < 600]
                self.fails[chat] = recent
                if len(recent) >= 5:
                    return self.send(chat, "Too many wrong tries. Wait 10 minutes.")
                if len(args) != 2:
                    return self.send(chat, "Send: /link <roll no> <PIN>\nExample: /link 2023UCS1234 482913")
                name = students_db.verify(con, args[0], args[1])
                if not name:
                    self.fails[chat].append(time.time())
                    return self.send(chat, "❌ Roll no or PIN is wrong. Ask your teacher for your PIN.")
                students_db.link_telegram(con, name, chat)
                return self.send(chat, f"🔗 Linked as {name}. You'll get a message here every "
                                       f"time your attendance is marked.")
            if cmd == "/unlink":
                students_db.unlink_telegram(con, chat)
                return self.send(chat, "Unlinked. You won't get attendance messages any more.")
            if not me:
                return self.send(chat, "👋 Face Attendance bot.\nLink your account first:\n"
                                       "/link <roll no> <PIN>")
            hist = students_db.history(con, me, limit=200)
            if cmd == "/today":
                today = datetime.now().strftime("%Y-%m-%d")
                rows = [h for h in hist if h["time"].startswith(today)]
                body = "\n".join(f"• {h['session']} — {h['time'][11:16]} "
                                 f"({'Face' if h['method'] == 'auto' else 'Manual'})" for h in rows)
                return self.send(chat, f"Today, {me}:\n{body or 'No attendance marked yet.'}")
            if cmd == "/history":
                body = "\n".join(f"• {h['time'][:16]} — {h['session']} "
                                 f"({'Face' if h['method'] == 'auto' else 'Manual'})" for h in hist[:10])
                return self.send(chat, f"Last records for {me}:\n{body or 'Nothing yet.'}")
            return self.send(chat, f"Hi {me}! Commands: /today  /history  /unlink")
        finally:
            con.close()
