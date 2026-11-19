# Face Attendance (CCTV / phone camera)

Pretrained InsightFace (SCRFD detector + ArcFace recognizer). No training from scratch:
each student's photos become one averaged 512-d embedding.

## Start the app
Double-click **run_app.bat** (or `.venv\Scripts\python app.py`).
- Teacher app: http://127.0.0.1:5000 — only on this laptop
- Student portal: http://<laptop-wifi-ip>:5001 — phones on the same Wi-Fi
  (allow Python on *Private networks* when Windows Firewall asks)

Tabs: **Live** (camera → automatic marking, manual mark), **Records** (sessions, proof, CSV),
**Students** (add photos, re-enroll, roll no / PIN, Telegram bot).

## Proof of every detection
For each automatic mark: face crop + full camera frame with box and banner
(name, match score, face px, votes, time, session) in `snapshots/`, plus a DB row.
Manual marks are labelled `manual (teacher)`; manual add/remove is logged in `audit_log`.

## Student notifications
1. In Telegram, talk to @BotFather → /newbot → copy the token → paste in Students tab.
2. Set each student's roll no and press Generate PIN (shown once — give it to the student).
3. Student sends `/link <roll no> <PIN>` to the bot once. From then on, every mark sends
   "Your attendance for <session> is marked through Face recognition".
   Bot commands: /today /history /unlink.

## Command-line tools
```
python capture.py --name Rahul          # collect photos with webcam
python cluster_faces.py <photo_folder>  # group an unlabeled photo dump by person
python cluster_faces.py --apply "01=Rahul,02=Priya"
python enroll.py                        # build embeddings from dataset/
python attend.py --source <cam>         # simple OpenCV-window version
```

## Files that are private (never share / commit)
`dataset/`, `raw_icloud/`, `candidates/`, `snapshots/`, `attendance.db`, `embeddings.npz`,
`secrets.json` (Telegram token + portal key).

## Settings (config.py)
`CAMERA_SOURCE`, `MODEL_NAME` (buffalo_s fast / buffalo_l accurate — re-enroll after changing),
`MATCH_THRESHOLD`, `VOTES_TO_MARK`, `MIN_FACE_PX`.
