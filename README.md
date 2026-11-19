# Long-Range Face Attendance

> **Status: ongoing project.** It is still being built and tested in real classrooms. Once it is complete, the full version will be released as open source.

**Marks classroom attendance from a distance using a single CCTV or phone camera: no queues, no ID cards, no roll call.**

The camera watches the room, finds every face (even small ones at the back), recognises who it is and marks them present, saving a photo as proof. Students get a Telegram message the moment they are marked.

It runs on a normal laptop CPU. No GPU and no model training from scratch are needed.

---

## How it works

```
Camera (CCTV / phone)  →  Face detection (SCRFD)  →  Face recognition (ArcFace, 512-d embedding)
        ↑                         ↓                                  ↓
  Auto-zoom on far faces    Tiled scan for tiny faces       Match against enrolled students
                                                                     ↓
                                     k-of-n voting over several frames  →  Mark present + proof photo
                                                                     ↓
                                          Teacher dashboard  ·  Student portal  ·  Telegram alert
```

| Part | What it does |
| --- | --- |
| **Detection**: InsightFace SCRFD | Finds faces in each frame. Large frames are also scanned in overlapping full-resolution tiles, so faces at the back of the room aren't lost when the frame is shrunk. |
| **Recognition**: InsightFace ArcFace | Turns each face into a 512-number embedding and compares it with every student's averaged embedding (cosine similarity, with a minimum margin over the 2nd-best match). |
| **Voting** | A student is marked only after they are recognised in several frames (k-of-n), so a single wrong frame can't mark the wrong person. |
| **Auto-zoom** | Zooms in on faces that are too small, then zooms back out. Works with a phone's digital zoom (IP Webcam app) or a motorised PTZ CCTV over ONVIF. When people are missing it sweeps the room in a grid. |
| **Proof** | Every automatic mark saves the face crop plus the full frame, with a banner showing name, match score, face size, votes, time and session. |
| **Apps** | Flask teacher app (live view, records, students) and a student portal for phones on the same Wi-Fi. |
| **Notifications** | Telegram bot: each student links once and is messaged every time they are marked. |

**Enrolment = no training.** Each student's photos (or a short video) are turned into embeddings and averaged into one 512-d vector (`enroll.py`). Adding a new student takes seconds.

---

## Quick start (Windows)

```bat
setup.bat          :: creates .venv and installs requirements (first run downloads ~120 MB of models)
run_enroll.bat     :: builds embeddings from dataset\<StudentName>\*.jpg
run_app.bat        :: starts the teacher app and the student portal
```

| App | Address | Who uses it |
| --- | --- | --- |
| Teacher app | `http://127.0.0.1:5000` | Only on the laptop running the system |
| Student portal | `http://LAPTOP-IP:5001` | Phones on the same Wi-Fi. Replace `LAPTOP-IP` with the laptop's Wi-Fi address (run `ipconfig` and look for *IPv4 Address*), and allow Python on **Private networks** when Windows Firewall asks. |

**Teacher app tabs**
- **Live**: camera feed, automatic marking, manual mark.
- **Records**: sessions, proof photos, CSV export.
- **Students**: add photos, re-enroll, roll number and PIN, Telegram bot setup.

Manual marks are labelled `manual (teacher)`, and every manual add or remove is written to an audit log.

---

## Camera options

Set `CAMERA_SOURCE` in `config.py`:

| Camera | Value |
| --- | --- |
| Laptop webcam | `0` |
| Android phone with the **IP Webcam** app | `"http://PHONE-IP:8080/video"` (MJPEG) or `"rtsp://PHONE-IP:8080/h264_ulaw.sdp"` (H.264) |
| CCTV / IP camera | `"rtsp://user:pass@CAMERA-IP:554/stream1"` |

PTZ cameras with motorised zoom are driven over ONVIF (`ONVIF_*` settings).

---

## Telegram notifications

1. In Telegram, open **@BotFather**, send `/newbot`, copy the token and paste it in the **Students** tab.
2. Set each student's roll number and press **Generate PIN**. It is shown only once, so give it to the student.
3. The student sends `/link ROLL_NO PIN` to the bot once. After that, every mark sends:
   *"Your attendance for SESSION is marked through Face recognition"*.

Bot commands: `/today`, `/history`, `/unlink`.

---

## Command-line tools

```bash
python capture.py --name Rahul            # collect photos of one person with the webcam
python cluster_faces.py PHOTO_FOLDER      # group an unlabelled photo dump by person
python cluster_faces.py --apply "01=Rahul,02=Priya"
python enroll.py                          # build embeddings from dataset/
python attend.py --source CAMERA          # simple OpenCV-window version, no web app
python evaluate_distance.py               # synthetic long-range accuracy test (see below)
```

---

## How far can it see?

`evaluate_distance.py` shrinks each enrolled photo until the face is only N pixels wide, adds CCTV-like blur and JPEG compression, pastes it into a 1920×1080 classroom frame and runs the real pipeline on it. Each photo is tested against a gallery built without it (leave-one-out).

On the synthetic test, per single frame:
- **0% wrong matches** down to 12-pixel faces. When unsure, it answers *Unknown* rather than the wrong name.
- **~87% correct** at 28 px and **~64%** at 22 px. Voting over several frames and auto-zoom raise this further.

The script also converts face size to metres for the camera in use. Real-room distance testing is ongoing.

---

## Key settings (`config.py`)

| Setting | Meaning |
| --- | --- |
| `MODEL_NAME` | `buffalo_s` (fast, default on CPU) or `buffalo_l` (more accurate, ~8× slower). Re-enrol after changing. |
| `MATCH_THRESHOLD`, `MARGIN` | How similar a face must be, and how far ahead of the 2nd-best match. |
| `VOTES_TO_MARK` | Frames needed before marking someone present. |
| `MIN_FACE_PX` | Smallest face (in pixels) the system tries to recognise. |
| `DET_TILES`, `TILE_SIZE` | Tiled full-resolution scan for small, far faces. |
| `AUTOZOOM_*`, `ONVIF_*` | Auto-zoom and PTZ behaviour. |

---

## Privacy

Face data never leaves the machine. These files are git-ignored and must never be shared:
`dataset/`, `raw_icloud/`, `candidates/`, `snapshots/`, `attendance.db`, `embeddings.npz` and `secrets.json` (Telegram token and portal key).

## Tech stack

Python · InsightFace (SCRFD + ArcFace) · ONNX Runtime · OpenCV · NumPy · scikit-learn · Flask · SQLite · Telegram Bot API · ONVIF

## Author

**Daksh Chugh**: B.Tech AI & ML, Bennett University · [GitHub](https://github.com/dakshchugh315-source) · [LinkedIn](https://www.linkedin.com/in/daksh-chugh-5175a8439/)
