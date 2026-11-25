# Database guide (for whoever takes over the database work)

The app uses **one SQLite file**: `attendance.db` in the project folder (path: `DB_FILE` in `config.py`).
It is **git-ignored** (it holds students' attendance and Telegram IDs) - never commit it.
All tables are created / migrated automatically by `attend.open_db()` the first time the app runs,
so a fresh clone needs no setup: start the app and the file appears.

Other private data that is NOT in the database (also git-ignored):

| What | Where | Made by |
| --- | --- | --- |
| Face embeddings (one 512-d vector per student) | `embeddings.npz` | `enroll.py` |
| Enrolment photos | `dataset/<Student Name>/` | teacher / `cluster_faces.py` |
| Proof photos of every automatic mark | `snapshots/` | `attend.mark_present()` |
| Telegram bot token + portal cookie key | `secrets.json` | Students tab / `notify.py` |

## Tables

### `attendance` - one row per student per session (the main table)
| Column | Meaning |
| --- | --- |
| `id` | auto number |
| `session` | session / lecture name typed in the app, e.g. `2026-11-19 Lecture` |
| `name` | student name (= folder name in `dataset/`) |
| `time` | when marked, `YYYY-MM-DD HH:MM:SS` (local time) |
| `similarity` | best face match score 0-1 (NULL for manual marks) |
| `face_px` | face width in pixels in the proof frame |
| `snapshot` / `frame_snapshot` | file names of the face crop / full frame in `snapshots/` |
| `votes` | frames that agreed before marking (3 for strong matches, else 5) |
| `method` | `auto` (face recognition) or `manual (teacher)` |
| `camera` | camera source used |
| `zoom` | zoom level of the proof frame (1.0 = no zoom) |

`UNIQUE(session, name)` + `INSERT OR IGNORE` = **a student can never be marked twice in one session**.
Keep this constraint if you migrate to another database.

### `students` - accounts for the portal / Telegram
| Column | Meaning |
| --- | --- |
| `name` | primary key, same as in `attendance` |
| `roll` | roll number (unique, upper-case) |
| `pin_hash`, `salt` | PBKDF2-SHA256 hash of the 6-digit PIN (the PIN itself is never stored) |
| `tg_chat_id` | Telegram chat linked with `/link ROLL PIN` (unique) |

### `notifications` - Telegram delivery log
`time, session, name, channel ('telegram'), status ('sent' / 'failed' / 'not linked' / 'no bot'), detail`

### `zoom_events` - every auto-zoom camera move
`session, time, action ('target' / 'sweep'), zoom, x, y (0-1 position), face_px, reason, seconds, recognised`

### `audit_log` - teacher actions
`time, session, name, action` (manual add / remove, deleted sessions)

## Who writes what

| Code | Writes |
| --- | --- |
| `attend.mark_present()` | `attendance` (auto) + proof photos |
| `attend.mark_manual()` | `attendance` (manual) |
| `app.py` `/api/manual`, `/api/delete` | `attendance`, `audit_log`, deletes `notifications` / `zoom_events` rows of a session |
| `students_db.py` | `students`, `notifications` |
| `app.Engine._make_autozoom` -> `log_event` | `zoom_events` |
| `student_portal.py` | reads only (a student sees only their own rows) |

## Things to keep when changing the database
1. **Parameterised queries only** (`?` placeholders) - no string-built SQL with user input.
2. Connections use `timeout=10` and are closed after each request (`contextlib.closing`).
3. The camera thread has its own connection; web requests open their own. Don't share one
   connection across threads.
4. If a write fails, `mark_present` deletes the proof photos and raises; the next frame retries.
   A replacement DB must keep "no record = no proof file = no Telegram message".
5. Times are stored as local-time text. If you move to PostgreSQL/MySQL, use `TIMESTAMP` and keep
   the `UNIQUE(session, name)` rule.
6. Tests that cover the database: `tests/test_system.py`, `tests/test_final.py` (DB failure,
   duplicates), `tests/test_audit_fixes.py`. They use a temporary database, never the real one.
