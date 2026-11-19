"""Central settings. Change values here instead of inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"          # dataset/<student_name>/*.jpg
EMBEDDINGS_FILE = ROOT / "embeddings.npz"
DB_FILE = ROOT / "attendance.db"
SNAPSHOT_DIR = ROOT / "snapshots"       # proof photo saved when someone is marked present

# Camera: 0 = laptop webcam. For phone/CCTV use an RTSP/HTTP url, e.g.
#   "rtsp://user:pass@192.168.1.20:554/stream1"
#   "http://192.168.1.5:8080/video"   (Android "IP Webcam" app)
CAMERA_SOURCE = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# InsightFace model pack:
#   "buffalo_s" -> ~8x faster on this laptop's CPU (default)
#   "buffalo_l" -> more accurate, better for far/small faces, but slow on CPU
# After changing this, run enroll.py again (embeddings differ per model).
MODEL_NAME = "buffalo_s"
DET_SIZE = (640, 640)       # bigger = finds smaller/farther faces, but slower
DET_THRESHOLD = 0.5

# Long range: also scan the frame in overlapping full-resolution tiles so small,
# far-away faces aren't shrunk away. Costs more CPU per frame.
DET_TILES = True
TILE_SIZE = 640
TILE_OVERLAP = 0.25
TILE_EDGE_MARGIN = 8        # tile boxes this close to an inner tile border are cut-off duplicates

# Recognition
MATCH_THRESHOLD = 0.40      # cosine similarity; below this -> "Unknown"
MARGIN = 0.05               # best match must beat 2nd best by this much
MIN_FACE_PX = 24            # don't try to recognise faces narrower than this
                            # (evaluate_distance.py: 0% wrong matches down to 12 px,
                            #  ~87% correct at 28 px, 64% at 22 px per single frame)

# Auto-zoom (phone running IP Webcam): software zooms onto far faces by itself
AUTOZOOM_MAX = 3.0          # digital zoom beyond ~2.5-3x adds no real detail on M31
AUTOZOOM_TARGET_PX = 48     # faces smaller than this (and not yet marked) get zoomed on
AUTOZOOM_GOAL_PX = 80       # zoom so the target face becomes about this wide
AUTOZOOM_WIDE_S = 4         # stay at 1x at least this long between moves
AUTOZOOM_DWELL_S = 4        # look this long after each zoom
AUTOZOOM_SETTLE_S = 2.0     # ignore decisions while the camera moves/refocuses (+H.264 delay)
AUTOZOOM_REVISIT_S = 25     # don't zoom on the same spot again within this time
AUTOZOOM_SWEEP_EVERY_S = 30 # scan the room in a grid this often while people are absent
AUTOZOOM_SWEEP_ZOOM = 2.5
AUTOZOOM_SWEEP_MAX_S = 300  # sweeps that find nobody back off 30 -> 60 -> 120 ... up to this
# Face-scan priority: while a face is detected / being scanned the camera stays put
AUTOZOOM_FACE_LOSS_S = 4    # face missing this long = genuinely lost (shorter = brief dropout)
AUTOZOOM_SCAN_STALL_S = 12  # a face that gains no vote this long stops holding (stranger / turned away)
AUTOZOOM_MAX_HOLD_S = 45    # one face can hold the camera at most this long

# PTZ CCTV over ONVIF (only if the camera has motorised zoom; check with camera_check.py)
ONVIF_OPTICAL_ZOOM = 4.0    # camera's optical zoom range (e.g. 4x for 2.8-12 mm)
ONVIF_HFOV_DEG = 90         # horizontal field of view at 1x (2.8 mm lens ~ 90-100 deg)
ONVIF_PORT = 80            # ONVIF service port (Hikvision 80, CP Plus/Dahua 80, some 8000)

# Far faces are also compared with a "low-res gallery" (enrollment photos shrunk
# to LOWRES_PX wide) because a blurry face matches a blurry reference better.
USE_LOWRES_GALLERY = False  # off until evaluate_distance.py shows it helps
LOWRES_PX = 24
LOWRES_BELOW_PX = 40        # use it for faces narrower than this

# Backlight / dim room: brighten a face locally before recognising it
ENHANCE_DARK_FACES = False  # tested: no gain on darkened faces (97-99% either way), so off
DARK_FACE_LEVEL = 70        # mean grey level (0-255) below which a face counts as dark

# Voting: a person is marked present once their name was recognised in this
# many frames within VOTE_WINDOW_S seconds (works even at 1-2 FPS).
VOTES_TO_MARK = 5
VOTES_STRONG = 3            # ...only 3 when the best match is strong (>= STRONG_MATCH)
STRONG_MATCH = 0.55
VOTE_WINDOW_S = 30
LABEL_HOLD_FRAMES = 6       # keep showing a track's name this many frames after it was recognised
REVERIFY_EVERY = 5          # re-recognise an already-marked face every N frames (catches swaps)
