"""Shared face detection + recognition helpers (InsightFace: SCRFD + ArcFace)."""
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

import config

_app = None


def get_app() -> FaceAnalysis:
    """Load the models once (first run downloads ~300 MB to ~/.insightface)."""
    global _app
    if _app is None:
        _app = FaceAnalysis(
            name=config.MODEL_NAME,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=-1, det_size=config.DET_SIZE, det_thresh=config.DET_THRESHOLD)
    return _app


def ensure_bgr(img):
    """IR / night cameras can give 1-channel (grayscale) frames, PNGs can have alpha.
    The models need 3-channel BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _detect_raw(img):
    bboxes, kpss = get_app().models["detection"].detect(img, max_num=0, metric="default")
    return bboxes, kpss


def tile_grid(w, h, tile=None, overlap=None):
    """Overlapping tiles about det_size big, so the detector sees them at ~full resolution."""
    tile = tile or config.TILE_SIZE
    overlap = config.TILE_OVERLAP if overlap is None else overlap
    if w <= tile * 1.2 and h <= tile * 1.2:
        return []
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y, min(tile, w), min(tile, h)) for y in ys for x in xs]


def detect(frame, tiles=None, roi=None):
    """Detection only (cheap part). Returns faces with .bbox, .kps, .det_score.

    tiles: run the detector on overlapping full-resolution tiles as well as on the
           whole (downscaled) frame. Finds far-away / small faces that would
           otherwise shrink below detector size. Default: config.DET_TILES.
    roi:   (x1, y1, x2, y2) only look inside this region (e.g. the seating area).
    """
    tiles = config.DET_TILES if tiles is None else tiles
    ox = oy = 0
    frame = ensure_bgr(frame)
    img = frame
    if roi:
        x1, y1, x2, y2 = (int(v) for v in roi)
        img, ox, oy = frame[y1:y2, x1:x2], x1, y1
    h, w = img.shape[:2]

    all_b, all_k = [], []
    b, k = _detect_raw(img)                      # whole view: catches big/near faces
    all_b.append(b)
    all_k.append(k)
    if tiles:
        m = config.TILE_EDGE_MARGIN             # px: "touching the tile border"
        for tx, ty, tw, th in tile_grid(w, h):
            b, k = _detect_raw(img[ty:ty + th, tx:tx + tw])
            if len(b):
                # A face cut by an INNER tile border gives a half box (a duplicate of the real
                # face, often "Unknown"). Drop it: the 25% overlap means a small face is whole in
                # a neighbouring tile, and a big one is found by the whole-frame pass.
                cut = np.zeros(len(b), bool)
                if tx > 0:
                    cut |= b[:, 0] <= m
                if tx + tw < w:
                    cut |= b[:, 2] >= tw - m
                if ty > 0:
                    cut |= b[:, 1] <= m
                if ty + th < h:
                    cut |= b[:, 3] >= th - m
                b = b[~cut]
                k = k[~cut] if k is not None else None
            if len(b):
                b = b.copy()
                b[:, [0, 2]] += tx
                b[:, [1, 3]] += ty
                if k is not None:
                    k = k.copy()
                    k[:, :, 0] += tx
                    k[:, :, 1] += ty
            all_b.append(b)
            all_k.append(k)

    B = np.concatenate([b for b in all_b if len(b)]) if any(len(b) for b in all_b) else np.zeros((0, 5))
    K = np.concatenate([k for b, k in zip(all_b, all_k) if len(b)]) if len(B) else None
    if len(B) == 0:
        return []
    if tiles:                                    # merge duplicates from overlapping tiles
        xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2, _ in B]
        keep = cv2.dnn.NMSBoxes(xywh, B[:, 4].astype(float).tolist(), config.DET_THRESHOLD, 0.4)
        keep = np.array(keep).flatten()
        B, K = B[keep], (K[keep] if K is not None else None)
        # NMS misses a small box lying inside a big one (low IoU). Keep one box per face:
        # largest first, drop any box mostly (>50% of its area) inside a kept one.
        area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        kept = []
        for i in np.argsort(-area):
            inside = False
            for j in kept:
                ix = max(0, min(B[i, 2], B[j, 2]) - max(B[i, 0], B[j, 0]))
                iy = max(0, min(B[i, 3], B[j, 3]) - max(B[i, 1], B[j, 1]))
                if ix * iy > 0.5 * min(area[i], area[j]):
                    inside = True
                    break
            if not inside:
                kept.append(i)
        kept = sorted(kept)
        B, K = B[kept], (K[kept] if K is not None else None)

    faces = []
    for i in range(B.shape[0]):
        bb = B[i, :4] + [ox, oy, ox, oy]
        kp = K[i] + [ox, oy] if K is not None else None
        faces.append(Face(bbox=bb, kps=kp, det_score=B[i, 4]))
    return faces


def face_brightness(frame, face):
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    h, w = frame.shape[:2]
    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    return float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()) if roi.size else 255.0


def _enhance(patch):
    """Brighten a dark / backlit face: CLAHE on lightness + gamma lift."""
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    mean = max(float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()), 1.0)
    gamma = min(max(np.log(120 / 255) / np.log(mean / 255), 0.4), 1.0)   # aim for mid-grey
    lut = np.array([((i / 255) ** gamma) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(out, lut)


def embed(frame, face, enhance=None):
    """Align + ArcFace (expensive part). Returns the 512-d normalised embedding.
    Dark faces (backlight from a window, dim room) are brightened locally first."""
    if face.get("embedding") is not None:
        return face.normed_embedding
    frame = ensure_bgr(frame)
    enhance = config.ENHANCE_DARK_FACES if enhance is None else enhance
    rec = get_app().models["recognition"]
    if enhance and face.kps is not None and face_brightness(frame, face) < config.DARK_FACE_LEVEL:
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        pad = int((x2 - x1) * 0.6)
        h, w = frame.shape[:2]
        ox, oy = max(0, x1 - pad), max(0, y1 - pad)
        patch = _enhance(frame[oy:min(h, y2 + pad), ox:min(w, x2 + pad)].copy())
        f2 = Face(bbox=face.bbox - [ox, oy, ox, oy], kps=face.kps - [ox, oy], det_score=face.det_score)
        rec.get(patch, f2)
        face["embedding"] = f2.embedding
        face["enhanced"] = True
        return face.normed_embedding
    rec.get(frame, face)
    return face.normed_embedding


def embed_batch(frame, faces):
    """Recognise many faces of one frame in ONE model call (a crowded classroom has
    10-40 faces per frame; one batched ArcFace run is cheaper than N single runs).
    Sets face.embedding on each face; equivalent to embed() without dark-face enhancement."""
    todo = [f for f in faces if f.get("embedding") is None and f.kps is not None]
    if not todo:
        return
    if config.ENHANCE_DARK_FACES:                 # rare path: keep the per-face enhancer
        for f in todo:
            embed(frame, f)
        return
    from insightface.utils import face_align
    frame = ensure_bgr(frame)
    rec = get_app().models["recognition"]
    size = rec.input_size[0]
    crops = [face_align.norm_crop(frame, landmark=f.kps, image_size=size) for f in todo]
    for start in range(0, len(crops), 32):        # keep memory bounded for very large classes
        feats = rec.get_feat(crops[start:start + 32])
        for f, e in zip(todo[start:start + 32], feats):
            f["embedding"] = e


def is_turned_away(face):
    """True for strong side/back views (nose far outside the eyes) - such faces
    often can't be recognised; shown as 'turned away' instead of 'Unknown'."""
    if face.kps is None:
        return False
    le, re_, nose = face.kps[0], face.kps[1], face.kps[2]
    eye_d = float(np.linalg.norm(re_ - le))
    width = float(face.bbox[2] - face.bbox[0])
    off = abs(float(nose[0] - (le[0] + re_[0]) / 2)) / max(eye_d, 1.0)
    return eye_d < 0.25 * width or off > 0.6


def lowres_embedding(img, face, px=None):
    """Embedding of the same face as a far-away camera would see it:
    shrunk to ~px wide, JPEG-compressed, then aligned from that small image."""
    px = px or config.LOWRES_PX
    scale = px / max(face_width(face), 1)
    if scale >= 1:
        return embed(img, face).copy()
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    f = Face(bbox=face.bbox * scale, kps=face.kps * scale, det_score=face.det_score)
    return embed(small, f).copy()


def face_width(face) -> int:
    x1, _, x2, _ = face.bbox
    return int(x2 - x1)


class Gallery:
    """Enrolled students: one averaged, normalised embedding per person."""

    def __init__(self, path=None):
        path = path or config.EMBEDDINGS_FILE          # read at call time, not import time
        if not path.exists():
            raise SystemExit("embeddings.npz not found. Run capture.py and enroll.py first.")
        data = np.load(path, allow_pickle=False)
        built_with = str(data["model"]) if "model" in data else "?"
        if built_with != config.MODEL_NAME:
            raise SystemExit(f"embeddings.npz was built with '{built_with}' but config uses "
                             f"'{config.MODEL_NAME}'. Run: python enroll.py")
        self.names = [str(n) for n in data["names"]]
        self.centroids = data["centroids"].astype(np.float32)
        self.lowres = (data["centroids_lr"].astype(np.float32)
                       if "centroids_lr" in data and config.USE_LOWRES_GALLERY else None)

    def match(self, emb, px=None):
        """Return (name, similarity). name is 'Unknown' if not confident."""
        sims = self.centroids @ emb
        if self.lowres is not None and px is not None and px < config.LOWRES_BELOW_PX:
            sims = np.maximum(sims, self.lowres @ emb)
        order = np.argsort(sims)[::-1]
        best = float(sims[order[0]])
        second = float(sims[order[1]]) if len(order) > 1 else -1.0
        if best < config.MATCH_THRESHOLD or best - second < config.MARGIN:
            return "Unknown", best
        return self.names[order[0]], best
