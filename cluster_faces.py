"""Group an unlabeled photo dump (e.g. an iCloud album) by person.

python cluster_faces.py raw_icloud/

Finds every face, groups same-person faces together, and writes:
  candidates/person_01/ ... face crops (with margin, so enroll.py can re-detect)
  candidates/overview.jpg   one row of sample faces per group
Then rename each candidates/person_XX folder to the student's name and move it
into dataset/  (or run: python cluster_faces.py --apply "01=Rahul,02=Priya,...").
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import config
from face_engine import detect, embed, face_width

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".avi", ".3gp"}
OUT = config.ROOT / "candidates"
MAX_SIDE = 1920          # phone photos are huge; this is plenty for faces


def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def iter_frames(root):
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        try:
            if ext in IMG_EXT:
                yield path.name, load_image(path)
            elif ext in VID_EXT:
                cap = cv2.VideoCapture(str(path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                for k, i in enumerate(range(0, total, max(1, total // 15))):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, f = cap.read()
                    if ok:
                        yield f"{path.name}#{k}", f
                cap.release()
        except Exception as e:  # corrupt / unsupported file
            print(f"  ! skip {path.name}: {e}")


def crop(img, bbox, pad=0.6):
    x1, y1, x2, y2 = bbox.astype(int)
    p = int((x2 - x1) * pad)
    h, w = img.shape[:2]
    return img[max(0, y1 - p):min(h, y2 + p), max(0, x1 - p):min(w, x2 + p)]


def build(root):
    from sklearn.cluster import AgglomerativeClustering

    embs, crops, srcs = [], [], []
    n_files = 0
    for name, img in iter_frames(root):
        n_files += 1
        for f in detect(img):
            if face_width(f) < config.MIN_FACE_PX or f.det_score < 0.6:
                continue
            embs.append(embed(img, f))
            crops.append(crop(img, f.bbox))
            srcs.append(name)
        if n_files % 20 == 0:
            print(f"  {n_files} files, {len(embs)} faces")
    print(f"{n_files} files -> {len(embs)} usable faces")
    if len(embs) < 2:
        raise SystemExit("Not enough faces found.")

    X = np.stack(embs)
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=0.6).fit_predict(X)
    groups = sorted(set(labels), key=lambda l: -(labels == l).sum())

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    rows, small = [], []
    for gi, l in enumerate(groups, 1):
        idx = np.where(labels == l)[0]
        folder = OUT / (f"person_{gi:02d}" if len(idx) >= 3 else "_rare")
        folder.mkdir(exist_ok=True)
        for k, i in enumerate(idx):
            cv2.imwrite(str(folder / f"{gi:02d}_{k:03d}.jpg"), crops[i])
        if len(idx) < 3:
            small.append(len(idx))
            continue
        thumbs = [cv2.resize(crops[i], (110, 110)) for i in idx[:10]]
        thumbs += [np.zeros((110, 110, 3), np.uint8)] * (10 - len(thumbs))
        label = np.full((110, 170, 3), 255, np.uint8)
        cv2.putText(label, f"{gi:02d}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        cv2.putText(label, f"{len(idx)} faces", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        rows.append(np.hstack([label] + thumbs))
        print(f"  person_{gi:02d}: {len(idx)} faces")
    if small:
        print(f"  _rare: {sum(small)} faces in {len(small)} tiny groups (strangers / bad shots)")
    cv2.imwrite(str(OUT / "overview.jpg"), np.vstack(rows))
    print(f"\nOpen {OUT / 'overview.jpg'} and tell me which number is who.")


def apply(mapping):
    """'01=Rahul,02=Priya' -> copy candidates/person_01 into dataset/Rahul, etc."""
    for part in mapping.split(","):
        num, name = (s.strip() for s in part.split("="))
        src = OUT / f"person_{int(num):02d}"
        dst = config.DATASET_DIR / name
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in src.glob("*.jpg"):
            shutil.copy(f, dst / f"icloud_{f.name}")
            n += 1
        print(f"  person_{int(num):02d} -> dataset/{name} ({n} photos)")
    print("Now run: python enroll.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", help="folder with the photo dump")
    ap.add_argument("--apply", help='e.g. "01=Rahul,02=Priya" (same person twice is fine)')
    a = ap.parse_args()
    if a.apply:
        apply(a.apply)
    elif a.root:
        build(a.root)
    else:
        ap.error("give a folder, or --apply")
