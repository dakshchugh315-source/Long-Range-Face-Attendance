"""Build embeddings.npz from dataset/<name>/*.jpg.

No model training from scratch: pretrained ArcFace turns every photo into a
512-number face fingerprint, and we average them per student.
Run again whenever you add or change students.
"""
import cv2
import numpy as np

import config
from face_engine import detect, embed

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".webm"}
FRAMES_PER_VIDEO = 40
MAX_SIDE = 1920          # 12 MP phone photos are shrunk first: same faces, ~6x faster


def _load_photo(path):
    """Any photo incl. iPhone HEIC, rotated upright (EXIF), as BGR."""
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def read_images(folder):
    """Yield images from photos AND phone videos in a student's folder."""
    for path in sorted(folder.iterdir()):
        ext = path.suffix.lower()
        if ext in IMG_EXT:
            try:
                img = _load_photo(path)
            except Exception as e:
                print(f"  ! {path.name}: can't read ({e})")
                continue
            yield img
        elif ext in VID_EXT:
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            step = max(1, total // FRAMES_PER_VIDEO)
            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()


def main():
    people = sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()) \
        if config.DATASET_DIR.exists() else []
    if not people:
        raise SystemExit("No students in dataset/. Run capture.py first.")

    names, centroids = [], []
    for person in people:
        embs, skipped_multi = [], 0
        for img in read_images(person):
            faces = detect(img, tiles=False)       # enrolment faces are large; tiles only cost time
            if not faces:
                continue
            # Several faces: use the biggest only if it clearly dominates,
            # otherwise we can't know which one is this student -> skip frame.
            area = lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            faces.sort(key=area, reverse=True)
            if len(faces) > 1 and area(faces[1]) > 0.4 * area(faces[0]):
                skipped_multi += 1
                continue
            embs.append(embed(img, faces[0]))

        if skipped_multi:
            print(f"  - {person.name}: skipped {skipped_multi} frames with several people in them")
        if len(embs) < 5:
            print(f"  ! {person.name}: only {len(embs)} usable photos, skipped (need 5+)")
            continue
        embs = np.stack(embs)
        # Drop outliers (someone else in the background, bad crop): keep only
        # embeddings close to the medoid = the most "typical" photo of this person.
        medoid = embs[np.argmax((embs @ embs.T).mean(axis=1))]
        keep = embs @ medoid >= 0.45
        if (~keep).any():
            print(f"  - {person.name}: dropped {int((~keep).sum())} photos that don't look like the rest")
        embs = embs[keep]
        if len(embs) < 5:
            print(f"  ! {person.name}: only {len(embs)} good photos left, skipped (need 5+)")
            continue
        c = embs.mean(axis=0)
        c /= np.linalg.norm(c)
        consistency = float((embs @ c).mean())
        names.append(person.name)
        centroids.append(c)
        print(f"  {person.name}: {len(embs)} photos, consistency {consistency:.2f}")

    if not names:
        raise SystemExit("Nobody enrolled.")
    centroids = np.stack(centroids).astype(np.float32)
    np.savez(config.EMBEDDINGS_FILE, names=np.array(names), centroids=centroids,
             model=np.array(config.MODEL_NAME))

    # warn if two students look too similar to the model
    sims = centroids @ centroids.T
    np.fill_diagonal(sims, -1)
    i, j = np.unravel_index(np.argmax(sims), sims.shape)
    if len(names) > 1:
        print(f"Most similar pair: {names[i]} & {names[j]} = {sims[i, j]:.2f}")
        for a, b in zip(*np.where(np.triu(sims, 1) > 0.5)):
            print(f"  !!! WARNING: {names[a]} & {names[b]} look like the SAME person "
                  f"({sims[a, b]:.2f}). Check their dataset folders for mixed-up photos.")
    print(f"Enrolled {len(names)} students -> {config.EMBEDDINGS_FILE.name}")


if __name__ == "__main__":
    main()
