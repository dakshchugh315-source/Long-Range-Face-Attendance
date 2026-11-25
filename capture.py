"""Collect face photos for one student.

Webcam:   python capture.py --name Rahul
Video:    python capture.py --name Rahul --video rahul.mp4

Webcam mode saves a photo automatically every few frames while a face is visible.
Ask the person to slowly turn left, right, up, down, move closer and farther.
Press Q to stop early.
"""
import argparse
import time

import cv2

import config
from face_engine import detect, face_width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="student name (folder name)")
    ap.add_argument("--video", help="optional video file instead of webcam")
    ap.add_argument("--count", type=int, default=40, help="photos to save")
    ap.add_argument("--every", type=int, default=5, help="save every Nth frame with a face")
    args = ap.parse_args()

    out_dir = config.DATASET_DIR / args.name.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    start_idx = len(list(out_dir.glob("*.jpg")))

    src = args.video if args.video else config.CAMERA_SOURCE
    from attend import open_camera
    cap = cv2.VideoCapture(src) if args.video else open_camera(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/video: {src}")

    saved, frame_no = 0, 0
    print(f"Saving to {out_dir} ... move your head slowly. Q = quit")
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        faces = detect(frame)
        view = frame.copy()

        if len(faces) == 1:
            f = faces[0]
            x1, y1, x2, y2 = f.bbox.astype(int)
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if frame_no % args.every == 0 and face_width(f) >= config.MIN_FACE_PX:
                # save the full frame; enroll.py re-detects and aligns the face
                cv2.imwrite(str(out_dir / f"{start_idx + saved:03d}.jpg"), frame)
                saved += 1
        elif len(faces) > 1:
            cv2.putText(view, "Only ONE person in frame please", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(view, f"{args.name}: {saved}/{args.count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Capture", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if args.video is None:
            time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} photos for {args.name}. Now run: python enroll.py")


if __name__ == "__main__":
    main()
