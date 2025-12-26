#!/usr/bin/env python3
"""
capturing_faces.py

Capture face crops from a webcam and store them under:
  dataset/<person_name>/*.jpg

Notes
- Uses OpenCV Haar Cascade for face detection.
- Saves *cropped face* images (color) for cleaner training data.
- No pairing/IDs required: you can pass a person name directly.

Example
  python3 capturing_faces.py --person "Yigit" --num-images 150
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2


@dataclass(frozen=True)
class CaptureConfig:
    dataset_dir: Path
    person: str
    num_images: int
    camera_index: int
    cascade_path: Path
    warmup_seconds: float
    interval_ms: int
    min_face_size: int
    margin: float
    preview: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture face crops into dataset/<person>/")
    p.add_argument("--dataset-dir", default="dataset", help="Root dataset folder")
    p.add_argument("--person", required=True, help="Person name (folder will be created under dataset/)")
    p.add_argument("--num-images", type=int, default=100, help="How many face images to capture")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument(
        "--cascade",
        default="Resources/haarcascade_frontalface_default.xml",
        help="Path to Haar cascade XML",
    )
    p.add_argument("--warmup", type=float, default=2.0, help="Warmup time in seconds before capturing")
    p.add_argument("--interval-ms", type=int, default=300, help="Delay between captures (ms)")
    p.add_argument("--min-face-size", type=int, default=60, help="Minimum face size in pixels")
    p.add_argument("--margin", type=float, default=0.15, help="Crop margin ratio around detected face")
    p.add_argument("--no-preview", action="store_true", help="Disable preview window")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(message)s")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def open_camera(index: int) -> cv2.VideoCapture:
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam


def load_cascade(path: Path) -> cv2.CascadeClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Face cascade not found: {path}")
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {path}")
    return cascade


def largest_face(faces) -> Optional[Tuple[int, int, int, int]]:
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def expand_box(x: int, y: int, w: int, h: int, margin: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    mx = int(w * margin)
    my = int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(img_w, x + w + mx)
    y1 = min(img_h, y + h + my)
    return x0, y0, x1, y1


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = CaptureConfig(
        dataset_dir=Path(args.dataset_dir),
        person=args.person.strip(),
        num_images=max(1, args.num_images),
        camera_index=args.camera,
        cascade_path=Path(args.cascade),
        warmup_seconds=max(0.0, args.warmup),
        interval_ms=max(0, args.interval_ms),
        min_face_size=max(20, args.min_face_size),
        margin=max(0.0, min(0.5, args.margin)),
        preview=not args.no_preview,
    )

    person_dir = cfg.dataset_dir / cfg.person
    ensure_dir(person_dir)

    cam = open_camera(cfg.camera_index)
    face_detector = load_cascade(cfg.cascade_path)

    logging.info("Capturing into: %s", person_dir)
    logging.info("Warmup for %.1fs. Look at the camera...", cfg.warmup_seconds)

    t0 = time.time()
    while time.time() - t0 < cfg.warmup_seconds:
        ret, frame = cam.read()
        if not ret:
            continue
        if cfg.preview:
            cv2.imshow("capture (warmup)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                logging.warning("Aborted during warmup.")
                cam.release()
                cv2.destroyAllWindows()
                return 1

    count = 0
    last_saved = 0.0

    try:
        while count < cfg.num_images:
            ret, frame = cam.read()
            if not ret:
                logging.warning("Camera read failed; retrying...")
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(cfg.min_face_size, cfg.min_face_size),
            )
            face = largest_face(faces)

            if face is not None:
                x, y, fw, fh = face
                x0, y0, x1, y1 = expand_box(x, y, fw, fh, cfg.margin, w, h)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, f"{cfg.person} {count}/{cfg.num_images}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                now = time.time()
                if (now - last_saved) * 1000 >= cfg.interval_ms:
                    face_crop = frame[y0:y1, x0:x1]
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = person_dir / f"{cfg.person}_{ts}_{count+1:04d}.jpg"
                    ok = cv2.imwrite(str(out_path), face_crop)
                    if ok:
                        count += 1
                        last_saved = now
                        logging.info("Saved %s", out_path.name)
                    else:
                        logging.warning("Failed to write image: %s", out_path)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if cfg.preview:
                cv2.imshow("capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                logging.warning("ESC pressed. Exiting.")
                break

        logging.info("Done. Captured %d images for '%s'.", count, cfg.person)
        return 0 if count > 0 else 2

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
