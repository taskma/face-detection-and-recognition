#!/usr/bin/env python3
"""
recognise_faces.py

Realtime face recognition using:
- OpenCV Haar Cascade for face detection
- Trained Keras classifier for identity prediction

Example
  python3 recognise_faces.py --model trained_models/face_model_128x128.keras --labels trained_models/labels_128x128.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    labels_path: Path
    cascade_path: Path
    camera_index: int
    img_height: int
    img_width: int
    min_confidence: float
    draw_all_scores: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime face recognition (OpenCV + Keras).")
    p.add_argument("--model", required=True, help="Path to .keras model")
    p.add_argument("--labels", required=True, help="Path to labels JSON (from learning_faces.py)")
    p.add_argument("--cascade", default="Resources/haarcascade_frontalface_default.xml", help="Haar cascade path")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--img-size", type=int, default=128, help="Model input size (square, must match trained model)")
    p.add_argument("--min-confidence", type=float, default=0.50, help="Min confidence to accept a match (0..1)")
    p.add_argument("--draw-all-scores", action="store_true", help="Overlay top-3 scores next to the face box")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(message)s")


def load_labels(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels")
    if not isinstance(labels, list) or not labels:
        raise RuntimeError(f"Invalid labels file: {path}")
    return [str(x) for x in labels]


def preprocess_face_bgr(face_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    # Training loader yields RGB; OpenCV reads BGR. Convert for consistency.
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = resized.astype("float32") / 255.0
    return np.expand_dims(x, axis=0)  # (1, H, W, 3)


def detect_faces(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    return list(faces) if faces is not None else []


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    img = int(args.img_size)
    cfg = AppConfig(
        model_path=Path(args.model),
        labels_path=Path(args.labels),
        cascade_path=Path(args.cascade),
        camera_index=int(args.camera),
        img_height=img,
        img_width=img,
        min_confidence=float(args.min_confidence),
        draw_all_scores=bool(args.draw_all_scores),
    )

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")
    if not cfg.labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {cfg.labels_path}")
    if not cfg.cascade_path.exists():
        raise FileNotFoundError(f"Cascade not found: {cfg.cascade_path}")

    labels = load_labels(cfg.labels_path)
    model = load_model(cfg.model_path)

    cascade = cv2.CascadeClassifier(str(cfg.cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cfg.cascade_path}")

    cam = cv2.VideoCapture(cfg.camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera index {cfg.camera_index}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    logging.info("Loaded model: %s", cfg.model_path.name)
    logging.info("Labels: %s", labels)
    logging.info("Press ESC to exit.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                logging.warning("Camera read failed; retrying...")
                continue

            faces = detect_faces(frame, cascade)

            for (x, y, w, h) in faces:
                margin = int(0.12 * max(w, h))
                x0 = max(0, x - margin)
                y0 = max(0, y - margin)
                x1 = min(frame.shape[1], x + w + margin)
                y1 = min(frame.shape[0], y + h + margin)

                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                x_in = preprocess_face_bgr(crop, cfg.img_height, cfg.img_width)
                probs = model.predict(x_in, verbose=0)[0]  # softmax probs
                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])

                if best_prob >= cfg.min_confidence and best_idx < len(labels):
                    name = labels[best_idx]
                    color = (0, 255, 0)
                else:
                    name = "unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, name, (x0 + 5, max(25, y0 - 10)), font, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"{best_prob*100:.0f}%", (x0 + 5, y1 - 8), font, 0.6, (255, 255, 0), 2)

                if cfg.draw_all_scores:
                    top = np.argsort(-probs)[: min(3, len(labels))]
                    yline = y0 + 20
                    for idx in top:
                        if idx >= len(labels):
                            continue
                        txt = f"{labels[int(idx)]}: {probs[int(idx)]*100:.0f}%"
                        cv2.putText(frame, txt, (x1 + 10, yline), font, 0.5, (200, 200, 200), 1)
                        yline += 18

            cv2.imshow("recognise_faces", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        return 0
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
