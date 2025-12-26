#!/usr/bin/env python3
"""
learning_faces.py

Train a CNN face classifier using images stored as:
  dataset/<person_name>/*.jpg

Outputs
- trained_models/face_model_<HxW>.keras
- trained_models/labels_<HxW>.json

Example
  python3 learning_faces.py --dataset-dir dataset --img-size 128 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: Path
    model_dir: Path
    img_height: int
    img_width: int
    batch_size: int
    epochs: int
    val_split: float
    seed: int
    learning_rate: float
    plot: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train face recognition classifier (Keras).")
    p.add_argument("--dataset-dir", default="dataset", help="Dataset root: dataset/<class>/*.jpg")
    p.add_argument("--model-dir", default="trained_models", help="Output folder for trained model")
    p.add_argument("--img-size", type=int, default=128, help="Square model input size (H=W)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--val-split", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--plot", action="store_true", help="Plot training curves (matplotlib required)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(message)s")


def build_datasets(cfg: TrainConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    if not cfg.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {cfg.dataset_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.dataset_dir,
        validation_split=cfg.val_split,
        subset="training",
        seed=cfg.seed,
        image_size=(cfg.img_height, cfg.img_width),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.dataset_dir,
        validation_split=cfg.val_split,
        subset="validation",
        seed=cfg.seed,
        image_size=(cfg.img_height, cfg.img_width),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    class_names = list(train_ds.class_names)
    logging.info("Classes: %s", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(cfg: TrainConfig, num_classes: int) -> keras.Model:
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(cfg.img_height, cfg.img_width, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Dropout(0.30)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="face_classifier")
    opt = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_labels(model_dir: Path, name: str, labels: list[str]) -> Path:
    out = model_dir / name
    out.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")
    return out


def maybe_plot(history: keras.callbacks.History) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("Plot requested but matplotlib not available: %s", e)
        return

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Acc")
    plt.plot(epochs_range, val_acc, label="Val Acc")
    plt.legend(loc="lower right")
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")
    plt.tight_layout()
    plt.show()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    img = int(args.img_size)
    cfg = TrainConfig(
        dataset_dir=Path(args.dataset_dir),
        model_dir=Path(args.model_dir),
        img_height=img,
        img_width=img,
        batch_size=max(1, int(args.batch_size)),
        epochs=max(1, int(args.epochs)),
        val_split=float(args.val_split),
        seed=int(args.seed),
        learning_rate=float(args.lr),
        plot=bool(args.plot),
    )

    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(cfg)
    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes (two folders under dataset/) to train a classifier.")

    model = build_model(cfg, num_classes)
    model.summary(print_fn=logging.info)

    model_name = f"face_model_{cfg.img_height}x{cfg.img_width}.keras"
    labels_name = f"labels_{cfg.img_height}x{cfg.img_width}.json"
    ckpt_path = cfg.model_dir / f"checkpoint_{cfg.img_height}x{cfg.img_width}.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
    )

    model_path = cfg.model_dir / model_name
    model.save(model_path)
    labels_path = save_labels(cfg.model_dir, labels_name, class_names)

    logging.info("Saved model:  %s", model_path)
    logging.info("Saved labels: %s", labels_path)
    logging.info("Run: recognise_faces.py --model %s --labels %s", model_path, labels_path)

    if cfg.plot:
        maybe_plot(history)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
