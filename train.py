from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from utils.preprocessing import IMAGE_SIZE, collect_dataset, make_model


PROJECT_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def split_dataset(
    samples: list[tuple[Path, int]], class_names: list[str], seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    by_class: dict[int, list[Path]] = {label: [] for label in range(len(class_names))}
    for path, label in samples:
        by_class[label].append(path)

    train_samples: list[tuple[Path, int]] = []
    validation_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []
    for label, class_name in enumerate(class_names):
        files = sorted(by_class[label])
        if len(files) < 3:
            raise ValueError(
                f"Class '{class_name}' has only {len(files)} image(s). "
                "At least 3 images per class are required for train/validation/test splits."
            )
        rng.shuffle(files)
        test_count = max(1, round(len(files) * 0.15))
        validation_count = max(1, round(len(files) * 0.15))
        if test_count + validation_count >= len(files):
            test_count = validation_count = 1
        test_files = files[:test_count]
        validation_files = files[test_count : test_count + validation_count]
        train_files = files[test_count + validation_count :]
        train_samples.extend((path, label) for path in train_files)
        validation_samples.extend((path, label) for path in validation_files)
        test_samples.extend((path, label) for path in test_files)
    rng.shuffle(train_samples)
    rng.shuffle(validation_samples)
    rng.shuffle(test_samples)
    return train_samples, validation_samples, test_samples


def make_tf_dataset(
    samples: list[tuple[Path, int]], batch_size: int, training: bool
) -> tf.data.Dataset:
    paths = [str(path) for path, _ in samples]
    labels = [label for _, label in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(samples), 10_000), seed=42, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def evaluate_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
) -> dict:
    true_labels = np.concatenate([labels.numpy() for _, labels in dataset])
    probabilities = model.predict(dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=np.arange(len(class_names))
        ).tolist(),
    }


def train(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_dir.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}. Download and extract PlantVillage "
            "so that each disease class has its own folder under this directory."
        )
    samples, class_names = collect_dataset(dataset_root, SUPPORTED_EXTENSIONS)
    train_samples, validation_samples, test_samples = split_dataset(samples, class_names, args.seed)
    print(
        f"Detected {len(class_names)} classes · "
        f"{len(train_samples)} training · {len(validation_samples)} validation · "
        f"{len(test_samples)} test images"
    )
    print("Classes:", ", ".join(class_names))

    tf.keras.utils.set_random_seed(args.seed)
    model = make_model(len(class_names))
    model.summary()
    train_ds = make_tf_dataset(train_samples, args.batch_size, training=True)
    validation_ds = make_tf_dataset(validation_samples, args.batch_size, training=False)
    test_ds = make_tf_dataset(test_samples, args.batch_size, training=False)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "crop_disease_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1
        ),
    ]
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model = tf.keras.models.load_model(model_path)
    metrics = evaluate_predictions(model, test_ds, class_names)
    manifest = [
        {"path": path.relative_to(dataset_root).as_posix(), "label": label}
        for path, label in test_samples
    ]
    with (args.model_dir / "test_manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    results = {
        "class_names": class_names,
        "metrics": metrics,
        "test_samples": len(test_samples),
        "image_size": IMAGE_SIZE,
        "dataset_dir": str(dataset_root),
        "seed": args.seed,
    }
    with (args.model_dir / "evaluation_results.json").open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
    print(f"\nSaved best model to: {model_path}")
    print(f"Saved held-out test results to: {args.model_dir / 'evaluation_results.json'}")
    print(
        "Test metrics: "
        + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items() if name != "confusion_matrix")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on a class-folder leaf image dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must both be positive.")
    args.dataset_dir = args.dataset_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    return args


if __name__ == "__main__":
    train(parse_args())
