from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train import make_tf_dataset
from utils.preprocessing import IMAGE_SIZE


PROJECT_DIR = Path(__file__).resolve().parent


def evaluate(model_dir: Path, dataset_dir: Path, batch_size: int) -> dict:
    model_path = model_dir / "crop_disease_model.keras"
    manifest_path = model_dir / "test_manifest.json"
    results_path = model_dir / "evaluation_results.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run `python train.py` first.")
    if not manifest_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            f"Test split metadata not found in {model_dir}. Run `python train.py` to create it."
        )
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    with results_path.open("r", encoding="utf-8") as results_file:
        previous_results = json.load(results_file)
    class_names = previous_results["class_names"]
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    samples: list[tuple[Path, int]] = []
    for item in manifest:
        image_path = dataset_dir / Path(item["path"])
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Test image is missing: {image_path}. Keep the extracted dataset unchanged after training."
            )
        samples.append((image_path, int(item["label"])))
    test_ds = make_tf_dataset(samples, batch_size, training=False)
    model = tf.keras.models.load_model(model_path)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=np.arange(len(class_names))
        ).tolist(),
    }
    previous_results.update(
        {"metrics": metrics, "test_samples": len(samples), "image_size": IMAGE_SIZE}
    )
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(previous_results, results_file, indent=2)

    print("\nHeld-out test metrics")
    for name in ("accuracy", "precision", "recall", "f1_score"):
        print(f"{name.title().replace('_', ' ')}: {metrics[name]:.4f}")
    print(
        "\nPer-class report:\n",
        classification_report(
            true_labels,
            predicted_labels,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        ),
    )
    print(f"Confusion matrix and metrics saved to: {results_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained model on its saved held-out test split.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "models")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    return args


if __name__ == "__main__":
    options = parse_args()
    evaluate(options.model_dir.resolve(), options.dataset_dir.resolve(), options.batch_size)
