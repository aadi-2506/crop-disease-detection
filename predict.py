from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from utils.disease_info import get_prediction_details
from utils.preprocessing import IMAGE_SIZE


PROJECT_DIR = Path(__file__).resolve().parent


def predict(image_path: Path, model_dir: Path) -> dict:
    model_path = model_dir / "crop_disease_model.keras"
    results_path = model_dir / "evaluation_results.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run `python train.py` first.")
    if not results_path.is_file():
        raise FileNotFoundError(f"Model class labels not found: {results_path}. Run `python train.py` first.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Could not open a valid JPG, JPEG or PNG image: {image_path}") from exc

    with results_path.open("r", encoding="utf-8") as results_file:
        metadata = json.load(results_file)
    class_names = metadata.get("class_names", [])
    if not class_names:
        raise ValueError(f"No class labels were found in {results_path}. Retrain the model.")
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0), verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise ValueError("Model output does not match the saved class labels. Retrain the model.")
    best_index = int(np.argmax(probabilities))
    crop, disease, status = get_prediction_details(class_names[best_index])
    result = {
        "crop": crop,
        "disease": disease,
        "confidence": float(probabilities[best_index]),
        "status": status,
        "class_name": class_names[best_index],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the crop disease in a leaf image.")
    parser.add_argument("image", type=Path, help="Path to a JPG, JPEG or PNG image.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "models")
    args = parser.parse_args()
    result = predict(args.image.resolve(), args.model_dir.resolve())
    print(f"Crop: {result['crop']}")
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Status: {'Healthy' if result['status'] == 'healthy' else 'Diseased'}")


if __name__ == "__main__":
    main()
