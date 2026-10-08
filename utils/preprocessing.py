from __future__ import annotations

from pathlib import Path

import tensorflow as tf


IMAGE_SIZE = 224
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_dataset(
    dataset_root: Path, extensions: set[str] | None = None
) -> tuple[list[tuple[Path, int]], list[str]]:
    """Find class folders anywhere below dataset_root and assign sorted labels."""
    allowed = extensions or SUPPORTED_IMAGE_EXTENSIONS
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}. Extract the PlantVillage class folders there."
        )
    class_directories = sorted(
        directory
        for directory in dataset_root.rglob("*")
        if directory.is_dir()
        and any(
            file_path.is_file() and file_path.suffix.lower() in allowed
            for file_path in directory.iterdir()
        )
    )
    if not class_directories:
        raise ValueError(
            f"No class folders containing JPG, JPEG or PNG images were found under {dataset_root}. "
            "Expected folders such as Apple___Apple_scab/ and Tomato___healthy/."
        )

    class_names = [directory.name for directory in class_directories]
    samples: list[tuple[Path, int]] = []
    for label, directory in enumerate(class_directories):
        image_paths = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed
        )
        samples.extend((path, label) for path in image_paths)
    if not samples:
        raise ValueError(f"No supported images were found in dataset class folders under {dataset_root}.")
    return samples, class_names


def make_model(number_of_classes: int) -> tf.keras.Model:
    if number_of_classes < 2:
        raise ValueError("At least two dataset classes are required for classification.")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="leaf_image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenetv2_preprocessing")(x)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="feature_dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="crop_disease_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
