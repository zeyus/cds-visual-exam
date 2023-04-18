"""
Dataset utilities.
"""

from pathlib import Path
from kaggle import api
import pandas as pd
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import preprocess_input


def download_file(path: Path):
    api.dataset_download_cli(
        "validmodel/indo-fashion-dataset",
        path=path,
        unzip=True)


def load_dataset(path: Path, image_size = 32, batch_size = 32):
    """Load the kaggle dataset and return the test and train data."""
    def convert_image_path(image_path):
        base_dir = path
        return str(base_dir / image_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    train_path = path / "train_data.json"
    test_path = path / "test_data.json"
    val_path = path / "val_data.json"

    train_data = []
    with open(train_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))

    test_data = []
    with open(test_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))

    val_data = []
    with open(val_path, "r") as f:
        for line in f:
            val_data.append(json.loads(line))

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    val_data = pd.DataFrame(val_data)

    train_data["image_path"] = train_data["image_path"].apply(
        convert_image_path)
    test_data["image_path"] = test_data["image_path"].apply(
        convert_image_path)
    val_data["image_path"] = val_data["image_path"].apply(
        convert_image_path)

    train_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    val_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    test_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_data = train_generator.flow_from_dataframe(
        train_data,
        x_col="image_path",
        y_col="class_label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        color_mode="rgb",
        subset="training",
    )

    test_data = test_generator.flow_from_dataframe(
        test_data,
        x_col="image_path",
        y_col="class_label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        color_mode="rgb",
    )

    val_data = val_generator.flow_from_dataframe(
        val_data,
        x_col="image_path",
        y_col="class_label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        color_mode="rgb",
    )

    return train_data, test_data, val_data
