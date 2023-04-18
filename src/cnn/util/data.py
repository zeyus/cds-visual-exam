"""
Dataset utilities.
"""

from pathlib import Path
from kaggle import api
import pandas as pd
import json


def download_file(path: Path):
    api.dataset_download_cli(
        "validmodel/indo-fashion-dataset",
        path=path,
        unzip=True)


def load_dataset(path: Path):
    """Load the kaggle dataset and return the test and train data."""
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

    return (pd.DataFrame(train_data),
            pd.DataFrame(test_data),
            pd.DataFrame(val_data))
