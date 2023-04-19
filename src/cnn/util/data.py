"""
Dataset utilities.
"""

import typing as t
import logging
from pathlib import Path
from kaggle import api
import pandas as pd
import numpy as np
import json
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
# from keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelBinarizer


def download_file(path: Path):
    api.dataset_download_cli(
        "validmodel/indo-fashion-dataset",
        path=path,
        unzip=True)


def load_dataset(path: Path, image_size=32, batch_size=32) -> t.Tuple[
            tf.data.Dataset,
            tf.data.Dataset,
            tf.data.Dataset,
            t.List]:
    """Load the kaggle dataset and return the test and train data."""
    def convert_image_path(image_path):
        return str(path / image_path)

    def class_label_to_bin(meta: pd.DataFrame, lb: LabelBinarizer):
        class_labels = lb.transform(meta["class_label"])
        return np.array(class_labels, dtype=np.uint8)

    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float16)  # maybe float32? i don't see why that would be necessary
        return img / 255.0  # type: ignore

    def process_path(file_paths):
        images = []
        for file_path in file_paths:
            image = tf.io.read_file(file_path)
            image = decode_img(image)  # type: ignore
            images.append(image)
        return images

    def configure_for_performance(
            ds: tf.data.Dataset,
            shuffle: bool = False,
            augment: bool = False) -> tf.data.Dataset:
        if shuffle:
            logging.info("Shuffling dataset")
            ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        if augment:
            logging.info("Augmenting dataset")
            augment_images = tf.keras.Sequential([
                # Rescaling(1. / 255),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.3),
            ])
            ds = ds.map(
                lambda x, y: (augment_images(x, training=True), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        ds = ds.batch(
            batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    if all((p.exists() for p in [
            path / f"{what}_{image_size}_data.tfrecord" for what in (
                "train",
                "test",
                "val")])):
        logging.info("Loading tfrecord dataset")
        train_data = tf.data.Dataset.load(
            str(path / f"train_{image_size}_data.tfrecord"))
        print(train_data)
        test_data = tf.data.Dataset.load(
            str(path / f"test_{image_size}_data.tfrecord"))
        val_data = tf.data.Dataset.load(
            str(path / f"val_{image_size}_data.tfrecord"))
        classes = np.load(path / "classes.npy")

        return (
            configure_for_performance(train_data, shuffle=True, augment=True),
            configure_for_performance(test_data),
            configure_for_performance(val_data),
            classes)
    else:
        logging.info("No tfrecord dataset found.")

    image_count = len(list(path.glob('images/*/*.jpg')))
    logging.info(f"Found {image_count} images")
    logging.info("Loading metadata")
    train_path = path / "train_data.json"
    test_path = path / "test_data.json"
    val_path = path / "val_data.json"

    train_meta = []
    with open(train_path, "r") as f:
        for line in f:
            train_meta.append(json.loads(line))

    test_meta = []
    with open(test_path, "r") as f:
        for line in f:
            test_meta.append(json.loads(line))

    val_meta = []
    with open(val_path, "r") as f:
        for line in f:
            val_meta.append(json.loads(line))

    train_meta = pd.DataFrame(train_meta)
    test_meta = pd.DataFrame(test_meta)
    val_meta = pd.DataFrame(val_meta)

    train_meta["image_path"] = train_meta["image_path"].apply(
        convert_image_path)
    test_meta["image_path"] = test_meta["image_path"].apply(
        convert_image_path)
    val_meta["image_path"] = val_meta["image_path"].apply(
        convert_image_path)
    # get all labels in a dictionary
    labels = val_meta["class_label"].unique()
    lb = LabelBinarizer()
    lb.fit(labels)

    # label_dict = dict(zip(labels, lb.transform(labels)))
    train_labels = class_label_to_bin(train_meta, lb)
    test_labels = class_label_to_bin(test_meta, lb)
    val_labels = class_label_to_bin(val_meta, lb)

    logging.info("Loaded metadata")

    logging.info("Preparing train data")
    train_data = tf.data.Dataset.from_tensor_slices((
        process_path(train_meta["image_path"]), train_labels))

    logging.info("Preparing test data")
    test_data = tf.data.Dataset.from_tensor_slices((
        process_path(test_meta["image_path"]), test_labels))
    logging.info("Preparing validation data")
    val_data = tf.data.Dataset.from_tensor_slices((
        process_path(val_meta["image_path"]), val_labels))
    logging.info("Prepared data")
    train_data.save(str(path / f"train_{image_size}_data.tfrecord"))
    train_data = configure_for_performance(train_data)
    test_data.save(str(path / f"test_{image_size}_data.tfrecord"))
    test_data = configure_for_performance(test_data)
    val_data.save(str(path / f"val_{image_size}_data.tfrecord"))
    val_data = configure_for_performance(val_data)

    np.save(path / "classes.npy", lb.classes_)

    return (train_data,
            test_data,
            val_data,
            lb.classes_.tolist())  # type: ignore
