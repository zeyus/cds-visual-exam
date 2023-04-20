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
import random
# from keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
# from keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelBinarizer


def download_file(path: Path):
    api.dataset_download_cli(
        "validmodel/indo-fashion-dataset",
        path=path,
        unzip=True)


def load_dataset(path: Path, input_shape=(32, 32, 3), batch_size=32, parallel_procs=None) -> t.Tuple[
            tf.data.Dataset,
            tf.data.Dataset,
            tf.data.Dataset,
            t.List]:
    """Load the kaggle dataset and return the test and train data."""

    augment_images = tf.keras.Sequential([
                # Rescaling(1. / 255),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
            ])

    def convert_image_path(image_path):
        return str(path / image_path)

    def class_label_to_bin(meta: pd.DataFrame, lb: LabelBinarizer):
        class_labels = lb.transform(meta["class_label"])
        return np.array(class_labels, dtype=np.uint8)

    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [input_shape[0], input_shape[1]])
        # fill with white and center the image
        w = tf.shape(img)[0]
        h = tf.shape(img)[1]
        pad_w = (input_shape[0] - w) // 2
        pad_h = (input_shape[1] - h) // 2
        img = tf.image.pad_to_bounding_box(
            img, pad_h, pad_w, input_shape[0], input_shape[1])
        img = tf.cast(img, tf.float16)
        return img / 255.0  # type: ignore

    def process_path(file_paths):
        images = []
        for file_path in file_paths:
            image = tf.io.read_file(file_path)
            image = decode_img(image)  # type: ignore
            images.append(image)
        return images

    def process_path_gen(paths, labels, augment=False):
        for file_path, label in zip(paths, labels):
            image = tf.io.read_file(file_path)
            image = decode_img(image)
            if augment:
                image = augment_images(image)
            yield image, label

    def configure_for_performance(
            ds: tf.data.Dataset,
            shuffle: bool = False) -> tf.data.Dataset:
        if shuffle:
            logging.info("Shuffling dataset")
            ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        ds = ds.cache()
        ds = ds.batch(
            batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    labels = (
        'blouse',
        'dhoti_pants',
        'dupattas',
        'gowns',
        'kurta_men',
        'leggings_and_salwars',
        'lehenga',
        'mojaris_men',
        'mojaris_women',
        'nehru_jackets',
        'palazzos',
        'petticoats',
        'saree',
        'sherwanis',
        'women_kurta')

    lb = LabelBinarizer()
    lb.fit(labels)

    # if all((p.exists() for p in [
    #         path / f"{what}_{input_shape[0]}x{input_shape[1]}_data.tfrecord" for what in (
    #             "train",
    #             "test",
    #             "val")])):
    #     logging.info("Loading tfrecord dataset")
    #     train_data = tf.data.Dataset.load(
    #         str(path / f"train_{input_shape[0]}x{input_shape[1]}_data.tfrecord"))
    #     print(train_data)
    #     test_data = tf.data.Dataset.load(
    #         str(path / f"test_{input_shape[0]}x{input_shape[1]}_data.tfrecord"))
    #     val_data = tf.data.Dataset.load(
    #         str(path / f"val_{input_shape[0]}x{input_shape[1]}_data.tfrecord"))
    #     classes = np.load(path / "classes.npy")

    #     return (
    #         configure_for_performance(train_data, shuffle=True, augment=True),
    #         configure_for_performance(test_data),
    #         configure_for_performance(val_data),
    #         classes)
    # else:
    #     logging.info("No tfrecord dataset found.")
    image_count = len(list(path.glob('images/**/*.jp*g')))
    logging.info(f"Found {image_count} images")
    logging.info("Loading metadata")

    def create_dataset(which: str, lb: LabelBinarizer, shuffle: bool = False, augment: bool = False):
        meta_path = path / f"{which}_data.json"
        logging.info(f"Loaded metadata for {which} dataset")
        ds_meta = []
        with open(meta_path, "r") as f:
            for line in f:
                ds_meta.append(json.loads(line))

        ds_meta = pd.DataFrame(ds_meta)

        ds_meta["image_path"] = ds_meta["image_path"].apply(
            convert_image_path)

        logging.info(f"Preparing class labels for {which} dataset")

        ds_labels = class_label_to_bin(ds_meta, lb)

        logging.info(f"Preparing {which} data (generator)...")
        # ds_data = tf.data.Dataset.from_tensor_slices((
        #     process_path(ds_meta["image_path"]), ds_labels))
        # preshuffle the data + labels for fun and profit
        logging.info(f"Preshuffle {which} data to avoid class groups")
        paths_and_lables = list(zip(ds_meta["image_path"], ds_labels))
        random.shuffle(paths_and_lables)
        ds_paths, ds_labels = zip(*paths_and_lables)
        ds_data = tf.data.Dataset.from_generator(
            process_path_gen,
            output_types=(tf.float16, tf.uint8),
            output_shapes=([input_shape[0], input_shape[1], input_shape[2]], [len(labels)]),
            args=(ds_paths, ds_labels, augment))
        # ds_data.save(str(path / f"{which}_{input_shape[0]}x{input_shape[1]}_data.tfrecord"))
        logging.info(f"Optimizing {which} data for performance")
        ds_data = configure_for_performance(
            ds_data,
            shuffle=shuffle)
        logging.info(f"Done preparing {which} data")

        return ds_data

    train_data = create_dataset(
        "train",
        lb=lb,
        shuffle=True,
        augment=True)
    test_data = create_dataset("test", lb=lb)
    val_data = create_dataset("val", lb=lb)

    return (train_data,
            test_data,
            val_data,
            lb.classes_.tolist())  # type: ignore
