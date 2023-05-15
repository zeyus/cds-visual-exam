"""Data utils."""

import typing as t
import logging
from pathlib import Path
from shutil import rmtree
from sklearn.preprocessing import LabelBinarizer
import cv2
import pickle
import numpy as np


def download_and_extract_cifar10(dest: Path, force: bool = False):
    """Download the CIFAR10 dataset and extract it to the specified destination.

    Args:
        dest (Path): Destination path.
        force (bool, optional): Should the file be downloaded if the dataset exists?
                                Defaults to False.
    """

    import tarfile
    import urllib.request

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = url.split('/')[-1]
    filepath = dest / filename
    dataset_dir = dest / 'cifar-10-batches-py'

    if not filepath.exists() or force:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists():
            logging.info(f'Removing existing file at {filepath}.')
            filepath.unlink()
        logging.info(f'Downloading {url} to {filepath}.')
        urllib.request.urlretrieve(url, filepath)
        logging.info(f'Downloaded {url} to {filepath}.')
    else:
        logging.info(f'File {filepath} already exists.')

    if (dataset_dir).exists() and not force:
        logging.info(f'Dataset already extracted to {dataset_dir}.')
        logging.info('Use --force to re-download and extract.')
        return True
    elif (dataset_dir).exists() and force:
        logging.info(f'Removing existing dataset at {dataset_dir}.')
        rmtree(dataset_dir, ignore_errors=True)

    with tarfile.open(filepath, 'r:gz') as tar:
        logging.info(f'Extracting dataset to {dest}.')
        tar.extractall(dest)
        logging.info(f'Extracted dataset to {dest}.')


def load_cifar10(dest: Path) -> t.Tuple[t.Tuple[t.Any, ...], t.Tuple[t.Any, ...]]:
    """Load the CIFAR10 dataset from the specified destination.

    Args:
        dest (Path): Destination path.
        out1d (bool, optional): Should the output be 1D? Defaults to False.

    Returns:
        Tuple[Tuple[Any, ...], Tuple[Any, ...]]: Tuple of tuples containing the
                                                 training and testing data.
    """

    dataset_dir = dest / 'cifar-10-batches-py'
    train_files = ['data_batch_1',
                   'data_batch_2',
                   'data_batch_3',
                   'data_batch_4',
                   'data_batch_5']
    test_files = ['test_batch']

    x_train = []
    y_train = []
    for filename in train_files:
        with open(dataset_dir / filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            x_train.append(data[b'data'])
            y_train.append(data[b'labels'])
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test = []
    y_test = []
    for filename in test_files:
        with open(dataset_dir / filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            x_test.append(data[b'data'])
            y_test.append(data[b'labels'])
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # lb = LabelBinarizer()
    # y_train = lb.fit_transform(y_train)
    # y_test = lb.fit_transform(y_test)

    # convert x
    x_train = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in x_train])
    x_test = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in x_test])

    # rescale x, convert to float32
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # reshape x
    nsamples, nx, ny = x_train.shape
    x_train = x_train.reshape((nsamples, nx * ny))
    nsamples, nx, ny = x_test.shape
    x_test = x_test.reshape((nsamples, nx * ny))

    return (x_train, y_train), (x_test, y_test)


def load_cifar10_meta(dest: Path) -> t.Dict[str, t.Any]:
    """Load the CIFAR10 metadata from the specified destination.

    Args:
        dest (Path): Destination path.

    Returns:
        Dict[str, Any]: Dictionary containing the metadata.
    """

    dataset_dir = dest / 'cifar-10-batches-py'
    meta_file = 'batches.meta'

    with open(dataset_dir / meta_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    return data