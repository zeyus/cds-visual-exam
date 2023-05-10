"""Data utils."""

import typing as t
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_test_and_train_loader(
        batch_size: int = 64,
        data_dir: str = './data',
        num_workers: int = 4) -> t.Tuple[
            DataLoader,
            DataLoader,
            t.Tuple[str, ...]]:
    """Get CIFAR10 test and train loader.

    Args:
        batch_size (int, optional): Defaults to 64.
        data_dir (str, optional): Defaults to './data'.
        num_workers (int, optional): Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader, Tuple[str, ...]]:
            train_loader, test_loader, classes
    """

    target_mean = (0.5, 0.5, 0.5)
    target_std = (0.5, 0.5, 0.5)
    batch_size = 64

    data_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize(target_mean, target_std)])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True,
        transform=data_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True,
        transform=data_transform)

    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers)

    classes = (
        'plane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck')

    return train_loader, test_loader, classes
