"""cli.py

CLI handling utilities.
"""

import logging
import typing as t
import argparse
from pathlib import Path
from .data import download_file, load_dataset
from .. import __version__
from ..model import get_model


def common_args(parser: argparse.ArgumentParser):
    """Add common arguments to an argument parser.

    Args:
        parser (argparse.ArgumentParser): an argument parser

    Returns:
        argparse.ArgumentParser: the argument parser
    """
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add_argument(
        '-m',
        '--model-save-path',
        help="Path to save the trained model(s)",
        type=Path,
        default=Path("models")
    )
    parser.add_argument(
        '--download',
        help="Download the dataset",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '-d',
        '--dataset-path',
        help="Path to the dataset",
        type=Path,
        default=Path("data")
    )
    parser.add_argument(
        '-s',
        '--image-size',
        help="The input size of the image, e.g. 32 for 32x32",
        type=int,
        default=32
    )

    return parser


def parse_args(
                args: t.Optional[t.List[str]] = None,
                extra_arg_func: t.Optional[t.Callable[
                    [argparse.ArgumentParser],
                    argparse.ArgumentParser]] = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        args (t.Optional[t.List[str]], optional):
            Arguments to parse. Defaults to None.
        extra_arg_func (t.Optional[t.Callable[
            [argparse.ArgumentParser],
            argparse.ArgumentParser]], optional):
                Function to add extra arguments. Defaults to None.
    """
    parser = argparse.ArgumentParser(
        description="Text classification CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if extra_arg_func:
        parser = extra_arg_func(parser)

    return common_args(parser).parse_args(args)


def run():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.download:
        if not args.dataset_path.exists():
            args.dataset_path.mkdir(parents=True)
            download_file(args.dataset_path)
        else:
            logging.warning(
                "Dataset path already exists. "
                "Skipping download.")
            logging.info("To force download, delete the dataset path first.")
    elif not args.dataset_path.exists():
        raise FileNotFoundError(
            "Dataset path does not exist. "
            "Please download the dataset first.")

    logging.info("Dataset path: %s", args.dataset_path)
    logging.info("Model save path: %s", args.model_save_path)
    logging.info("Version: %s", __version__)
    logging.info("Running...")
    train_data, test_data, val_data = load_dataset(args.dataset_path)
    print(val_data)
    model = get_model()
