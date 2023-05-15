"""cli.py

CLI handling utilities.
"""

import logging
import typing as t
import argparse
from pathlib import Path
from .data import download_and_extract_cifar10, load_cifar10, load_cifar10_meta
from .stats import plot_history, generate_save_basename, save_classification_report
from ..models import train_and_validate_model, cifar10LRModel, cifar10NNModel
from datetime import datetime
from .. import __version__


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
        '--no-download',
        help="Do not attempt to download the CIFAR10 dataset",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--force',
        help="Force download of the CIFAR10 dataset",
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
        '-b',
        '--batch-size',
        help="The batch size",
        type=int,
        default=256
    )
    parser.add_argument(
        '-e',
        '--epochs',
        help="The number of epochs",
        type=int,
        default=150
    )
    parser.add_argument(
        '-n',
        '--neural-network',
        help="Use a neural network model, otherwise use logistic regression",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '-o',
        '--out',
        help="The output path for the plots and stats",
        type=Path,
        default=Path("out"))

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
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args = parse_args()
    if not args.no_download:
        download_and_extract_cifar10(args.dataset_path, args.force)

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_cifar10(args.dataset_path)

    # print the shapes
    logging.info(f"x_train shape: {x_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"x_test shape: {x_test.shape}")
    logging.info(f"y_test shape: {y_test.shape}")

    metadata = load_cifar10_meta(args.dataset_path)
    label_names: t.List[str] = [x.decode() for x in metadata.get(b'label_names')]  # type: ignore
    logging.info(f"Label names: {label_names}")

    # Train and validate the model
    if args.neural_network:
        logging.info("Using neural network model")
        model = cifar10NNModel(epochs=args.epochs, batch_size=args.batch_size)
    else:
        logging.info("Using logistic regression model")
        model = cifar10LRModel(epochs=args.epochs)

    history = train_and_validate_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=args.epochs)

    if len(history['loss']) > 0:
        # Plot the history
        history_filename = generate_save_basename(str(model.__class__.__name__), 'cifar10', 'history', prefix=timestamp_str)
        plot_history(args.out / history_filename, history)

    # Save the classification report
    classification_report_filename = generate_save_basename(str(model.__class__.__name__), 'cifar10', 'classification_report', prefix=timestamp_str)
    y_pred = model.predict(x_test)

    save_classification_report(
        args.out / classification_report_filename,
        y_test,
        y_pred,
        label_names)
