"""cli.py

CLI handling utilities.
"""

import logging
import typing as t
import argparse
from pathlib import Path
from .data import download_file, load_dataset
from .stats import plot_history, generate_save_basename, save_classification_report, EpochTrackerCallback
from .. import __version__
from . import set_tf_optim
from ..model import get_model, save_best_callback, get_model_resnet
import tensorflow as tf


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

    # either -s for image size or -w and -h for width and height
    image_size_group = parser.add_mutually_exclusive_group()
    image_size_group.add_argument(
        '-s',
        '--image-size',
        help="The image size",
        type=int,
        default=32
    )
    w_h_group = image_size_group.add_argument_group()
    w_h_group.add_argument(
        '-w',
        '--image-width',
        help="The image width",
        type=int
    )
    w_h_group.add_argument(
        '-t',
        '--image-height',
        help="The image height",
        type=int
    )

    parser.add_argument(
        '-b',
        '--batch-size',
        help="The batch size",
        type=int,
        default=32
    )
    parser.add_argument(
        '-e',
        '--epochs',
        help="The number of epochs",
        type=int,
        default=10
    )
    parser.add_argument(
        '-o',
        '--out',
        help="The output path for the plots and stats",
        type=Path,
        default=Path("out"))

    parser.add_argument('-n',
                        '--no-train',
                        help="Do not train the model",
                        action="store_true",
                        default=False)
    parser.add_argument('-c',
                        '--from-checkpoint',
                        help="Use the checkpoint at the given path",
                        type=Path,
                        default=None)
    parser.add_argument('-r',
                        '--resnet',
                        help="Use ResNet50 as the base model.",
                        action="store_true",
                        default=False)
    parser.add_argument('-p',
                        '--parallel',
                        help="Number of workers / threads to use for parallel processing.",
                        type=int,
                        default=4)
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

    if args.parallel > 0:
        set_tf_optim(args.parallel)

    if args.image_width and args.image_height:
        input_shape = (args.image_width, args.image_height, 3)
    else:
        input_shape = (args.image_size, args.image_size, 3)

    logging.info("Image input size: %s", input_shape)

    logging.info("Dataset path: %s", args.dataset_path)
    logging.info("Model save path: %s", args.model_save_path)
    logging.info("Version: %s", __version__)
    logging.info("Running...")
    train_data, test_data, val_data, classes = load_dataset(
        args.dataset_path,
        input_shape,
        args.batch_size)

    if args.resnet:
        logging.info("Using ResNet50 as the base model.")
        model_save_path = args.model_save_path / "resnet"
        model_func = get_model_resnet
    else:
        logging.info("Using VGG16 as the base model.")
        model_save_path = args.model_save_path / "vgg16"
        model_func = get_model
    if not model_save_path.exists():
        model_save_path.mkdir(parents=True)

    if not args.out.exists():
        args.out.mkdir(parents=True)

    logging.info("Classes: %s", classes)

    out_file_basename = generate_save_basename(
        args.out,
        "resnet" if args.resnet else "vgg16",
        input_shape,
        args.batch_size,
        args.epochs,
    )

    model_save_basename = generate_save_basename(
        model_save_path,
        "resnet" if args.resnet else "vgg16",
        input_shape,
        args.batch_size,
        args.epochs,
    )

    if args.from_checkpoint:
        logging.info("Loading model from checkpoint: %s", args.from_checkpoint)
        model = tf.keras.models.load_model(args.from_checkpoint)
    else:
        model = model_func(
            output_classes=len(classes),
            input_shape=input_shape)

    model_save_filename = model_save_basename.with_suffix(".tf")

    if not args.no_train:
        logging.info("Training...")
        epoch_tracker = EpochTrackerCallback()
        H = None
        try:
            H = model.fit(
                train_data,
                epochs=args.epochs,
                validation_data=val_data,
                callbacks=[save_best_callback(model_save_filename), epoch_tracker],
                verbose=1,
                workers=args.parallel,
                use_multiprocessing=True if args.parallel > 1 else False)
        except KeyboardInterrupt:
            logging.info(f"Training interrupted. Current Epoch: {epoch_tracker.EPOCH}")
            if H is None:
                H = model.history
        if epoch_tracker.EPOCH > 0:
            plot_history(H, epoch_tracker.EPOCH, out_file_basename)
    logging.info("Evaluating...")
    save_classification_report(
        out_file_basename,
        model,
        test_data,
        classes)
