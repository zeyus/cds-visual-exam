"""cli.py

CLI handling utilities.
"""

import logging
import typing as t
import argparse
from pathlib import Path

import torch
from .. import __version__
from .data import get_dataloaders, get_target_encoder, predict_single_image
from .helper import preview_targets
from ..model import get_loss_fn, get_optimizer, train, get_timestamp, get_classification_report
from ..model.viz import activation_maximization
from ..model.vggspiders import get_vgg_spiders


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
        '-o',
        '--output-path',
        help="Path to save the output, figures, stats, etc.",
        type=Path,
        default=Path("out")
    )
    parser.add_argument(
        '--download',
        help="Download the dataset from kaggle",
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
        '-e',
        '--epochs',
        help="Number of epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        '-n',
        '--net-name',
        help="Name of the network to use (e.g. vgg19)",
        type=str,
        default="vgg19"
    )

    # batch size
    parser.add_argument(
        '-b',
        '--batch-size',
        help="Batch size",
        type=int,
        default=32)

    parser.add_argument(
        '-t',
        '--test-best-models',
        help="Test the best models",
        action="store_true",
        default=False
    )

    parser.add_argument(
        '-w',
        '--weights',
        help="Path to the weights file (.pth)",
        type=Path,
        default=None
    )

    parser.add_argument(
        '-V',
        '--visualize',
        help="Visualize the model (activation maximization)",
        action="store_true",
        default=False
    )

    parser.add_argument(
        '-i',
        '--image-path',
        help="Predict the class for a single input image",
        type=Path,
        default=None
    )

    parser.add_argument(
        '-r',
        '--rotation',
        help="Random rotation angle (degrees) for the input image",
        type=int,
        default=90
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
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info('Using device: {}'.format(dev))

    if args.test_best_models:
        logging.info('Testing best models')
        # get all pth files in model_save_path
        pth_files = list(args.model_save_path.glob("*_best_model.pth"))
        # prepare loss function
        loss_fn = get_loss_fn()
        timestamp = get_timestamp()
        target_encoder = get_target_encoder()
        for pth_file in pth_files:
            # get model name - first part of file name
            model_name = pth_file.name.split("_")[1]
            # model name optionally has batch size after vggxx
            batch_info = model_name.split("-")
            batch_size = args.batch_size
            if len(batch_info) > 1:
                model_name = batch_info[0]
                batch_size = int(batch_info[1])
            # load model
            logging.info(f'Loading saved model weights: {pth_file}')
            model = get_vgg_spiders(vgg=model_name, saved_model=pth_file)
            model.to(dev)
            # tb_writer = get_writer(args.output_path, model_name, timestamp)
            # prepare dataloaders
            logging.info('Preparing dataloader')
            _, test_dataloader, _ = get_dataloaders(
                device=dev,
                data_dir=args.dataset_path,
                batch_size=batch_size,
                test=True
            )
            # logging.info(f'Testing model: {model_name}')
            # test_loss, test_acc = validate(1, tb_writer, test_dataloader, model, loss_fn, which='test')  # type: ignore
            # tb_writer.close()
            # logging.info('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_acc))
            report = get_classification_report(
                model,
                test_dataloader,
                dev,
                targets=list(target_encoder.classes_))
            logging.info(f'Classification report for model: {model_name}')
            logging.info(report)
            # save classification report to file
            report_file = args.output_path / f'{model_name}_classification_report.txt'
            with open(report_file, 'w') as f:
                f.write(report)
        return

    # prepare loss function
    loss_fn = get_loss_fn()
    # prepare model
    logging.info('Preparing model')
    model = get_vgg_spiders(vgg=args.net_name, saved_model=args.weights)

    model.to(dev)

    if args.visualize:
        if not args.weights:
            raise ValueError('You need to specify a weights file to visualize the model')
        logging.info('Visualizing model')
        target_encoder = get_target_encoder()
        for i in range(0, len(target_encoder.classes_)):
            logging.info(f'Visualizing class: {target_encoder.classes_[i]}')
            class_name_safe = target_encoder.classes_[i].replace(" ", "_")
            out_dir = args.output_path / args.net_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_img = out_dir / f'activation_max_{args.rotation}_{args.net_name}_{class_name_safe}.png'
            activation_maximization(
                model=model,
                class_index=i,
                img_save_file=out_img,
                device=dev,
                num_iterations=2000)
        return

    if args.image_path:
        logging.info('Predicting class for image')
        target_encoder = get_target_encoder()

        pred_class, pred_prob = predict_single_image(
            model=model,
            img_path=args.image_path,
            target_encoder=target_encoder,
            device=dev)
        # target_encoder.classes_[pred_class]
        logging.info(f'Predicted class: ({pred_class}), probability: {pred_prob}')
        return

    # prepare optimizer
    optimizer = get_optimizer(model)

    # prepare dataloaders
    logging.info('Preparing dataloaders')
    train_dataloader, _, val_dataloader = get_dataloaders(
        device=dev,
        data_dir=args.dataset_path,
        batch_size=args.batch_size,
        test=False,
        rotation=args.rotation
    )

    # preview targets
    preview_targets(train_dataloader.dataset, args.output_path)  # type: ignore

    # train
    train(
        model=model,
        epochs=args.epochs,
        training_loader=train_dataloader,  # type: ignore
        validation_loader=val_dataloader,  # type: ignore
        loss_fn=loss_fn,
        optimizer=optimizer,
        model_save_dir=args.model_save_path,
        out_dir=args.output_path,
        model_name=str(args.rotation) + '_' + args.net_name + '-' + str(args.batch_size))
