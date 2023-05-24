from datetime import datetime
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torchvision import transforms
from pathlib import Path
import logging
from tqdm import tqdm
import typing as t
from ignite.engine import Engine
from ignite.metrics import ClassificationReport
from ignite.utils import to_onehot

torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_preprocessor(device: torch.device, randomization: bool = False, rotation: int = 90) -> transforms.Compose:
    transform_list = []
    if randomization:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(rotation))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Lambda(lambda x: x.to(device)))
    transform_list.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)


def get_loss_fn() -> _Loss:
    """
    Get the loss function.
    """
    return CrossEntropyLoss()


def get_optimizer(model) -> Optimizer:
    """
    Get the optimizer.
    """
    return AdamW(model.parameters(), lr=1e-3)


def train_one_epoch(
        epoch_index: int,
        tb_writer: SummaryWriter,
        data_loader: DataLoader,
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        window_size: int = 5) -> float:
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    with tqdm(data_loader) as pbar:
        for i, data in enumerate(pbar):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            pbar.set_description(' loss: {}'.format(loss.item()))
            if i % window_size == window_size-1:
                last_loss = running_loss / window_size  # loss per batch
                # logging.info('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(data_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

    return last_loss


def get_classification_report(model, test_dataloader: DataLoader, device: torch.device, targets: t.List[str]):
    engine = Engine(model)
    report = ClassificationReport(15, is_multilabel=False, device=device, output_dict=False, labels=targets)
    report.attach(engine, "classification_report")
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).unsqueeze(0)
            # Print shapes and a few values for debugging
            print("Targets shape:", targets.shape)
            print("First few targets:", targets[:5])
            print("Outputs shape:", outputs.shape)
            print("First few outputs:", outputs[:5])
            print("Preds shape:", preds.shape)
            print("First few preds:", preds[:5])
            # Update the report
            report.update((preds, targets))

    # Compute and print results
    results = report.compute()
    return results


def validate(
        epoch_index: int,
        tb_writer: SummaryWriter,
        data_loader: DataLoader,
        model: Module,
        loss_fn: _Loss,
        which: str = "validation") -> t.Tuple[float, float]:
    running_loss = 0.
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    accuracy = correct / total
    last_loss = running_loss / (i + 1)
    logging.info(f'  {which} loss: {last_loss}')
    tb_writer.add_scalar(f'Loss/{which}', last_loss, epoch_index)
    logging.info(f'  accuracy: {accuracy}')
    tb_writer.add_scalar(f'Accuracy/{which}', accuracy, epoch_index)

    return last_loss, accuracy


def get_writer(out_dir: Path, model_name: str, timestamp: str) -> SummaryWriter:
    return SummaryWriter(log_dir=str(out_dir / 'runs' / f"spiders_{model_name}_{timestamp}"))


def get_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def train(
        model: Module,
        epochs: int,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        out_dir: Path,
        model_save_dir: Path,
        model_name: str,
        window_size: int = 5) -> None:

    timestamp = get_timestamp()
    tb_writer = get_writer(out_dir, model_name, timestamp)

    best_vloss = 1_000_000.
    epoch = 0
    for epoch in range(epochs):
        logging.info('Epoch {}/{}'.format(epoch + 1, epochs))
        model.train()
        avg_loss = train_one_epoch(epoch, tb_writer, training_loader, model, loss_fn, optimizer, window_size)
        model.eval()
        avg_vloss, _ = validate(epoch, tb_writer, validation_loader, model, loss_fn)
        logging.info('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        tb_writer.add_scalars('Training vs. Validation Loss', {'train': avg_loss, 'validation': avg_vloss}, epoch + 1)
        tb_writer.flush()
        if avg_vloss < best_vloss:
            logging.info('New best model found! Saving model...')
            best_vloss = avg_vloss
            torch.save(model.state_dict(), str(model_save_dir / f'{model_name}_best_model.pth'))

    logging.info('Training complete.')
    model.eval()
    tb_writer.flush()
    tb_writer.close()
