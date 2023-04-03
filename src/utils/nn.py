"""Neural network utilities."""

import typing as t
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def init_model_weights_rand(model: torch.nn.Module) -> None:
    """Initialize model weights with random values.

    Args:
        model (torch.nn.Module): Model to initialize.
    """
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)


def train_model(
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        epochs: int = 10,
        device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        ) -> t.Tuple[t.List[float], t.List[float]]:
    """Train model.

    Args:
        model (torch.nn.Module): Model to train.
        loader (DataLoader): Data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.modules.loss._Loss): Loss function.
        epochs (int, optional): Defaults to 10.
        device (torch.device, optional):
            Defaults to torch.device(
                'gpu' if torch.cuda.is_available() else 'cpu').

    Returns:
        Tuple[List[float], List[float]]: train_loss, train_acc
    """

    train_loss = []
    train_acc = []
    logging.info('Training model...')
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        batch_idx = 0
        logging.info(f'Epoch {epoch + 1}/{epochs}')
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader), 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += int(torch.sum(preds == labels.data))
            running_total += labels.size(0)

        train_loss.append(running_loss / batch_idx)
        train_acc.append(running_correct / running_total)
        logging.info(
            f'Loss: {train_loss[-1]:.4f} | '
            f'Accuracy: {train_acc[-1]:.4f}')

    logging.info('Finished training model.')

    return train_loss, train_acc
