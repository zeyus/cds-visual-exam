import tqdm
import logging
import typing as t

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import log_loss, accuracy_score


def cifar10LRModel(epochs: int = 150):
    """Logistic Regression model for CIFAR10 dataset.

    Returns:
        LogisticRegression: Logistic Regression model.
    """
    return LogisticRegression(
        penalty='l2',
        solver='saga',
        multi_class='multinomial',
        max_iter=epochs,
        verbose=1,
        warm_start=True,
        tol=1e-4,
        n_jobs=-1
    )


def cifar10NNModel(epochs: int, batch_size: int):
    """Neural Network model for CIFAR10 dataset.

    Returns:
        MLPClassifier: Neural Network model.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 256, 128,),
        learning_rate='adaptive',
        alpha=0.015,
        learning_rate_init=0.02,
        early_stopping=False,
        activation='relu',
        solver='sgd',
        max_iter=epochs,
        verbose=False,
        batch_size=batch_size,
        n_iter_no_change=20,
        momentum=0.9,
        warm_start=True,
    )


def train_and_validate_model(model: t.Union[MLPClassifier, LogisticRegression], x_train, y_train, x_val, y_val, epochs=10):
    """Train and validate the model.

    Args:
        model (t.Union[MLPClassifier, LogisticRegression]): Model to train.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        t.Dict[str, t.Any]: Training history.
    """
    # model.fit(x_train, y_train)
    # return {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    if isinstance(model, LogisticRegression):
        # we just fit the model and return the history
        try:
            model.fit(x_train, y_train)
        except KeyboardInterrupt:
            logging.info('Keyboard interrupt, stopping training')
        return {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # train
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    classes = np.unique(y_train)
    try:
        t = tqdm.tqdm(range(epochs), desc='Epoch', postfix={'loss': 0, 'accuracy': 0, 'val_loss': 0, 'val_accuracy': 0})
        for epoch in t:
            # logging.info(f'Epoch {epoch + 1}/{epochs}')
            model.partial_fit(x_train, y_train, classes=classes)
            y_pred = model.predict_proba(x_train)
            y_pred_val = model.predict_proba(x_val)
            loss = log_loss(y_train, y_pred, labels=classes)
            loss_val = log_loss(y_val, y_pred_val, labels=classes)
            acc = accuracy_score(y_train, y_pred.argmax(axis=-1))
            acc_val = accuracy_score(y_val, y_pred_val.argmax(axis=-1))
            history['loss'].append(loss)
            history['accuracy'].append(acc)
            history['val_loss'].append(loss_val)
            history['val_accuracy'].append(acc_val)
            t.set_postfix({'loss': loss, 'accuracy': acc, 'val_loss': loss_val, 'val_accuracy': acc_val})
            # logging.info(f'Train loss: {loss:.4f} - Train accuracy: {acc:.4f}')
            # logging.info(f'Val loss: {loss_val:.4f} - Val accuracy: {acc_val:.4f}')
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt, stopping training')

    return history
