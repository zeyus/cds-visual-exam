import logging
import typing as t
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_history(save_path: Path, history: t.Dict[str, t.Any]):
    """Plot the training history.

    Args:
        save_path (Path): Path to save the plot.
        history (t.Dict[str, t.Any]): Training history.
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(history["loss"])), history["loss"], label="train_loss")
    plt.plot(
        np.arange(0, len(history["val_loss"])), history["val_loss"], label="val_loss"
    )
    plt.plot(
        np.arange(0, len(history["accuracy"])),
        history["accuracy"],
        label="train_acc",
    )
    plt.plot(
        np.arange(0, len(history["val_accuracy"])),
        history["val_accuracy"],
        label="val_acc",
    )
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    logging.info(f"Saving plot to {save_path}")
    plt.savefig(save_path.with_suffix(".png"))
    plt.close()


def save_classification_report(save_path: Path, y_true: t.Any, y_pred: t.Any, target_names: t.List[str]):
    """Save the classification report.

    Args:
        save_path (Path): Path to save the report.
        report (str): Classification report.
    """
    logging.info(f"Saving classification report to {save_path}")
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    with open(save_path.with_suffix(".txt"), "w") as f:
        f.write(report)  # type: ignore


def generate_save_basename(model_name: str, dataset_name: str, what: str, prefix: str = ""):
    """Generate a basename for saving.

    Args:
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        what (str): What this is (e.g. history).
        prefix (str, optional): Prefix for the basename. Defaults to "".

    Returns:
        str: Basename for saving.
    """
    prefix = f"{prefix}_" if prefix else ""
    return f"{prefix}{model_name}_{dataset_name}_{what}"
