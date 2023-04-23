import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import logging
import itertools


def safe_file_name(name: str) -> str:
    """Replace all special characters with underscores."""
    return "".join([c if c.isalnum() else "_" for c in name])


def generate_save_basename(
        base_path,
        model_name,
        input_size,
        batch_size,
        iterations,
        extra_params=[]):
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_timestamp}_{model_name}_{input_size[0]}x"
    file_name += f"{input_size[1]}_batch{batch_size}_iter{iterations}"
    if extra_params:
        file_name += "_" + "_".join(extra_params)
    file_name = safe_file_name(file_name)
    return base_path / file_name


def plot_history(H, epochs, save_path=None):
    logging.info("Plotting history...")
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs),
             H.history["val_loss"],
             label="val_loss",
             linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs),
             H.history["val_accuracy"],
             label="val_acc",
             linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    if save_path:
        save_path = save_path.with_suffix(".history.png")
        plt.savefig(save_path)
        # save history to csv
        history_df = pd.DataFrame(H.history)
        history_df.to_csv(save_path.with_suffix(".csv"))
    else:
        plt.show()


def save_confusion_matrix(save_path, y_true, y_pred, classes):
    logging.info("Saving confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation="nearest", cmap="PiYG")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path)


def save_classification_report(save_path, model, test_data, classes):
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    y_pred = model.predict(test_data)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.array([]) 
    logging.info("Generating classification report...")
    for _, cls in test_data.as_numpy_iterator():
        y_true = np.append(y_true, cls.argmax(axis=1))

    report = classification_report(y_true, y_pred, target_names=classes)

    out_filename = save_path.with_suffix(".txt")
    with open(out_filename, "w") as f:
        f.write("\n".join(model_summary))
        f.write(report)  # type: ignore

    save_confusion_matrix(
        save_path.with_suffix(".cm.png"), y_true, y_pred, classes)

    logging.info(report)


class EpochTrackerCallback(tf.keras.callbacks.Callback):
    EPOCH = 0

    def on_epoch_end(self, batch, logs=None):
        self.EPOCH += 1
