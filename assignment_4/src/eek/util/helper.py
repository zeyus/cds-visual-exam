from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from pathlib import Path
import logging
from .data import EekSpidersDataset


def preview_targets(dataset: Dataset[EekSpidersDataset], dest: Path):
    """
    Preview the targets in the dataset.
    """

    # if png exist, just return to save time
    if (dest / "target_distribution.png").exists():
        return

    plt.margins(0.05, tight=True)

    logging.info("Plotting target distribution")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(dataset.img_labels["labels"])
    ax.set_xticks(range(len(dataset.target_encoder.classes_)))
    ax.set_xticklabels(dataset.target_encoder.classes_, rotation=90)
    ax.set_title("Target distribution")
    fig.savefig(dest / "target_distribution.png")

    logging.info("Plotting target examples")
    # show an example of each target (3x5)
    cols = 3
    rows = 5
    fig, ax = plt.subplots(rows, cols, figsize=(5, 15))
    fig.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
    for i, target in enumerate(dataset.target_encoder.classes_):
        # get a random image with the target
        target_idx = dataset.img_labels[dataset.img_labels["labels"] == i].sample().index[0]
        img = dataset[target_idx][0]
        # convert to PIL image
        img = ToPILImage()(img)
        ax[i // cols, i % cols].imshow(img)
        ax[i // cols, i % cols].set_title(target)
        # reduce title font size
        ax[i // cols, i % cols].title.set_fontsize(9)
        ax[i // cols, i % cols].axis("off")
    # reduce margins
    fig.subplots_adjust(hspace=0.01, wspace=0.01)

    fig.savefig(dest / "target_examples.png")
