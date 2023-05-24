import os
import pandas as pd
import typing as t
from pathlib import Path
from sklearn import preprocessing
import torch
from torch import as_tensor
from torch import device
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import logging
import numpy as np
from ..model import get_preprocessor


class EekSpidersDataset(Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            transform=None,
            target_transform=None,
            which="train",
            target_encoder=None):
        logging.info(f"Loading {which} dataset from {annotations_file}")
        annotations = pd.read_csv(annotations_file,
                                  usecols=["filepaths", "labels"])
        # replace \ with / in filepaths
        annotations["filepaths"] = annotations["filepaths"].str.replace("\\", "/")
        if which == "train":
            # filter to only include filepaths prefixed with "train"
            annotations = annotations[annotations["filepaths"].str.startswith("train")]
        elif which == "test":
            # filter to only include filepaths prefixed with "test"
            annotations = annotations[annotations["filepaths"].str.startswith("test")]
        elif which == "valid":
            # filter to only include filepaths prefixed with "valid"
            annotations = annotations[annotations["filepaths"].str.startswith("valid")]
        # peek at the first 5 rows
        if target_encoder is None:
            target_encoder = preprocessing.LabelEncoder()
            target_encoder.fit(annotations["labels"])
        self.target_encoder = target_encoder
        labels = as_tensor(target_encoder.transform(annotations["labels"])).long()
        annotations["labels"] = labels
        self.img_labels = annotations
        self.labels = labels.cuda()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_target_encoder():
    """
    Get the target transform.
    """
    targets = [
        'Black Widow',
        'Blue Tarantula',
        'Bold Jumper',
        'Brown Grass Spider',
        'Brown Recluse Spider',
        'Deinopis Spider',
        'Golden Orb Weaver',
        'Hobo Spider',
        'Huntsman Spider',
        'Ladybird Mimic Spider',
        'Peacock Spider',
        'Red Knee Tarantula',
        'Spiny-backed Orb-weaver',
        'White Kneed Tarantula',
        'Yellow Garden Spider'
    ]

    target_encoder = preprocessing.LabelEncoder()
    target_encoder.fit(targets)

    return target_encoder


def get_dataloaders(
        device: device,
        data_dir: Path,
        batch_size: int = 32,
        test: bool = False,
        rotation: int = 90) -> t.Tuple[
            t.Optional[DataLoader[EekSpidersDataset]],
            t.Optional[DataLoader[EekSpidersDataset]],
            t.Optional[DataLoader[EekSpidersDataset]]]:
    """
    Get the train, test and validation dataloaders.
    """

    target_encoder = get_target_encoder()

    std_img_transforms = get_preprocessor(device, randomization=False)

    if test:
        test_dataset = EekSpidersDataset(
            annotations_file=data_dir / "spiders.csv",
            img_dir=data_dir,
            which="test",
            transform=std_img_transforms,
            target_encoder=target_encoder,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True)
        return None, test_dataloader, None

    train_image_transforms = get_preprocessor(device, randomization=True, rotation=rotation)
    train_dataset = EekSpidersDataset(
        annotations_file=data_dir / "spiders.csv",
        img_dir=data_dir,
        which="train",
        transform=train_image_transforms,
        target_encoder=target_encoder,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)

    val_dataset = EekSpidersDataset(
        annotations_file=data_dir / "spiders.csv",
        img_dir=data_dir,
        which="valid",
        transform=std_img_transforms,
        target_encoder=target_encoder,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False)

    return train_dataloader, None, val_dataloader


def predict_single_image(
        model,
        img_path: Path,
        target_encoder: preprocessing.LabelEncoder,
        device: device) -> t.Tuple[str, float]:
    """
    Predict the class of a single image.
    """
    transforms = get_preprocessor(device, randomization=False)
    image = Image.open(img_path)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = transforms(image)
    image = image.unsqueeze(0)  # type: ignore
    model.eval()
    with torch.no_grad():
        preds = model(image)
    preds = preds.cpu().numpy()
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_class = target_encoder.inverse_transform([pred_idx])[0]
    pred_prob = preds[0][pred_idx]
    return pred_class, pred_prob
