from torchvision.models import (
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    VGG,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,

)
import torch
from torch import nn  # , compile
import logging
from pathlib import Path
import typing as t


def get_vgg_spiders(
        num_classes: int = 15,
        vgg: str = "vgg19",
        saved_model: t.Optional[Path] = None) -> VGG:
    """
    Get the model.
    """
    selected_vgg = vgg.lower()
    if selected_vgg == "vgg11":
        model_class = vgg11
        model_weights = VGG11_Weights.DEFAULT
    elif selected_vgg == "vgg13":
        model_class = vgg13
        model_weights = VGG13_Weights.DEFAULT
    elif selected_vgg == "vgg16":
        model_class = vgg16
        model_weights = VGG16_Weights.DEFAULT
    elif selected_vgg == "vgg19":
        model_class = vgg19
        model_weights = VGG19_Weights.DEFAULT
    else:
        raise ValueError(f"Unknown VGG model: {selected_vgg}")
    logging.info(f"Loading {selected_vgg} model")
    model = model_class(weights=model_weights)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    # compile(model, backend="inductor")  # not available on windows
    if saved_model is not None:
        logging.info(f"Loading saved model from {saved_model}")
        model.load_state_dict(torch.load(saved_model))
    return model
