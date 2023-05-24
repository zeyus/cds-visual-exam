import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from tqdm import tqdm
from numpy import asarray, percentile, tile
from scipy.ndimage import gaussian_filter
from . import get_preprocessor


def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook


def activation_maximization(
        model: torch.nn.Module,
        class_index: int,
        img_save_file: Path,
        device: torch.device,
        num_iterations: int = 100,
        regularize: bool = True):
    model.eval()
    # for module in enumerate(model.named_children()):
    #     print(module)

    # Create a random noise image
    # img = np.random.uniform(150, 180, (224, 224, 3))
    # img = img.astype(np.uint8)

    # Preprocess the image
    # preprocess = get_preprocessor(device, randomization=False)
    # img = preprocess(img)
    # add an extra dimension to match the input size expected by the model
    img = torch.randn(3, 224, 224, device=device)
    img = img.unsqueeze(0)  # type: ignore
    # img = Variable(img, requires_grad=True)  # wrap the tensor in a Variable so we can compute gradients
    img.requires_grad_(True)

    logging.info(f"Performing activation maximization for class {class_index}...")

    best_activation = -float('inf')
    best_img = img
    learning_rate = torch.tensor(100)

    activation_dictionary = {}
    layer_name = 'classifier_final'
    model.classifier[-1].register_forward_hook(layer_hook(activation_dictionary, layer_name))  # type: ignore

    # Perform activation maximization
    for i in tqdm(range(num_iterations)):
        img.retain_grad()
        model(img)
        layer_out = activation_dictionary[layer_name]
        layer_out[0][class_index].backward(retain_graph=True)
        img_grad = img.grad
        img = torch.add(img, torch.mul(img_grad, learning_rate))  # type: ignore

        if regularize:
            with torch.no_grad():
                # l2 regularization
                img = torch.mul(img, (0.9))
                temp = img.squeeze(0).cpu()
                if i % 4 == 0:
                    # temp = img.squeeze(0)
                    temp = temp.numpy()
                    for channel in range(3):
                        cimg = gaussian_filter(temp[channel], 1)
                        temp[channel] = cimg
                    temp = torch.from_numpy(temp)

                # img = input.detach().squeeze(0)
                norm = torch.norm(temp, dim=0)
                norm = norm.numpy()
                smalls = norm < percentile(norm, 30)
                smalls = tile(smalls, (3, 1, 1))
                temp = temp - temp*smalls
                abs_img = torch.abs(temp)
                smalls = abs_img < percentile(abs_img, 30)
                temp = temp - temp*smalls
                img = temp.to(device).unsqueeze(0)

        img.requires_grad_(True)
        if best_activation < layer_out[0][class_index]:
            best_activation = layer_out[0][class_index]
            best_img = img

    # Convert the image back to its original form
    img = best_img.data.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img * 255)

    # Save the image
    logging.info(f"Saving image to {img_save_file}...")
    img = Image.fromarray(img)
    img.save(img_save_file)
