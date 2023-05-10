# Test fine tuning of VGG16 model on the Oxford 102 flower dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import Flowers102
from tqdm import tqdm
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import torchvision


# Train and evaluate
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, use_gpu=True):
    since = time.time()
    if use_gpu:
        model = model.cuda()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    # Load the pretrained model from pytorch
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    # Add our layer with 102 outputs
    features.extend([nn.Linear(num_features, 102)])
    vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier

    # If you want to finetune the model, uncomment the following line
    for param in vgg16.features.parameters():
        param.require_grad = True

    # Check to see that your last layer produces the expected number of outputs
    print(vgg16.classifier[6].out_features)

    # Load the data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/oxford102'

    # prepare data set
    flowers_train = Flowers102(
        data_dir, download=True, transform=data_transforms['train'])
    flowers_val = Flowers102(data_dir, download=True,
                             split='val', transform=data_transforms['val'])

    image_datasets = {'train': flowers_train, 'val': flowers_val}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # load class names from imagelabels.mat

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[x for x in classes])

    # Finetuning the convnet
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(
        vgg16.classifier.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv, step_size=7, gamma=0.1)
    vgg16 = train_model(vgg16, criterion, optimizer_conv,
                        exp_lr_scheduler, num_epochs=25, use_gpu=use_gpu)

    # Save the model
    torch.save(vgg16.state_dict(), 'vgg16_finetuned.pth')
