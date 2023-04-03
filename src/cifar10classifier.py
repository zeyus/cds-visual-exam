import torch
import logging
from utils.data import get_cifar10_test_and_train_loader
from utils.nn import train_model, init_model_weights_rand
from models.Cifar10CNN import Cifar10CNN


def main(device: torch.device):
    train_loader, test_loader, classes = get_cifar10_test_and_train_loader()
    model = Cifar10CNN().to(device)
    model.apply(init_model_weights_rand)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_loss, train_acc = train_model(
        model,
        train_loader,
        optimizer,
        loss_fn,
        epochs=10,
        device=device)
    logging.info(f'Train loss: {train_loss}')
    logging.info(f'Train accuracy: {train_acc}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device)
