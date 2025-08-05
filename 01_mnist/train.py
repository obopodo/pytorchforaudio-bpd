from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.dense_layers(x)
        preds = self.softmax(logits)
        return preds


def load_mnist_data(root="./data", which: Literal["train", "test", "both"] = "both"):
    if which not in ["train", "test", "both"]:
        raise ValueError("Parameter 'which' must be one of 'train', 'test', or 'both'.")

    train, test = None, None
    if which in ["train", "both"]:
        train = datasets.MNIST(root=root, train=True, download=True, transform=ToTensor())
    if which in ["test", "both"]:
        test = datasets.MNIST(root=root, train=False, download=True, transform=ToTensor())
    return train, test


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():  # for Apple Silicon Macs
        return "mps"
    else:
        return "cpu"


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        preds = model(X)
        loss = loss_fn(preds, y)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        # print(f"Batch {batch}, Loss: {loss.item()}")
    print(f"Loss: {loss.item()}")


def train(model, train_data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        train_one_epoch(model, train_data_loader, loss_fn, optimizer, device)
        print("-" * 30)
    print("Training complete.")


if __name__ == "__main__":
    train_data, test_data = load_mnist_data()
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    device = get_device()
    print(f"Using device: {device}")

    feed_forward_net = FeedForwardNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    train(
        model=feed_forward_net,
        train_data_loader=train_data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
    )

    torch.save(feed_forward_net.state_dict(), "feed_forward_net.pth")
    print("Model saved to feed_forward_net.pth")
