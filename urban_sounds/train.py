from pathlib import Path

import numpy as np
import torch
from torch import nn, save
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchinfo import summary

from urban_sounds.cnn import CNNNet
from urban_sounds.dataset import UrbanSoundDataset
from urban_sounds.predict import predict
from urban_sounds.preprocess import TransformPipeline


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: str,
    save_checkpoint: bool = False,
):
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        preds = model(X)
        loss = loss_fn(preds, y)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Batch {batch}, Batch shape: {X.shape}, Loss: {loss.item()}")
    print(f"Loss: {loss.item()}")
    if save_checkpoint:
        save(model.state_dict(), "checkpoint.pth")


def train(
    model: nn.Module,
    train_data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: str,
    epochs: int,
    save_checkpoint: bool,
):
    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        train_one_epoch(model, train_data_loader, loss_fn, optimizer, device, save_checkpoint=save_checkpoint)
        print("-" * 30)
    print("Training complete.")


if __name__ == "__main__":
    ANNOTATIONS_FILE = Path(__file__).parent / "UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = Path(__file__).parent / "UrbanSound8K/audio"
    LEARNING_RATE = 0.001
    EPOCHS = 2

    transform_pipeline = TransformPipeline(
        target_sample_rate=22050,
        num_samples=22050,
        n_fft=1024,
        n_mels=64,
    )

    train_dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=transform_pipeline,
        is_test=False,
        test_fold=10,
    )
    test_dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=transform_pipeline,
        is_test=True,
        test_fold=10,
    )
    device = train_dataset.device
    print("Using device:", device)

    train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    model = CNNNet().to(train_dataset.device)
    # summary(model, input_size=(1, 1, 64, 44))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    train(
        model=model,
        train_data_loader=train_data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        save_checkpoint=True,
    )

    # Prediction
    np.random.seed(0)
    for _ in range(5):
        i = np.random.randint(0, len(test_dataset))
        sample, target = test_dataset[i]
        sample = sample.unsqueeze(0).to(device)
        predicted_class, expected_class = predict(model, sample, target)
        print(f"Predicted: {predicted_class}, Expected: {expected_class}")
