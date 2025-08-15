import torch
from torch import nn


class CNNNet(nn.Module):
    # TODO: the good idea is to pass conv layers parameters as arguments to init
    # and then try to select the best using hyperparameter tuning
    def __init__(self, num_classes: int):
        super().__init__()
        # 4 conv blocks -> flatten -> linear -> softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,  # number of conv filters
                kernel_size=3,
                stride=1,
                padding=1,  # equals 2 in original Valerio's implementation, yet idk why should it be
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Consider adding stride == 2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten(0, -1)
        # TODO: recap how this is calculated: 128 filters, but what are 5 and 4???
        self.fc = nn.Linear(128 * 4 * 2, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        try:
            x = self.fc(x)
        except:
            print("Shape before flattening", x.shape)
            raise
        x = self.softmax(x)  # TODO: could be that it's better to move to predict
        return x
