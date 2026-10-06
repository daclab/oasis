"""ResNet-18's building blocks, separately, so each can be simulated in reasonable time."""

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, conv1x1


class stem(nn.Module):
    """conv 7x7/2 + BN + ReLU + max-pool 3x3/2 (ResNet-18's first layers)."""

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))


class basic_block(nn.Module):
    """BasicBlock: two 3x3 convs with an identity shortcut (as in layer1)."""

    def __init__(self, channels=64):
        super().__init__()
        self.block = BasicBlock(channels, channels)

    def forward(self, x):
        return self.block(x)


class down_block(nn.Module):
    """BasicBlock with stride 2 and a 1x1 conv shortcut (first block of layer2..4)."""

    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        downsample = nn.Sequential(
            conv1x1(in_channels, out_channels, stride=2), nn.BatchNorm2d(out_channels)
        )
        self.block = BasicBlock(in_channels, out_channels, stride=2, downsample=downsample)

    def forward(self, x):
        return self.block(x)


class head(nn.Module):
    """Global average pool + flatten + fully connected classifier."""

    def __init__(self, channels=512, classes=1000):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(x), 1))
