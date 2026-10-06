import torch
import torchvision


class resnet18(torch.nn.Module):
    """torchvision ResNet-18 (ImageNet stem, 1000 classes), randomly initialised."""

    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet18(weights=None)

    def forward(self, x):
        return self.net(x)
