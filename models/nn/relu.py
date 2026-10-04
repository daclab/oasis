import torch
import torch.nn.functional as F


class relu(torch.nn.Module):
    """ReLU(x + y). From From-Algorithm-to-RTL/examples/relu."""

    def forward(self, x, y):
        return F.relu(x + y)
