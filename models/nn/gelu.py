import torch
import torch.nn.functional as F


class gelu(torch.nn.Module):
    """GELU(x + y). From From-Algorithm-to-RTL/examples/gelu."""

    def forward(self, x, y):
        return F.gelu(x + y)
