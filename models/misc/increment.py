import torch


class increment(torch.nn.Module):
    """B[0] = A + 1. From From-Algorithm-to-RTL/examples/increment."""

    def forward(self, a):
        return a + 1
