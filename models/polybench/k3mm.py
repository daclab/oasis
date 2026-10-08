import torch


class k3mm(torch.nn.Module):
    """PolyBench 3mm: G = (A @ B) @ (C @ D)."""

    def forward(self, a, b, c, d):
        e = a @ b
        f = c @ d
        g = e @ f
        return g
