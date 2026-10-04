import torch


class gemm(torch.nn.Module):
    """GEMM: C = A @ B."""

    def forward(self, a, b):
        c = a @ b
        return c
