"""Minimal reproductions for the softmax row-maximum bug seen in `attention`."""

import torch


class rowmax_idx(torch.nn.Module):
    """Row maximum via max(dim): the lowering also tracks the argmax index (as softmax does)."""

    def forward(self, x):
        return torch.max(x, dim=-1).values


class rowmax(torch.nn.Module):
    """Row maximum via amax: values only, no index."""

    def forward(self, x):
        return torch.amax(x, dim=-1)
