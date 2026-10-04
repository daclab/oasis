"""Model preparation before export: weights as kernel arguments.

Registry entries with "weights": "args" (models/data.py) export the model with its
parameters and buffers as extra forward() arguments, so each becomes an external memory
of the kernel instead of a constant baked into the IR.
"""

from __future__ import annotations

import torch


def weight_tensors(model: torch.nn.Module) -> tuple[list[str], list[torch.Tensor]]:
    """Names and values of every parameter and buffer, in a stable order."""
    state = dict(model.named_parameters())
    state |= {n: b for n, b in model.named_buffers() if not n.endswith("num_batches_tracked")}
    names = list(state)
    return names, [state[n].detach().clone() for n in names]


class WeightsAsArgs(torch.nn.Module):
    """forward(*activations, *weights): the model with its weights passed in as arguments."""

    def __init__(self, model: torch.nn.Module, names: list[str], n_activations: int):
        super().__init__()
        self.model = model
        self.names = names
        self.n_activations = n_activations

    def forward(self, *args):
        acts, weights = args[: self.n_activations], args[self.n_activations :]
        return torch.func.functional_call(self.model, dict(zip(self.names, weights)), acts)


def with_weight_args(model: torch.nn.Module, activations) -> tuple[torch.nn.Module, tuple]:
    """(module to export, all forward() arguments: activations then weights)."""
    names, weights = weight_tensors(model)
    return WeightsAsArgs(model, names, len(activations)), (*activations, *weights)
