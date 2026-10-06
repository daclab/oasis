"""Model preparation before export: inference rewrites and weights-as-arguments.

Registry entries (models/data.py) list steps under "prepare"; they run in order on the
seeded, eval-mode model. The prepared model is what gets compiled and what the golden
output is computed from.
"""

from __future__ import annotations

import torch


def randomize_bn(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Give every BatchNorm non-trivial statistics and affine parameters (seeded).

    A freshly initialised BatchNorm (mean 0, var 1, weight 1, bias 0) is almost the
    identity, which would make BN folding untested. Values stay in a numerically tame range.
    """
    g = torch.Generator().manual_seed(seed + 1)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            n = m.num_features
            with torch.no_grad():
                m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
                m.weight.copy_(torch.rand(n, generator=g) + 0.5)
                m.bias.copy_(torch.randn(n, generator=g) * 0.1)
    return model


def fold_bn(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Fold eval-mode BatchNorm into the preceding conv (weights and bias).

    Removes BN's runtime 1/sqrt(var + eps) (math.rsqrt, f64 eps), which hardware doesn't
    need at inference time. Uses torch.fx's conv+bn fusion; the model must be traceable.
    """
    from torch.fx.experimental.optimization import fuse

    return fuse(model.eval())


STEPS = {"randomize_bn": randomize_bn, "fold_bn": fold_bn}


def apply_prepare(model: torch.nn.Module, steps: list[str], seed: int) -> torch.nn.Module:
    for step in steps:
        if step not in STEPS:
            raise KeyError(f"unknown prepare step '{step}'; known: {', '.join(STEPS)}")
        model = STEPS[step](model, seed)
    return model.eval()


def weight_tensors(model: torch.nn.Module) -> tuple[list[str], list[torch.Tensor]]:
    """Names and values of every parameter and buffer, in a stable order."""
    state = dict(model.named_parameters())
    state |= {n: b for n, b in model.named_buffers() if not n.endswith("num_batches_tracked")}
    names = list(state)
    return names, [state[n].detach().clone() for n in names]


class WeightsAsArgs(torch.nn.Module):
    """forward(*activations, *weights): the model with its weights passed in as arguments.

    Each weight tensor becomes a memref argument of the kernel, i.e. an external memory
    in the hardware, instead of a huge constant baked into the IR.
    """

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
