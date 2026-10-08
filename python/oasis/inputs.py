"""Seeded input generation for benchmarks (shapes come from models/data.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oasis.models import Benchmark

SEED = 0


def make_inputs(bench: Benchmark, seed: int = SEED):
    """Deterministic inputs: f32 uniform in [-1, 1), i32 uniform in [-8, 8).

    Arguments listed in `token_inputs` are token ids: int64 uniform in [0, vocab_size).
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    tensors = []
    for i, shape in enumerate(bench.input_shapes):
        if i in bench.token_inputs:
            vocab = bench.init_kwargs["vocab_size"]
            tensors.append(torch.randint(0, vocab, shape, generator=g, dtype=torch.int64))
        elif bench.dtype == "f32":
            tensors.append(torch.rand(shape, generator=g, dtype=torch.float32) * 2 - 1)
        elif bench.dtype == "i32":
            tensors.append(torch.randint(-8, 8, shape, generator=g, dtype=torch.int32))
        else:
            raise ValueError(f"unsupported dtype '{bench.dtype}'")
    return tuple(tensors)


def save_inputs(inputs, out_dir: Path) -> Path:
    """Write inputs to <out_dir>/inputs.npz as arg0, arg1, ... (forward() argument order)."""
    path = out_dir / "inputs.npz"
    np.savez(path, **{f"arg{i}": t.numpy() for i, t in enumerate(inputs)})
    return path
