"""Export a PyTorch model to linalg-on-tensors MLIR text with torch-mlir's FX importer."""

from __future__ import annotations

from oasis.stages import StageError


def export_linalg(model, inputs, func_name: str = "forward") -> str:
    """Return linalg-on-tensors MLIR (text) for `model` traced with `inputs`."""
    try:
        import torch
        from torch_mlir import fx
    except ImportError as e:
        raise StageError(
            "linalg",
            f"cannot import torch / torch_mlir ({e}). Install the torch-mlir nightly wheel "
            "(see README: Setup).",
        ) from e

    with torch.no_grad():
        module = fx.export_and_import(
            model, *inputs, output_type="linalg-on-tensors", func_name=func_name
        )
    return str(module)
