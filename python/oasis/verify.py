"""Reference (golden) outputs from eager PyTorch, and comparison against hardware results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# f32 hardware accumulates in a different order than PyTorch, so allow small error.
F32_RTOL = 1e-4
F32_ATOL = 1e-4


def golden(model, inputs) -> np.ndarray:
    """Run the (prepared) model in eager PyTorch and return its output as a numpy array."""
    import torch

    with torch.no_grad():
        out = model(*inputs)
    return out.numpy()


def save_golden(expected: np.ndarray, out_dir: Path) -> Path:
    path = out_dir / "golden.npy"
    np.save(path, expected)
    return path


@dataclass
class CheckResult:
    ok: bool
    max_abs_err: float
    mismatches: int

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return f"{status}: max |err| = {self.max_abs_err:.3g}, mismatches = {self.mismatches}"


def compare(actual: np.ndarray, expected: np.ndarray, dtype: str) -> CheckResult:
    """f32: within F32_RTOL/F32_ATOL. Integer types: bit-exact."""
    actual = np.asarray(actual).reshape(expected.shape)
    err = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if dtype == "f32":
        bad = ~np.isclose(actual, expected, rtol=F32_RTOL, atol=F32_ATOL)
    else:
        bad = actual != expected
    return CheckResult(
        ok=not bad.any(), max_abs_err=float(err.max(initial=0)), mismatches=int(bad.sum())
    )
