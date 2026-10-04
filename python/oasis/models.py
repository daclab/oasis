"""Benchmark loading.

Model files (models/<suite>/<name>.py) contain only a torch.nn.Module class, like
Stream-HLS pymodels. The registry models/data.py says which class to build and the
input shapes per size. Inputs: oasis.inputs. Checking: oasis.verify.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from oasis.config import REPO_ROOT

SEED = 0  # model weight initialisation (inputs use oasis.inputs.SEED)


def models_dir() -> Path:
    """Directory holding model files and data.py (override with OASIS_MODELS_DIR)."""
    return Path(os.environ.get("OASIS_MODELS_DIR", REPO_ROOT / "models"))


def _load_file(path: Path) -> ModuleType:
    """Import a Python file by path."""
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"oasis_models_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry() -> dict:
    """The BENCHMARKS dict from models/data.py."""
    return _load_file(models_dir() / "data.py").BENCHMARKS


@dataclass
class Benchmark:
    """One benchmark at one size."""

    name: str
    size: str
    model_cls: type
    input_shapes: list[tuple[int, ...]]
    dtype: str
    init_kwargs: dict = field(default_factory=dict)
    weights: str = "inline"  # "args": parameters/buffers become forward() arguments

    def build(self):
        """The model as compiled: seeded construction, eval mode."""
        import torch

        torch.manual_seed(SEED)
        return self.model_cls(**self.init_kwargs).eval()


def load_benchmark(name: str, size: str = "small") -> Benchmark:
    """Look up `name` in models/data.py and load its model class."""
    benchmarks = registry()
    if name not in benchmarks:
        raise KeyError(f"unknown benchmark '{name}'; known: {', '.join(sorted(benchmarks))}")
    entry = benchmarks[name]
    if size not in entry["inputs"]:
        raise KeyError(
            f"benchmark '{name}' has no size '{size}'; known: {', '.join(entry['inputs'])}"
        )
    file_part, cls_name = entry["model"].split(":")
    module = _load_file(models_dir() / file_part)
    model_cls = getattr(module, cls_name, None)
    if model_cls is None:
        raise AttributeError(f"{file_part} has no class '{cls_name}'")
    return Benchmark(
        name=name,
        size=size,
        model_cls=model_cls,
        input_shapes=[tuple(s) for s in entry["inputs"][size]],
        dtype=entry.get("dtype", "f32"),
        init_kwargs=dict(entry.get("init", {}).get(size, {})),
        weights=entry.get("weights", "inline"),
    )
