"""Which Calyx memory holds which kernel argument.

Parsed from the Calyx MLIR written by hlstool (NN_calyx.mlir) as plain text, so no MLIR
bindings are needed. In `main`, the external memories look like

    calyx.seq_mem @mem_0 <[1024] x 32> [10] {external = true}
    calyx.invoke @forward_instance[arg_mem_0 = mem_0, arg_mem_1 = mem_1, ...]

hlstool turns the kernel's memref arguments *and* its internal buffers (memref.alloc) into
memories of `main`, in that order: `arg_mem_<i>` for i < #arguments are the arguments
(inputs in forward() order, then the output, from buffer-results-to-out-params); the rest
are intermediate buffers (e.g. conv activations), which start at zero and are not checked.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_COMPONENT_RE = re.compile(r"calyx\.component @(\w+)")
_SEQ_MEM_RE = re.compile(r"calyx\.seq_mem @(\w+) <\[([\d, ]+)\] x (\d+)> \[([\d, ]+)\]([^\n]*)")
_INVOKE_RE = re.compile(r"calyx\.invoke @\w+\[([^\]]*)\]")
_BINDING_RE = re.compile(r"arg_mem_(\d+)\s*=\s*(\w+)")


class MemMapError(RuntimeError):
    """The Calyx memories don't match the benchmark's inputs/outputs."""


@dataclass
class Memory:
    name: str  # memory name in Calyx `main`, also the .dat/.out file stem
    size: int  # number of words (memories are flattened)
    width: int  # bits per word
    arg_index: int  # position among the kernel's memref arguments
    role: str = ""  # "input", "output" or "buffer"
    shape: tuple[int, ...] = ()  # numpy shape of the tensor it holds


def _main_component(text: str) -> str:
    """Text of the `main` component (memories and the top-level invoke live there)."""
    starts = [(m.start(), m.group(1)) for m in _COMPONENT_RE.finditer(text)]
    for i, (pos, name) in enumerate(starts):
        if name == "main":
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            return text[pos:end]
    raise MemMapError("no `calyx.component @main` found in the Calyx IR")


def parse_memories(calyx_mlir: str) -> list[Memory]:
    """External memories of `main`, ordered by the kernel argument they are bound to."""
    main = _main_component(calyx_mlir)
    mems = {}
    for name, dims, width, _idx, rest in _SEQ_MEM_RE.findall(main):
        if "external" not in rest:
            continue
        size = int(np.prod([int(d) for d in dims.split(",")]))
        mems[name] = (size, int(width))
    invoke = _INVOKE_RE.search(main)
    if not invoke:
        raise MemMapError("no calyx.invoke in `main`; cannot tell which memory is which argument")
    out = []
    for arg_index, mem_name in _BINDING_RE.findall(invoke.group(1)):
        if mem_name not in mems:
            raise MemMapError(f"invoke binds arg_mem_{arg_index} to unknown memory '{mem_name}'")
        size, width = mems[mem_name]
        out.append(Memory(mem_name, size, width, int(arg_index)))
    return sorted(out, key=lambda m: m.arg_index)


def bind(memories: list[Memory], inputs: list[np.ndarray], output: np.ndarray) -> list[Memory]:
    """Assign roles and shapes: inputs, then the output, then internal buffers."""
    if len(memories) < len(inputs) + 1:
        raise MemMapError(
            f"expected at least {len(inputs)} input memories + 1 output memory, the design "
            f"has {len(memories)} ({', '.join(m.name for m in memories)})"
        )
    tensors = [*inputs, output]
    for mem in memories[len(tensors) :]:
        mem.role = "buffer"
        mem.shape = (mem.size,)
    for i, (mem, tensor) in enumerate(zip(memories, tensors, strict=False)):
        mem.role = "output" if i == len(inputs) else "input"
        mem.shape = tuple(tensor.shape)
        if mem.size != tensor.size:
            raise MemMapError(
                f"{mem.name} (arg {mem.arg_index}) has {mem.size} words but the {mem.role} "
                f"tensor {tuple(tensor.shape)} has {tensor.size} elements"
            )
        if mem.width != tensor.dtype.itemsize * 8:
            raise MemMapError(
                f"{mem.name} is {mem.width}-bit but the {mem.role} tensor is {tensor.dtype}"
            )
    return memories


def save(memories: list[Memory], path: Path) -> None:
    path.write_text(json.dumps([asdict(m) for m in memories], indent=2) + "\n")


def load(path: Path) -> list[Memory]:
    return [Memory(**{**m, "shape": tuple(m["shape"])}) for m in json.loads(path.read_text())]
