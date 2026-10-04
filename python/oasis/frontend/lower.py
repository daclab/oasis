"""Run frontend MLIR pass lists (linalg-on-tensors -> scf/memref/arith).

Two runners, selected by [frontend].runner in config/toolchain.toml:
  inprocess : torch_mlir.passmanager.PassManager (same LLVM as the exporter)
  mlir-opt  : external mlir-opt built at torch-mlir's pinned LLVM commit (tool key
              `frontend_mlir_opt`)
Never point the frontend at CIRCT's mlir-opt: the two LLVMs are decoupled on purpose.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from oasis.config import Toolchain
from oasis.stages import StageError


def pipeline_string(passes: list[str]) -> str:
    """Wrap a pass list into a textual pass pipeline anchored on builtin.module."""
    return f"builtin.module({','.join(passes)})"


def run_inprocess(mlir_text: str, passes: list[str], stage: str) -> str:
    """Parse `mlir_text`, run `passes` with torch-mlir's PassManager, return the result."""
    try:
        from torch_mlir import ir
        from torch_mlir.passmanager import PassManager
    except ImportError as e:
        raise StageError(stage, f"cannot import torch_mlir ({e})") from e

    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = False
        try:
            module = ir.Module.parse(mlir_text)
            PassManager.parse(pipeline_string(passes)).run(module.operation)
        except Exception as e:  # MLIR raises MLIRError / ValueError with diagnostics
            raise StageError(stage, str(e)) from e
        return str(module)


def run_mlir_opt(mlir_text: str, passes: list[str], stage: str, tool: str, log: Path) -> str:
    """Run `passes` with an external mlir-opt binary."""
    cmd = [tool, f"--pass-pipeline={pipeline_string(passes)}"]
    proc = subprocess.run(cmd, input=mlir_text, capture_output=True, text=True, check=False)
    log.write_text(f"$ {' '.join(cmd)}\n\n{proc.stderr}")
    if proc.returncode != 0:
        raise StageError(stage, f"{Path(tool).name} exited with {proc.returncode}", log)
    return proc.stdout


def run_frontend_passes(
    mlir_text: str, passes: list[str], stage: str, toolchain: Toolchain, log: Path
) -> str:
    """Dispatch to the configured frontend runner."""
    if toolchain.frontend_runner == "inprocess":
        log.write_text(f"inprocess pipeline: {pipeline_string(passes)}\n")
        return run_inprocess(mlir_text, passes, stage)
    if toolchain.frontend_runner == "mlir-opt":
        tool = toolchain.resolve("frontend_mlir_opt")
        return run_mlir_opt(mlir_text, passes, stage, tool, log)
    raise StageError(stage, f"unknown [frontend].runner '{toolchain.frontend_runner}'")
