"""Generic stage plumbing: numbered dumps, logs, errors that name the failing stage."""

from __future__ import annotations

import re
import resource
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


class StageError(RuntimeError):
    """A pipeline stage failed. The message names the stage and points at its log."""

    def __init__(self, stage: str, message: str, log: Path | None = None):
        self.stage = stage
        self.log = log
        where = f" (log: {log})" if log else ""
        super().__init__(f"stage '{stage}' failed: {message}{where}")


@dataclass
class Workspace:
    """Output directory for one compilation: <out>/NN_<stage>.mlir and <out>/logs/."""

    out: Path
    dump_all: bool = False

    def __post_init__(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "logs").mkdir(exist_ok=True)

    def dump_path(self, index: int, stage: str, ext: str = ".mlir") -> Path:
        return self.out / f"{index:02d}_{stage}{ext}"

    def log_path(self, stage: str) -> Path:
        return self.out / "logs" / f"{stage}.log"

    def write_dump(self, index: int, stage: str, text: str, force: bool = False) -> Path | None:
        """Write a stage's output if --dump-all is on (or `force`). Returns the path written."""
        if not (self.dump_all or force):
            return None
        path = self.dump_path(index, stage)
        path.write_text(text)
        return path

    def clear_dumps(self) -> list[Path]:
        """Delete every NN_<stage>.* file of an earlier compile. Returns the paths removed.

        A pipeline with different stages numbers them differently, so its files would sit
        next to this run's (07_design.sv and 09_design.sv) and `find` could pick the wrong one.
        """
        stale = sorted(p for p in self.out.glob("[0-9][0-9]_*") if p.is_file())
        for path in stale:
            path.unlink()
        return stale

    def find(self, stage: str) -> Path:
        """The output file of `stage` (NN_<stage>.*), or a clear error if it is missing or
        ambiguous."""
        matches = sorted(self.out.glob(f"[0-9][0-9]_{stage}.*"))
        if not matches:
            raise StageError(
                stage, f"no output file NN_{stage}.* in {self.out}; run `oasis compile` first"
            )
        if len(matches) > 1:
            names = ", ".join(p.name for p in matches)
            raise StageError(
                stage,
                f"several output files for one stage in {self.out} ({names}), left by compiles "
                "with different pipelines; run `oasis compile` again",
            )
        return matches[0]


def _raise_stack_limit() -> None:
    """Child-process hook: stack soft limit = hard limit (unlimited if allowed)."""
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))


def run_tool(
    stage: str,
    exe: str,
    args: list[str],
    out_path: Path,
    log: Path,
    stdout_to_file: bool = False,
    unlimited_stack: bool = False,
) -> None:
    """Run an external tool, log the command and its output, raise StageError on failure.

    unlimited_stack: raise the stack limit as far as allowed (like `ulimit -s unlimited`);
    the Calyx compiler recurses over the control program and overflows the default 8 MB on
    large designs such as ResNet-18.
    """
    cmd = [exe, *args]
    preexec = _raise_stack_limit if unlimited_stack else None
    if stdout_to_file:
        with open(out_path, "w") as f:
            proc = subprocess.run(
                cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=False, preexec_fn=preexec
            )
        output = proc.stderr
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, preexec_fn=preexec)
        output = proc.stdout + proc.stderr
    log.write_text(f"$ {shlex.join(cmd)}\n\n{output}")
    if proc.returncode != 0:
        raise StageError(stage, f"{Path(exe).name} exited with {proc.returncode}", log)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise StageError(stage, f"{Path(exe).name} produced no output at {out_path}", log)


# Matches the op name at the start of an MLIR line, in custom or generic form, e.g.
#   "%0 = arith.addf ..."   "scf.for %i = ..."   "%a, %b = \"foo.bar\"(...)"
_OP_RE = re.compile(
    r'^\s*(?:%[\w.#:]+(?:\s*,\s*%[\w.#:]+)*\s*=\s*)?"?([a-z_][a-z0-9_]*)\.([a-z_][\w.]*)"?'
)


def ops_used(mlir_text: str) -> set[str]:
    """Op names (dialect.op) appearing at the start of lines in textual MLIR.

    Bare `return`/`module` (custom forms of func.return / builtin.module) carry no dialect
    prefix and are ignored. Line-based on purpose: works without any MLIR bindings.
    """
    ops = set()
    for line in mlir_text.splitlines():
        m = _OP_RE.match(line)
        if m:
            ops.add(f"{m.group(1)}.{m.group(2)}")
    return ops


def check_handoff(mlir_text: str, allowed_dialects: list[str]) -> None:
    """Raise StageError if the hand-off IR uses ops outside `allowed_dialects`."""
    bad = sorted(op for op in ops_used(mlir_text) if op.split(".")[0] not in allowed_dialects)
    if bad:
        raise StageError(
            "handoff",
            f"IR contains ops outside the hand-off dialects {allowed_dialects}: {', '.join(bad)}",
        )
