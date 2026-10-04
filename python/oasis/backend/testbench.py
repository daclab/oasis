"""Generate the simulation directory: tb.sv, memory init files, memory map."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np

from oasis.backend import memmap
from oasis.backend.memmap import Memory
from oasis.data import write_dat

RESET_CYCLES = 3


def fill_template(name: str, values: dict[str, object]) -> str:
    """Load templates/<name> and replace every @@KEY@@; fail if any placeholder is left."""
    text = resources.files("oasis.backend").joinpath("templates", name).read_text()
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", str(value))
    if "@@" in text:
        left = text[text.index("@@") : text.index("@@") + 40]
        raise ValueError(f"unfilled placeholder in {name}: {left!r}")
    return text


def render_tb(clock_mhz: float, cycle_limit: int) -> str:
    """tb.sv for a clock of `clock_mhz` and a default cycle limit."""
    return fill_template(
        "tb.sv",
        {
            "HALF_PERIOD_NS": f"{500.0 / clock_mhz:g}",
            "CLOCK_MHZ": f"{clock_mhz:g}",
            "RESET_CYCLES": RESET_CYCLES,
            "CYCLE_LIMIT": cycle_limit,
        },
    )


def write_sim_dir(
    sim_dir: Path,
    memories: list[Memory],
    inputs: list[np.ndarray],
    dtype: str,
    clock_mhz: float,
    cycle_limit: int,
) -> None:
    """Write tb.sv, mem_N.dat for every memory (output and buffers start at 0), memmap.json."""
    sim_dir.mkdir(parents=True, exist_ok=True)
    # Outputs of an earlier run would be mistaken for results of the new design.
    stale_logs = [sim_dir / n for n in ("sim.log", "xvlog.log", "xelab.log", "sim_out.npy")]
    for stale in [*sim_dir.glob("*.out"), *sim_dir.glob("mem_*.dat"), *stale_logs]:
        stale.unlink(missing_ok=True)
    (sim_dir / "tb.sv").write_text(render_tb(clock_mhz, cycle_limit))
    for mem in memories:
        if mem.role == "input":
            data = inputs[mem.arg_index]
        else:  # output, or an intermediate buffer
            data = np.zeros(mem.shape, dtype=np.float32 if dtype == "f32" else np.int64)
        write_dat(sim_dir / f"{mem.name}.dat", data, dtype, mem.width)
    memmap.save(memories, sim_dir / "memmap.json")
