"""Verilator: write the simulation script (an alternative to Vivado xsim).

Same testbench (tb.sv), memory files and checks as xsim; Verilator compiles the RTL to
C++, which simulates much faster on large designs. Note that Verilator is 2-state: values
xsim would show as `x` (e.g. reads of never-written memory) come out as 0 or random bits,
so `oasis check` sees wrong numbers instead of undefined ones.
"""

from __future__ import annotations

from pathlib import Path

from oasis.backend.testbench import fill_template
from oasis.config import Toolchain


def write_verilator_script(sim_dir: Path, design_sv: Path, tc: Toolchain) -> Path:
    """sim/run_verilator.sh: build tb.sv + the design with Verilator, then run it."""
    text = fill_template(
        "run_verilator.sh",
        {
            "SIM_DIR": sim_dir,
            "VERILATOR": tc.resolve("verilator"),
            "DESIGN_SV": design_sv.resolve(),
            "CYCLE_LIMIT": tc.cycle_limit,
        },
    )
    path = sim_dir / "run_verilator.sh"
    path.write_text(text)
    path.chmod(0o755)
    return path
