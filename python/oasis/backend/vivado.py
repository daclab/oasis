"""Vivado: write xsim/synthesis scripts, run them, and parse their logs and reports.

OASIS writes plain bash + Tcl scripts so the same run can be repeated by hand:
    bash out/<bench>_<size>/sim/run_xsim.sh
    bash out/<bench>_<size>/synth/run_synth.sh
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from oasis.backend.fixups import drop_port_redeclarations, sort_modules
from oasis.backend.testbench import fill_template
from oasis.config import FpgaTarget, Toolchain
from oasis.stages import StageError, run_tool


def _write(path: Path, text: str, executable: bool = False) -> Path:
    path.write_text(text)
    if executable:
        path.chmod(0o755)
    return path


def write_xsim_script(sim_dir: Path, design_sv: Path, tc: Toolchain) -> Path:
    """sim/run_xsim.sh: compile, elaborate and run tb.sv + the design in xsim."""
    text = fill_template(
        "run_xsim.sh",
        {
            "SIM_DIR": sim_dir,
            "XVLOG": tc.resolve("xvlog"),
            "XELAB": tc.resolve("xelab"),
            "XSIM": tc.resolve("xsim"),
            "DESIGN_SV": design_sv.resolve(),
            "CYCLE_LIMIT": tc.cycle_limit,
        },
    )
    return _write(sim_dir / "run_xsim.sh", text, executable=True)


def generate_synth_sv(
    futil: Path, out_sv: Path, tc: Toolchain, log: Path, disable_verify: bool = False
) -> None:
    """Synthesis RTL: same Calyx program, without $readmemh/$writememh/final (calyx --synthesis).

    Gets the same Vivado-compatibility fix-up as the simulation RTL, and a fixed module order
    (Calyx's order changes from run to run, and synthesis results depend on it).
    `disable_verify` drops Calyx's `$fatal` multiple-assignment checks (calyx
    --disable-verify), which Yosys's SystemVerilog frontend doesn't need; Vivado's synthesis
    ignores them.
    """
    flags = ["--synthesis", "--disable-verify"] if disable_verify else ["--synthesis"]
    run_tool(
        "synth_rtl",
        tc.resolve("calyx"),
        ["-l", str(tc.path("calyx_lib")), "-b", "verilog", *flags, str(futil)],
        out_sv,
        log,
        stdout_to_file=True,
        unlimited_stack=True,
    )
    fixed, _ = drop_port_redeclarations(out_sv.read_text())
    out_sv.write_text(sort_modules(fixed))


# Everything write_synth_scripts and a Vivado run put in synth/ (besides the RTL).
SYNTH_FILES = (
    "clocks.xdc",
    "synth.tcl",
    "run_synth.sh",
    "utilization.rpt",
    "timing.rpt",
    "synth.log",
)


def write_synth_scripts(synth_dir: Path, design_sv: Path, top: str, tc: Toolchain) -> Path:
    """synth/{clocks.xdc, synth.tcl, run_synth.sh} for out-of-context synthesis of `top`."""
    # Reports of an earlier run would be mistaken for results of the new design.
    for name in ("utilization.rpt", "timing.rpt", "synth.log"):
        (synth_dir / name).unlink(missing_ok=True)
    fpga: FpgaTarget = tc.fpga
    values = {
        "SYNTH_DIR": synth_dir,
        "VIVADO": tc.resolve("vivado"),
        "DESIGN_SV": design_sv.resolve(),
        "TOP": top,
        "PART": fpga.part,
        "CLOCK_MHZ": f"{fpga.clock_mhz:g}",
        "PERIOD_NS": f"{fpga.period_ns:.3f}",
    }
    _write(synth_dir / "clocks.xdc", fill_template("clocks.xdc", values))
    _write(synth_dir / "synth.tcl", fill_template("synth.tcl", values))
    return _write(synth_dir / "run_synth.sh", fill_template("run_synth.sh", values), True)


def run_script(script: Path, stage: str) -> None:
    """Run a generated script, streaming its output; raise StageError on failure."""
    print(f"$ bash {script}", flush=True)
    proc = subprocess.run(["bash", str(script)], check=False)
    if proc.returncode != 0:
        raise StageError(stage, f"{script.name} exited with {proc.returncode}")


# ---- log / report parsing ---------------------------------------------------------------

_CYCLES_RE = re.compile(r"Simulated\s+(-?\d+)\s+cycles")
_TIMEOUT_RE = re.compile(r"TIMEOUT: reached limit of\s+(\d+)\s+cycles")


def parse_sim_log(text: str) -> dict:
    """{"cycles": int | None, "timeout": bool} from the testbench's $display lines."""
    m = _CYCLES_RE.search(text)
    return {
        "cycles": int(m.group(1)) if m else None,
        "timeout": bool(_TIMEOUT_RE.search(text)),
    }


# Resource rows in report_utilization (UltraScale+ names first, 7-series names second).
_UTIL_ROWS = {
    "lut": ("CLB LUTs", "Slice LUTs"),
    "ff": ("CLB Registers", "Slice Registers"),
    "bram": ("Block RAM Tile",),
    "uram": ("URAM",),
    "dsp": ("DSPs",),
}


def parse_utilization(text: str) -> dict:
    """{"lut", "ff", "bram", "uram", "dsp"} -> used count (first matching table row)."""
    out: dict[str, float] = {}
    for key, names in _UTIL_ROWS.items():
        for name in names:
            m = re.search(
                rf"^\|\s*{re.escape(name)}\*?\s*\|\s*([\d.]+)\s*\|", text, flags=re.MULTILINE
            )
            if m:
                value = float(m.group(1))
                out[key] = int(value) if value.is_integer() else value
                break
    return out


def parse_wns(text: str) -> float | None:
    """Worst negative slack (ns) from the Design Timing Summary of report_timing_summary."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "WNS(ns)" in line and "TNS(ns)" in line:
            for row in lines[i + 1 : i + 6]:
                fields = row.split()
                if not fields or set(fields[0]) <= {"-"}:
                    continue
                try:
                    return float(fields[0])
                except ValueError:
                    return None  # e.g. "inf" when nothing is constrained
    return None


def fmax_mhz(period_ns: float, wns_ns: float | None) -> float | None:
    """Achievable clock from target period and slack: 1000 / (period - WNS)."""
    if wns_ns is None or period_ns - wns_ns <= 0:
        return None
    return 1000.0 / (period_ns - wns_ns)
