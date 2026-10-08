"""Yosys: an open-source alternative to Vivado synthesis (cell counts only, no timing).

`synth_xilinx -family <yosys_family>` maps the kernel to Xilinx primitives, and `stat -json`
counts them. Same RTL as Vivado's synthesis (calyx --synthesis), plus --disable-verify:
    bash out/<bench>_<size>/synth/run_yosys.sh

Counts are not Vivado's: Yosys maps carry chains to CARRY4 even for UltraScale+ (two CARRY4
cover one CARRY8's 8 bits), and its LUT counts run higher. Compare Yosys numbers with each
other.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from oasis.backend.testbench import fill_template
from oasis.config import Toolchain

RESULTS = ("stat.json", "yosys.log")
# Everything write_synth_scripts and a Yosys run put in synth/ (besides the RTL).
SYNTH_FILES = ("synth.ys", "run_yosys.sh", *RESULTS)

# report.json key -> Xilinx primitives counted under it (regular expressions).
_CELL_GROUPS = {
    "lut": r"LUT[1-6]",
    "ff": r"FD[RSCP]E(_1)?",
    "carry4": r"CARRY4",
    "dsp": r"DSP48E\d",
    "muxf": r"MUXF[789]",
    "lutram": r"RAM(16|32|64|128|256|512)(X\d+[SD]|M\d*)",
    "uram": r"URAM288(_BASE)?",
}
_BRAM_HALF, _BRAM_FULL = r"RAMB18E\d", r"RAMB36E\d"


def write_synth_scripts(synth_dir: Path, design_sv: Path, top: str, tc: Toolchain) -> Path:
    """synth/{synth.ys, run_yosys.sh} for Yosys synthesis of `top`."""
    for name in RESULTS:  # results of an earlier run would be mistaken for the new design's
        (synth_dir / name).unlink(missing_ok=True)
    values = {
        "SYNTH_DIR": synth_dir,
        "YOSYS": tc.resolve("yosys"),
        "DESIGN_SV": design_sv.resolve(),
        "TOP": top,
        "FAMILY": tc.fpga.yosys_family,
    }
    (synth_dir / "synth.ys").write_text(fill_template("synth.ys", values))
    script = synth_dir / "run_yosys.sh"
    script.write_text(fill_template("run_yosys.sh", values))
    script.chmod(0o755)
    return script


def parse_stat(text: str) -> dict:
    """Resource counts from `stat -json` output: lut, ff, carry4, dsp, muxf, lutram, bram,
    uram, plus every cell type (`cells`) so nothing is lost."""
    cells = json.loads(text)["design"]["num_cells_by_type"]
    cells = {k: v for k, v in cells.items() if not k.startswith("$")}  # $scopeinfo: not hardware

    def count(pattern: str) -> int:
        return sum(n for cell, n in cells.items() if re.fullmatch(pattern, cell))

    out = {key: count(pattern) for key, pattern in _CELL_GROUPS.items()}
    bram = count(_BRAM_FULL) + 0.5 * count(_BRAM_HALF)  # in RAMB36 tiles, like Vivado
    out["bram"] = int(bram) if bram.is_integer() else bram
    out["cells"] = dict(sorted(cells.items()))
    return out
