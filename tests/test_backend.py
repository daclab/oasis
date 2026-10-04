"""Testbench/script generation and Vivado report parsing. Never runs Vivado."""

import os
import shutil

import pytest

from oasis.backend.testbench import render_tb
from oasis.backend.vivado import fmax_mhz, parse_sim_log, parse_utilization, parse_wns
from oasis.cli import main
from oasis.config import ToolNotFoundError, load_toolchain


def test_tb_clock_and_limit():
    tb = render_tb(clock_mhz=200, cycle_limit=1234)
    assert "HALF_PERIOD_NS = 2.5;" in tb
    assert "CYCLE_LIMIT = 1234;" in tb
    assert "@@" not in tb
    assert "main main (" in tb


def test_parse_sim_log():
    assert parse_sim_log("blah\nSimulated 123456 cycles\n") == {"cycles": 123456, "timeout": False}
    assert parse_sim_log("TIMEOUT: reached limit of 10 cycles without done") == {
        "cycles": None,
        "timeout": True,
    }


UTIL_US = """
1. CLB Logic
------------

+----------------------------+-------+-------+------------+-----------+-------+
|          Site Type         |  Used | Fixed | Prohibited | Available | Util% |
+----------------------------+-------+-------+------------+-----------+-------+
| CLB LUTs*                  |  4321 |     0 |          0 |   1303680 |  0.33 |
|   LUT as Logic             |  4300 |     0 |          0 |   1303680 |  0.33 |
| CLB Registers              |  2100 |     0 |          0 |   2607360 |  0.08 |
+----------------------------+-------+-------+------------+-----------+-------+
| Block RAM Tile |    0 |     0 |          0 |      2016 |  0.00 |
| URAM           |    0 |     0 |          0 |       960 |  0.00 |
| DSPs           |    5 |     0 |          0 |      9024 |  0.06 |
"""

UTIL_7SERIES = """
| Slice LUTs*                |  999 |     0 |          0 |     53200 |  1.88 |
| Slice Registers            |  512 |     0 |          0 |    106400 |  0.48 |
| Block RAM Tile    |  1.5 |     0 |          0 |       140 |  1.07 |
| DSPs           |    2 |     0 |          0 |       220 |  0.91 |
"""

TIMING = """
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)
    -------      -------  ---------------------  -------------------      -------
     -1.250      -80.000                    120                 5000        0.030
"""


def test_parse_utilization_ultrascale_and_7series():
    assert parse_utilization(UTIL_US) == {"lut": 4321, "ff": 2100, "bram": 0, "uram": 0, "dsp": 5}
    assert parse_utilization(UTIL_7SERIES) == {"lut": 999, "ff": 512, "bram": 1.5, "dsp": 2}


def test_parse_wns_and_fmax():
    wns = parse_wns(TIMING)
    assert wns == -1.25
    assert fmax_mhz(5.0, wns) == pytest.approx(160.0)
    assert fmax_mhz(5.0, None) is None
    assert parse_wns("no timing here") is None


def _backend_tools_available() -> bool:
    tc = load_toolchain()
    try:
        tc.path("calyx_lib")
    except ToolNotFoundError:
        return False
    tools = ("oasis_opt", "circt_opt", "hlstool", "circt_translate", "calyx")
    return all(tc.available(k) for k in tools)


def test_compile_and_tb_generation(tmp_path):
    """compile -> 07_design.sv, then tb: scripts + data. Vivado is not run here."""
    pytest.importorskip("torch_mlir")
    if not _backend_tools_available():
        pytest.skip("CIRCT/Calyx tools not configured")
    out = tmp_path / "gemm"
    assert main(["compile", "gemm", "--out", str(out), "--no-tb"]) == 0
    assert (out / "07_design.sv").exists()
    assert (out / "04_scf_flat.mlir").exists()  # oasis-opt stage
    assert not (out / "sim").exists()
    # The tb step writes the Vivado scripts, so it needs the Vivado tool paths to resolve.
    tc = load_toolchain()
    if not all(tc.available(k) for k in ("vivado", "xvlog", "xelab", "xsim")):
        pytest.skip("Vivado not configured")
    assert main(["tb", "gemm", "--out", str(out)]) == 0
    for f in ("tb.sv", "mem_0.dat", "mem_1.dat", "mem_2.dat", "memmap.json", "run_xsim.sh"):
        assert (out / "sim" / f).exists(), f
    for f in ("design_synth.sv", "synth.tcl", "clocks.xdc", "run_synth.sh"):
        assert (out / "synth" / f).exists(), f
    assert "readmemh" not in (out / "synth" / "design_synth.sv").read_text()
    assert "-top forward" in (out / "synth" / "synth.tcl").read_text()


@pytest.mark.skipif(
    os.environ.get("OASIS_RUN_VIVADO") != "1" or shutil.which("bash") is None,
    reason="Vivado runs are opt-in: set OASIS_RUN_VIVADO=1",
)
def test_vivado_sim_e2e(tmp_path):
    """Full flow including xsim. Opt-in only: slow and needs Vivado."""
    out = tmp_path / "gemm"
    assert main(["compile", "gemm", "--out", str(out)]) == 0  # includes tb
    assert main(["sim", "gemm", "--out", str(out)]) == 0


def test_tb_has_no_always_ff():
    """xsim rejects always_ff variables that the initial block also drives (VRFC 10-3818)."""
    assert "always_ff @" not in render_tb(clock_mhz=200, cycle_limit=10)


def _fake_vivado(monkeypatch, sim_ok: bool = True):
    """Replace the Vivado script runner: write what xsim/synthesis would produce."""
    import numpy as np

    from oasis.backend import memmap, vivado
    from oasis.data import write_dat

    calls = []

    def fake_run_script(script, stage):
        calls.append(stage)
        d = script.parent
        if stage == "sim":
            (d / "sim.log").write_text("Simulated 1234 cycles\n")
            golden = np.load(d.parent / "golden.npy")
            for mem in memmap.load(d / "memmap.json"):
                data = golden if mem.role == "output" else np.zeros(mem.shape, np.float32)
                if mem.role == "output" and not sim_ok:
                    data = golden + 1.0
                write_dat(d / f"{mem.name}.out", data, "f32")
        else:
            (d / "utilization.rpt").write_text(UTIL_US)
            (d / "timing.rpt").write_text(TIMING)

    monkeypatch.setattr(vivado, "run_script", fake_run_script)
    return calls


def _need_backend():
    pytest.importorskip("torch_mlir")
    if not _backend_tools_available():
        pytest.skip("CIRCT/Calyx tools not configured")
    tc = load_toolchain()
    if not all(tc.available(k) for k in ("vivado", "xvlog", "xelab", "xsim")):
        pytest.skip("Vivado paths not configured (needed to write the scripts)")


def test_compile_sim_synth_flags(tmp_path, monkeypatch, capsys):
    """`oasis compile gemm --sim --synth` runs tb, then xsim + check, then synthesis."""
    _need_backend()
    calls = _fake_vivado(monkeypatch)
    out = tmp_path / "gemm"
    assert main(["compile", "gemm", "--out", str(out), "--sim", "--syn"]) == 0
    assert calls == ["sim", "synth"]
    text = capsys.readouterr().out
    assert "sim:   1234 cycles" in text and "check: PASS" in text and "LUT=4321" in text


def test_compile_sim_reports_failure(tmp_path, monkeypatch):
    _need_backend()
    _fake_vivado(monkeypatch, sim_ok=False)
    out = tmp_path / "gemm"
    assert main(["compile", "gemm", "--out", str(out), "--sim"]) == 2  # check FAIL


def test_sim_needs_tb():
    assert main(["compile", "gemm", "--no-tb", "--sim"]) == 1
    assert main(["compile", "gemm", "--stop-after", "scf", "--synth"]) == 1


def test_check_reports_unfinished_simulation(tmp_path, monkeypatch, capsys):
    """A sim.log without a result means the run is still going, not a timeout."""
    _need_backend()
    out = tmp_path / "gemm"
    assert main(["compile", "gemm", "--out", str(out)]) == 0
    assert not (out / "sim" / "sim.log").exists()  # no stale log after (re)generating tb
    (out / "sim" / "sim.log").write_text("# xsim oasis_tb -runall\n")  # started, not done
    assert main(["check", "gemm", "--out", str(out)]) == 1
    assert "not finished yet" in capsys.readouterr().err


def test_verilator_script(tmp_path):
    """run_verilator.sh uses fud2's flags and the testbench plusargs."""
    from oasis.backend.verilator import write_verilator_script
    from oasis.config import Toolchain

    tc = Toolchain(tools={"verilator": "sh"}, cycle_limit=123)  # `sh` stands in for verilator
    design = tmp_path / "07_design.sv"
    design.write_text("module main; endmodule\n")
    text = write_verilator_script(tmp_path, design, tc).read_text()
    assert "--binary --top-module toplevel -fno-inline" in text
    assert "./verilator_obj/Vtoplevel" in text and "+DATA=$PWD" in text
    assert "CYCLE_LIMIT:-123" in text and "@@" not in text


def test_sim_uses_selected_simulator(tmp_path, monkeypatch):
    """`oasis compile --sim --simulator verilator` runs run_verilator.sh."""
    _need_backend()
    from oasis.backend import vivado
    from oasis.config import Toolchain

    calls = _fake_vivado(monkeypatch)
    ran = []
    real = vivado.run_script

    def record(script, stage):
        ran.append(script.name)
        real(script, stage)

    monkeypatch.setattr(vivado, "run_script", record)
    real_available, real_resolve = Toolchain.available, Toolchain.resolve
    monkeypatch.setattr(  # pretend Verilator is installed; everything else is real
        Toolchain, "available", lambda self, k: k == "verilator" or real_available(self, k)
    )
    monkeypatch.setattr(
        Toolchain,
        "resolve",
        lambda self, k: "/bin/true" if k == "verilator" else real_resolve(self, k),
    )
    out = tmp_path / "gemm"
    assert main(["compile", "gemm", "--out", str(out), "--sim", "--simulator", "verilator"]) == 0
    assert ran == ["run_verilator.sh"] and calls == ["sim"]
