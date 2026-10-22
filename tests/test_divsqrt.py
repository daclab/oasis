"""Pinned Calyx divider repair: guard checks and a short real RTL regression."""

import subprocess
from pathlib import Path

import pytest

from oasis.backend.fixups import (
    dynamic_divsqrt_import,
    prepare_rtl,
    repair_divsqrt_handshake,
    vivado_compat_sv,
)
from oasis.backend.vivado import generate_synth_sv
from oasis.config import load_toolchain
from oasis.stages import Workspace, run_tool

ORIGINAL = Path(__file__).parent / "fixtures" / "calyx_divsqrt_original.sv"


def test_divider_repair_guard_and_idempotence():
    old = ORIGINAL.read_text()
    fixed, count = repair_divsqrt_handshake(old)
    assert count == 1
    assert repair_divsqrt_handshake(fixed) == (fixed, 0)
    assert repair_divsqrt_handshake("module other; endmodule") == ("module other; endmodule", 0)
    with pytest.raises(ValueError, match="unrecognized"):
        repair_divsqrt_handshake(old.replace("done_buf[31]", "done_buf[30]"))
    with pytest.raises(ValueError, match="unrecognized"):
        repair_divsqrt_handshake("module std_divSqrtFN; endmodule")


def test_both_output_paths_repair_divider(tmp_path, monkeypatch):
    """Exercise the real simulation and synthesis entry points, including idempotence."""
    old = ORIGINAL.read_text()
    source = tmp_path / "raw.sv"
    source.write_text(old)
    simulation = tmp_path / "simulation.sv"
    ws = Workspace(tmp_path)
    vivado_compat_sv(source, simulation, ws, tmp_path / "sim.log")

    def fake_codegen(stage, exe, args, out_path, log, **kwargs):
        out_path.write_text(old)
        log.write_text("fake codegen\n")

    monkeypatch.setattr("oasis.backend.vivado.run_tool", fake_codegen)
    tc = load_toolchain()
    monkeypatch.setattr(type(tc), "resolve", lambda self, key: "calyx")
    monkeypatch.setattr(type(tc), "path", lambda self, key: tmp_path)
    synthesis = tmp_path / "synth.sv"
    generate_synth_sv(tmp_path / "design.futil", synthesis, tc, tmp_path / "synth.log")
    assert simulation.read_text().strip() == synthesis.read_text().strip()
    assert "repaired 1" in (tmp_path / "synth.log").read_text()


FUTIL = """import "primitives/core.futil";
import "primitives/float/divSqrtFN.futil";
component main() -> (a: 32, b: 32, c: 32) {
  cells {
    d = std_divSqrtFN(8, 24, 32);
    r0 = std_reg(32); r1 = std_reg(32); r2 = std_reg(32);
  }
  wires {
    a = r0.out; b = r1.out; c = r2.out;
    d.control = 1'd0; d.sqrtOp = 1'd0; d.roundingMode = 3'd0;
    group divide0 {
      d.left = 32'd1065353216; d.right = 32'd1073741824;
      d.go = !d.done ? 1'd1;
      r0.in = d.out; r0.write_en = d.done;
      divide0[done] = r0.done;
    }
    group divide1 {
      d.left = 32'd1077936128; d.right = 32'd1073741824;
      d.go = !d.done ? 1'd1;
      r1.in = d.out; r1.write_en = d.done;
      divide1[done] = r1.done;
    }
    group divide2 {
      d.left = 32'd3212836864; d.right = 32'd1073741824;
      d.go = !d.done ? 1'd1;
      r2.in = d.out; r2.write_en = d.done;
      divide2[done] = r2.done;
    }
  }
  control { seq { divide0; divide1; divide2; } }
}
"""


def test_real_divider_handshake(tmp_path):
    """Small Verilator unit test; never runs Vivado, synthesis, or full attention."""
    tc = load_toolchain()
    if not all(tc.available(k) for k in ("calyx", "verilator")):
        pytest.skip("Calyx/Verilator not configured")
    lib = tc.path("calyx_lib")
    futil = tmp_path / "divider.futil"
    dynamic = dynamic_divsqrt_import(FUTIL, lib)
    assert "@interval" not in dynamic
    assert dynamic_divsqrt_import(dynamic, lib) == dynamic
    futil.write_text(dynamic)
    raw = tmp_path / "raw.sv"
    run_tool(
        "divider",
        tc.resolve("calyx"),
        ["-l", str(lib), "-b", "verilog", str(futil)],
        raw,
        tmp_path / "calyx.log",
        stdout_to_file=True,
    )
    fixed = tmp_path / "fixed.sv"
    text, message = prepare_rtl(raw.read_text())
    assert "repaired 1" in message
    fixed.write_text(text)
    tb = tmp_path / "divider_tb.sv"
    tb.write_text((Path(__file__).parent / "fixtures" / "divider_tb.sv").read_text())
    # Run the identical behavioral test against original and repaired code.
    for label, design in (("original", raw), ("fixed", fixed)):
        obj = tmp_path / label
        cmd = [
            tc.resolve("verilator"),
            str(design),
            str(tb),
            "--binary",
            "--top-module",
            "divider_tb",
            "-Wno-fatal",
            "-j",
            "2",
            "-Mdir",
            str(obj),
        ]
        with (tmp_path / f"{label}_build.log").open("w") as log:
            result = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False
            )
        assert result.returncode == 0, (tmp_path / f"{label}_build.log").read_text()[-4000:]
        result = subprocess.run(
            [str(obj / "Vdivider_tb")], capture_output=True, text=True, timeout=10, check=False
        )
        (tmp_path / f"{label}_sim.log").write_text(result.stdout + result.stderr)
        if label == "original":
            assert result.returncode != 0, "regression did not detect the original defect"
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            assert "PASS divider handshake" in result.stdout
