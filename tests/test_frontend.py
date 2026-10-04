"""Frontend: PyTorch -> linalg -> scf hand-off IR. Skipped without torch_mlir."""

import pytest

from oasis.cli import main
from oasis.stages import StageError, check_handoff, ops_used

HANDOFF = ["builtin", "func", "scf", "memref", "arith"]


def test_ops_used_parses_custom_and_generic_forms():
    text = """
module {
  func.func @f(%arg0: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %0, %1 = "foo.bar"(%c0) : (index) -> (f32, f32)
    scf.for %i = %c0 to %c0 step %c0 {
      memref.store %0, %arg0[%i] : memref<4xf32>
    }
    return
  }
}"""
    assert ops_used(text) == {"func.func", "arith.constant", "foo.bar", "scf.for", "memref.store"}
    with pytest.raises(StageError, match="foo.bar"):
        check_handoff(text, HANDOFF)


def test_gemm_frontend_e2e(tmp_path):
    pytest.importorskip("torch_mlir")
    out = tmp_path / "gemm"
    assert main(["compile", "gemm", "--out", str(out), "--stop-after", "scf"]) == 0

    assert (out / "inputs.npz").exists() and (out / "golden.npy").exists()

    linalg = (out / "00_linalg.mlir").read_text()
    assert "linalg.matmul" in linalg and "tensor<32x32xf32>" in linalg

    scf = (out / "03_scf.mlir").read_text()
    check_handoff(scf, HANDOFF)
    ops = ops_used(scf)
    assert {"scf.for", "memref.load", "memref.store", "arith.mulf", "arith.addf"} <= ops
    # a, b, then C written in place as the last argument; no temporary buffers.
    sig = "func.func @forward(" + ", ".join(f"%arg{i}: memref<32x32xf32>" for i in range(3)) + ")"
    assert sig in scf
    assert "memref.alloc" not in scf


@pytest.mark.parametrize("bench", ["relu", "ffnn", "increment"])
def test_ported_benchmarks_reach_handoff(bench, tmp_path):
    """From-Algorithm-to-RTL benchmarks lower to scf/memref/arith."""
    pytest.importorskip("torch_mlir")
    out = tmp_path / bench
    assert main(["compile", bench, "--out", str(out), "--stop-after", "scf"]) == 0
    check_handoff((out / "03_scf.mlir").read_text(), HANDOFF)
