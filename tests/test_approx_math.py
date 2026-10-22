"""Check the emitted arithmetic against an independent exponential reference."""

import numpy as np
import pytest

from oasis.frontend.lower import run_inprocess, run_mlir_opt
from oasis.stages import StageError, check_handoff


def evaluate_exp(values):
    """Interpret the actual emitted scalar IR over a numpy array of f32 inputs."""
    pytest.importorskip("torch_mlir")
    from torch_mlir import ir

    source = """module { func.func @exp(%x: f32) -> f32 {
      %y = math.exp %x : f32
      return %y : f32
    }}"""
    lowered = run_inprocess(source, ["oasis-approx-math"], "test")
    check_handoff(lowered, ["builtin", "func", "arith"])
    with ir.Context():
        module = ir.Module.parse(lowered)
        block = module.body.operations[0].regions[0].blocks[0]
        env = {block.arguments[0]: np.asarray(values, dtype=np.float32)}
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for op in block.operations:
                name = op.operation.name
                args = [env[value] for value in op.operands]
                if name == "func.return":
                    return env[op.operands[0]]
                if name == "arith.constant":
                    value = np.float32(ir.FloatAttr(op.attributes["value"]).value)
                elif name == "arith.addf":
                    value = args[0] + args[1]
                elif name == "arith.mulf":
                    value = args[0] * args[1]
                elif name == "arith.cmpf":
                    predicate = ir.IntegerAttr(op.attributes["predicate"]).value
                    assert predicate in (2, 4)  # OGT, OLT
                    value = args[0] > args[1] if predicate == 2 else args[0] < args[1]
                elif name == "arith.select":
                    value = np.where(*args)
                else:
                    pytest.fail(f"unexpected emitted operation: {name}")
                env[op.results[0]] = value


def test_exp_accuracy_and_special_values():
    x = np.concatenate(
        [np.linspace(-104, 88, 100001, dtype=np.float32), [-np.inf, -1e30, -0.0, 0.0]]
    ).astype(np.float32)
    actual = evaluate_exp(x)
    expected = np.exp(x.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-44)
    assert actual[-1] == actual[-2] == 1.0
    special = evaluate_exp([np.inf, 1e30, 89.0, np.nan])
    assert np.isposinf(special[:3]).all()
    assert np.isnan(special[3])


def test_softmax_accuracy():
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(256, 64)).astype(np.float32) * 10
    scores = np.concatenate([scores, np.zeros((1, 64), dtype=np.float32)])
    shifted = scores - scores.max(axis=-1, keepdims=True)
    numerators = evaluate_exp(shifted)
    actual = numerators / numerators.sum(axis=-1, keepdims=True)
    expected = np.exp(shifted.astype(np.float64))
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("size", ["small", "medium"])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_with_emitted_exp(size, causal, monkeypatch):
    torch = pytest.importorskip("torch")
    from oasis.inputs import make_inputs
    from oasis.models import load_benchmark
    from oasis.verify import compare, golden

    bench = load_benchmark("attention", size)
    model = bench.build()
    model.causal = causal
    inputs = make_inputs(bench)
    expected = golden(model, inputs)

    def softmax(scores, dim):
        shifted = scores - scores.max(dim=dim, keepdim=True).values
        numerators = torch.from_numpy(evaluate_exp(shifted.numpy()))
        return numerators / numerators.sum(dim=dim, keepdim=True)

    monkeypatch.setattr(torch.nn.functional, "softmax", softmax)
    assert compare(golden(model, inputs), expected, "f32").ok


def test_exp_rejects_unsupported_type():
    pytest.importorskip("torch_mlir")
    with pytest.raises(StageError, match="only scalar f32"):
        run_inprocess(
            "module { func.func @f(%x: f64) -> f64 { %y = math.exp %x : f64 return %y : f64 } }",
            ["oasis-approx-math"],
            "test",
        )


def test_inexact_constant_truncation_is_folded():
    pytest.importorskip("torch_mlir")
    text = run_inprocess(
        "module { func.func @f() -> f32 {"
        " %x = arith.constant 2.8284271247461903 : f64"
        " %y = arith.truncf %x : f64 to f32 return %y : f32 } }",
        ["oasis-approx-math", "canonicalize"],
        "test",
    )
    assert "arith.truncf" not in text
    assert "f64" not in text
    from torch_mlir import ir

    with ir.Context():
        module = ir.Module.parse(text)
        constant = module.body.operations[0].regions[0].blocks[0].operations[0]
        value = ir.FloatAttr(constant.attributes["value"]).value
        assert value == float(np.float32(np.sqrt(8.0)))


def test_external_runner_dispatches_custom_rewrite(tmp_path, monkeypatch):
    pytest.importorskip("torch_mlir")
    from types import SimpleNamespace

    calls = []

    def run(cmd, **kwargs):
        assert "oasis-approx-math" not in " ".join(cmd)
        calls.append(kwargs["input"])
        return SimpleNamespace(returncode=0, stdout=kwargs["input"], stderr="")

    monkeypatch.setattr("oasis.frontend.lower.subprocess.run", run)
    log = tmp_path / "frontend.log"
    result = run_mlir_opt(
        "module { func.func @f(%x: f32) -> f32 { %y = math.exp %x : f32 return %y : f32 } }",
        ["canonicalize", "oasis-approx-math", "cse"],
        "test",
        "frontend-mlir-opt",
        log,
    )
    assert len(calls) == 2
    assert "math.exp" in calls[0] and "math.exp" not in calls[1]
    assert "math.exp" not in result
    assert "oasis-approx-math" in log.read_text()
