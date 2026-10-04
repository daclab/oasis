"""Benchmark registry, seeded inputs and golden outputs. No toolchain needed."""

import numpy as np
import pytest

from oasis.models import load_benchmark, registry

torch = pytest.importorskip("torch")

from oasis.inputs import make_inputs
from oasis.verify import compare, golden


def test_registry_entries_load():
    for name, entry in registry().items():
        for size in entry["inputs"]:
            bench = load_benchmark(name, size)
            assert isinstance(bench.build(), torch.nn.Module)


def test_gemm_inputs_follow_registry_shapes():
    bench = load_benchmark("gemm", "medium")
    inputs = make_inputs(bench)
    assert [tuple(t.shape) for t in inputs] == [(200, 240), (240, 220)]
    assert all(t.dtype == torch.float32 for t in inputs)


def test_inputs_and_golden_deterministic():
    bench = load_benchmark("gemm")
    a, b = make_inputs(bench), make_inputs(bench)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    np.testing.assert_array_equal(golden(bench.build(), a), golden(bench.build(), b))


def test_gemm_golden_matches_formula():
    bench = load_benchmark("gemm")
    a, b = make_inputs(bench)
    assert compare(golden(bench.build(), (a, b)), (a @ b).numpy(), "f32").ok


def test_unknown_benchmark_and_size():
    with pytest.raises(KeyError, match="unknown benchmark"):
        load_benchmark("nope")
    with pytest.raises(KeyError, match="no size"):
        load_benchmark("gemm", "huge")


def test_compare_int_is_bit_exact():
    x = np.arange(6, dtype=np.int32)
    assert compare(x, x, "i32").ok
    y = x.copy()
    y[3] += 1
    r = compare(y, x, "i32")
    assert not r.ok and r.mismatches == 1
