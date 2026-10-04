"""Memory map parsed from Calyx MLIR text."""

import numpy as np
import pytest

from oasis.backend.memmap import MemMapError, bind, load, parse_memories, save

CALYX = """
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
    %mem_2.addr0, %mem_2.read_data = calyx.seq_mem @mem_2 <[1024] x 32> [10] {external = true} : i10, i32
    %mem_1.addr0, %mem_1.read_data = calyx.seq_mem @mem_1 <[1024] x 32> [10] {external = true} : i10, i32
    %mem_0.addr0, %mem_0.read_data = calyx.seq_mem @mem_0 <[1024] x 32> [10] {external = true} : i10, i32
    calyx.control {
      calyx.seq {
        calyx.invoke @forward_instance[arg_mem_0 = mem_0, arg_mem_1 = mem_1, arg_mem_2 = mem_2]() -> ()
      }
    }
  }
  calyx.component @forward(%clk: i1 {clk}) -> (%done: i1 {done}) {
    %arg_mem_0.addr0 = calyx.seq_mem @arg_mem_0 <[1024] x 32> [10] : i10
  }
}
"""


def _tensors():
    a = np.zeros((32, 32), np.float32)
    return [a, a.copy()], a.copy()


def test_parse_orders_by_argument():
    mems = parse_memories(CALYX)
    assert [(m.name, m.arg_index, m.size, m.width) for m in mems] == [
        ("mem_0", 0, 1024, 32),
        ("mem_1", 1, 1024, 32),
        ("mem_2", 2, 1024, 32),
    ]


def test_bind_roles_and_round_trip(tmp_path):
    inputs, out = _tensors()
    mems = bind(parse_memories(CALYX), inputs, out)
    assert [m.role for m in mems] == ["input", "input", "output"]
    save(mems, tmp_path / "memmap.json")
    assert load(tmp_path / "memmap.json") == mems


def test_bind_rejects_size_mismatch():
    inputs, out = _tensors()
    with pytest.raises(MemMapError, match="1024 words"):
        bind(parse_memories(CALYX), [inputs[0], np.zeros((4, 4), np.float32)], out)


def test_bind_rejects_too_few_memories():
    inputs, out = _tensors()
    with pytest.raises(MemMapError, match="expected at least 3 input"):
        bind(parse_memories(CALYX), [*inputs, inputs[0]], out)


def test_extra_memories_are_buffers():
    """hlstool also externalises internal buffers (memref.alloc); they come after the args."""
    inputs, out = _tensors()
    mems = bind(parse_memories(CALYX), inputs[:1], out)
    assert [m.role for m in mems] == ["input", "output", "buffer"]
    assert mems[2].shape == (1024,)


def test_no_main_component():
    with pytest.raises(MemMapError, match="main"):
        parse_memories("module {}")
