"""Unit tests for config loading. No toolchain needed."""

import pytest

from oasis.config import Toolchain, ToolNotFoundError, env_var_for, load_pipeline, load_toolchain


def test_env_var_name():
    assert env_var_for("circt_opt") == "OASIS_CIRCT_OPT"


def test_local_overrides_global(tmp_path):
    (tmp_path / "toolchain.toml").write_text(
        '[tools]\na = "x"\nb = "y"\n[sim]\nsimulator = "verilator"\n'
    )
    (tmp_path / "toolchain.local.toml").write_text(
        '[tools]\nb = "z"\n[sim]\nsimulator = "icarus"\n'
    )
    tc = load_toolchain(tmp_path)
    assert tc.tools == {"a": "x", "b": "z"}
    assert tc.simulator == "icarus"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("OASIS_MY_TOOL", "sh")  # `sh` is on PATH everywhere
    tc = Toolchain(tools={"my_tool": "definitely-not-a-real-binary"})
    assert tc.resolve("my_tool").endswith("sh")


def test_missing_tool_message_names_key_and_env():
    tc = Toolchain(tools={"circt_opt": "definitely-not-a-real-binary"})
    with pytest.raises(ToolNotFoundError) as e:
        tc.resolve("circt_opt")
    msg = str(e.value)
    assert "circt_opt" in msg and "OASIS_CIRCT_OPT" in msg and "toolchain.local.toml" in msg
    assert not tc.available("circt_opt")


def test_v0_pipeline_loads():
    p = load_pipeline()
    assert p.stages[0].name == "linalg"
    assert p.stages[0].kind == "export"
    assert sum(s.handoff for s in p.stages) == 1
    assert set(p.handoff_dialects) >= {"func", "scf", "memref", "arith"}


def test_default_simulator_is_verilator(monkeypatch):
    monkeypatch.delenv("OASIS_SIMULATOR", raising=False)
    assert load_toolchain().simulator == "verilator"


def test_pinned_versions_agree():
    """config/constraints.txt pins the same Python packages as config/versions.toml."""
    import tomllib

    from oasis.config import REPO_ROOT

    versions = tomllib.loads((REPO_ROOT / "config" / "versions.toml").read_text())
    pinned = versions["frontend"] | versions["dev"]
    lines = (REPO_ROOT / "config" / "constraints.txt").read_text().splitlines()
    constraints = dict(line.split("==") for line in lines if line and not line.startswith("#"))
    assert constraints == {k: v for k, v in pinned.items() if k != "python"}
    for section in ("circt", "llvm", "calyx"):
        assert len(versions["backend"][section]["commit"]) == 40  # full hashes, not short ones
