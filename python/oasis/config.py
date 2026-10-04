"""Load toolchain and pipeline configuration (TOML files + environment overrides)."""

from __future__ import annotations

import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Directory holding toolchain.toml and pipelines/ (override with OASIS_CONFIG_DIR)."""
    return Path(os.environ.get("OASIS_CONFIG_DIR", REPO_ROOT / "config"))


class ToolNotFoundError(RuntimeError):
    """A required external tool or configured path could not be resolved."""


def env_var_for(key: str) -> str:
    """Environment variable that overrides tool `key`, e.g. circt_opt -> OASIS_CIRCT_OPT."""
    return f"OASIS_{key.upper()}"


def _merge(base: dict, extra: dict) -> dict:
    """Recursively merge `extra` into a copy of `base`."""
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class FpgaTarget:
    """FPGA part, clock and synthesis top from [fpga]."""

    part: str = "xcu55c-fsvh2892-2L-e"
    clock_mhz: float = 200.0
    synth_top: str = "{top}"

    @property
    def period_ns(self) -> float:
        return 1000.0 / self.clock_mhz


@dataclass
class Toolchain:
    """Resolved toolchain settings."""

    tools: dict[str, str] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    frontend_runner: str = "inprocess"
    simulator: str = "xsim"
    cycle_limit: int = 10_000_000
    fpga: FpgaTarget = field(default_factory=lambda: FpgaTarget())

    def resolve(self, key: str) -> str:
        """Return an executable path for tool `key`, or raise ToolNotFoundError.

        Order: env var OASIS_<KEY>, then config value (local overrides global), then PATH.
        """
        candidate = os.environ.get(env_var_for(key)) or self.tools.get(key) or key
        if "/" in candidate and not os.path.isabs(candidate):
            candidate = str(REPO_ROOT / candidate)  # repo-relative, e.g. build/cmake/bin/oasis-opt
        found = shutil.which(candidate)
        if found is None:
            raise ToolNotFoundError(
                f"tool '{key}' not found (tried '{candidate}'). Set [tools].{key} in "
                f"config/toolchain.local.toml or export {env_var_for(key)}=/path/to/tool"
            )
        return found

    def path(self, key: str) -> Path:
        """Return configured directory/file `key` from [paths] (env OASIS_<KEY> wins)."""
        value = os.environ.get(env_var_for(key)) or self.paths.get(key)
        if not value or not Path(value).expanduser().exists():
            raise ToolNotFoundError(
                f"path '{key}' not found (got '{value}'). Set [paths].{key} in "
                f"config/toolchain.local.toml or export {env_var_for(key)}=/path"
            )
        return Path(value).expanduser()

    def available(self, key: str) -> bool:
        """True if tool `key` resolves to an executable."""
        try:
            self.resolve(key)
        except ToolNotFoundError:
            return False
        return True


def load_toolchain(directory: Path | None = None) -> Toolchain:
    """Read toolchain.toml, layer toolchain.local.toml on top, return a Toolchain."""
    directory = directory or config_dir()
    data: dict = {}
    for name in ("toolchain.toml", "toolchain.local.toml"):
        path = directory / name
        if path.exists():
            data = _merge(data, tomllib.loads(path.read_text()))
    sim = data.get("sim", {})
    fpga = data.get("fpga", {})
    return Toolchain(
        tools=dict(data.get("tools", {})),
        paths=dict(data.get("paths", {})),
        frontend_runner=os.environ.get(
            "OASIS_FRONTEND_RUNNER", data.get("frontend", {}).get("runner", "inprocess")
        ),
        simulator=os.environ.get("OASIS_SIMULATOR", sim.get("simulator", "xsim")),
        cycle_limit=int(os.environ.get("OASIS_CYCLE_LIMIT", sim.get("cycle_limit", 10_000_000))),
        fpga=FpgaTarget(
            part=os.environ.get("OASIS_FPGA_PART", fpga.get("part", FpgaTarget.part)),
            clock_mhz=float(
                os.environ.get("OASIS_CLOCK_MHZ", fpga.get("clock_mhz", FpgaTarget.clock_mhz))
            ),
            synth_top=fpga.get("synth_top", FpgaTarget.synth_top),
        ),
    )


@dataclass
class StageSpec:
    """One stage of a pipeline config."""

    name: str
    kind: str
    passes: list[str] = field(default_factory=list)
    handoff: bool = False
    # kind = "tool" only
    tool: str = ""
    args: list[str] = field(default_factory=list)
    ext: str = ".mlir"
    stdout: bool = False


@dataclass
class Pipeline:
    """An ordered list of stages plus the hand-off dialect allow-list."""

    name: str
    stages: list[StageSpec]
    handoff_dialects: list[str]


def load_pipeline(path: Path | None = None) -> Pipeline:
    """Read a pipeline TOML (default: config/pipelines/v0.toml)."""
    path = path or config_dir() / "pipelines" / "v0.toml"
    data = tomllib.loads(Path(path).read_text())
    stages = [
        StageSpec(
            name=s["name"],
            kind=s["kind"],
            passes=list(s.get("passes", [])),
            handoff=bool(s.get("handoff", False)),
            tool=s.get("tool", ""),
            args=list(s.get("args", [])),
            ext=s.get("ext", ".mlir"),
            stdout=bool(s.get("stdout", False)),
        )
        for s in data.get("stage", [])
    ]
    meta = data.get("pipeline", {})
    return Pipeline(
        name=meta.get("name", Path(path).stem),
        stages=stages,
        handoff_dialects=list(meta.get("handoff_dialects", [])),
    )
