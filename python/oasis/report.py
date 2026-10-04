"""report.json: one summary per compiled benchmark (simulation, check, synthesis)."""

from __future__ import annotations

import json
from pathlib import Path


def update(out_dir: Path, section: str, values: dict) -> dict:
    """Merge `values` into report.json[section] and return the whole report."""
    path = out_dir / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.setdefault(section, {}).update(values)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def reset(out_dir: Path) -> None:
    """Start a fresh report.json (a new compile invalidates earlier sim/check/synth results)."""
    (out_dir / "report.json").unlink(missing_ok=True)


def load(out_dir: Path) -> dict:
    path = out_dir / "report.json"
    return json.loads(path.read_text()) if path.exists() else {}


def summary(report: dict) -> str:
    """One human-readable block for the terminal."""
    lines = []
    sim = report.get("sim", {})
    if sim:
        cycles = sim.get("cycles")
        lines.append("sim:   TIMEOUT" if sim.get("timeout") else f"sim:   {cycles} cycles")
    check = report.get("check", {})
    if check:
        lines.append(
            f"check: {'PASS' if check.get('ok') else 'FAIL'}  max|err| = {check.get('max_abs_err')}"
            f"  mismatches = {check.get('mismatches')}"
        )
    synth = report.get("synth", {})
    if synth:
        util = "  ".join(
            f"{k.upper()}={synth[k]}" for k in ("lut", "ff", "dsp", "bram", "uram") if k in synth
        )
        fmax = synth.get("fmax_mhz")
        lines.append(
            f"synth: {synth.get('top')} on {synth.get('part')} @ {synth.get('clock_mhz')} MHz"
        )
        lines.append(f"       {util}")
        lines.append(
            f"       WNS = {synth.get('wns_ns')} ns"
            + (f"  Fmax ≈ {fmax:.1f} MHz" if isinstance(fmax, float) else "")
        )
    return "\n".join(lines) if lines else "(no results yet)"
