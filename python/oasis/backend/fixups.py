"""Text fix-ups between CIRCT's Calyx export and the Calyx compiler.

Each fix-up takes the previous stage's file and writes a corrected copy. They work on text
because no CIRCT Python bindings are built, and they fail loudly rather than guess.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from oasis.stages import StageError, Workspace

# calyx.constant @cst_1 <0xFF800000 : f32> : i32      (hex for values MLIR prints that way)
# calyx.constant @cst_0 <0.000000e+00 : f32> : i32
_MLIR_FLOAT_CONST_RE = re.compile(r"calyx\.constant @(\w+) <([^:>]+?)\s*:\s*f(16|32|64)>")
# cst_1 = std_float_const(0, 32, -inf);
_FUTIL_FLOAT_CONST_RE = re.compile(r"(\b\w+) = std_float_const\(\s*(\d+)\s*,\s*(\d+)\s*,[^)]*\);")

_PACK = {"32": (">f", ">I"), "64": (">d", ">Q")}


def _float_bits(text: str, width: str) -> int:
    """IEEE-754 bits of an MLIR float literal (hex bit pattern or decimal)."""
    text = text.strip()
    if text.lower().startswith("0x"):
        return int(text, 16)
    if width not in _PACK:
        raise ValueError(f"f{width} constants are not supported")
    fmt_f, fmt_i = _PACK[width]
    return struct.unpack(fmt_i, struct.pack(fmt_f, float(text)))[0]


def exact_float_consts(calyx_mlir: str) -> dict[str, tuple[int, int]]:
    """{constant name: (width, IEEE bits)} from hlstool's Calyx MLIR."""
    return {
        name: (int(width), _float_bits(value, width))
        for name, value, width in _MLIR_FLOAT_CONST_RE.findall(calyx_mlir)
    }


def float_consts_to_bits(futil: str, consts: dict[str, tuple[int, int]]) -> str:
    """Replace every `std_float_const` with `std_const` holding the exact IEEE bits.

    circt-translate prints float constants with printf("%f"), which is lossy (1e-7 becomes
    0.000000) and can print `-inf`; the Calyx parser only accepts unsigned decimal floats.
    The float primitives consume raw bits, so an integer constant of the same width is
    equivalent and exact.
    """

    def replace(m: re.Match) -> str:
        name, rep, width = m.group(1), m.group(2), int(m.group(3))
        if rep != "0":
            raise ValueError(f"{name}: unsupported float representation {rep} (0 = IEEE-754)")
        if name not in consts:
            raise ValueError(f"{name}: no exact value found in the Calyx MLIR")
        c_width, bits = consts[name]
        if c_width != width:
            raise ValueError(f"{name}: width {width} in .futil but f{c_width} in Calyx MLIR")
        return f"{name} = std_const({width}, {bits});"

    return _FUTIL_FLOAT_CONST_RE.sub(replace, futil)


def calyx_float_consts(in_path: Path, out_path: Path, ws: Workspace, log: Path) -> None:
    """Fix-up stage: exact float constants in the exported .futil."""
    consts = exact_float_consts(ws.find("calyx").read_text())
    text = in_path.read_text()
    try:
        fixed = float_consts_to_bits(text, consts)
    except ValueError as e:
        raise StageError("futil", str(e), log) from e
    n = len(_FUTIL_FLOAT_CONST_RE.findall(text))
    log.write_text(f"replaced {n} std_float_const with exact std_const bits\n")
    out_path.write_text(fixed)


_MODULE_RE = re.compile(r"\bmodule\b(.*?)\bendmodule\b", re.DOTALL)
_PORT_RE = re.compile(
    r"\b(?:input|output|inout)\b\s*(?:wire|reg|logic)?\s*(?:signed)?\s*(?:\[[^\]]*\]\s*)*(\w+)"
)


def drop_port_redeclarations(sv: str) -> tuple[str, int]:
    """Remove body `wire X;` lines that re-declare an ANSI port X of the same module.

    Calyx's HardFloat divider (divSqrtRecFN_small) declares `output sqrtOpOut` in its port
    list and then `wire sqrtOpOut;` in the body. Icarus/Verilator accept that; Vivado
    rejects it (VRFC 10-9336). The port is already a net, so the line is redundant.
    Only exact `wire NAME;` lines (no range, no assignment) are removed.
    """
    removed = 0

    def fix_module(m: re.Match) -> str:
        nonlocal removed
        text = m.group(0)
        header_end = text.find(");")
        if header_end < 0:
            return text
        ports = set(_PORT_RE.findall(text[:header_end]))
        body = text[header_end:]
        pattern = re.compile(r"^[ \t]*wire[ \t]+(\w+)[ \t]*;[ \t]*\n", re.MULTILINE)

        def drop(w: re.Match) -> str:
            nonlocal removed
            if w.group(1) in ports:
                removed += 1
                return ""
            return w.group(0)

        return text[:header_end] + pattern.sub(drop, body)

    return _MODULE_RE.sub(fix_module, sv), removed


def vivado_compat_sv(in_path: Path, out_path: Path, ws: Workspace, log: Path) -> None:
    """Fix-up stage: make Calyx's SystemVerilog acceptable to Vivado (xvlog / synthesis)."""
    fixed, n = drop_port_redeclarations(in_path.read_text())
    log.write_text(f"removed {n} redundant port re-declaration(s) (wire X; for port X)\n")
    out_path.write_text(fixed)


FIXUPS = {"calyx_float_consts": calyx_float_consts, "vivado_compat_sv": vivado_compat_sv}
