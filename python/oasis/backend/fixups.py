"""Text fix-ups between CIRCT's Calyx export and the Calyx compiler.

Each fix-up takes the previous stage's file and writes a corrected copy. They work on text
because no CIRCT Python bindings are built, and they fail loudly rather than guess.
"""

from __future__ import annotations

import json
import re
import struct
from hashlib import sha256
from importlib import resources
from pathlib import Path

from oasis.config import Toolchain, load_toolchain
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


def calyx_float_consts(
    in_path: Path,
    out_path: Path,
    ws: Workspace,
    log: Path,
    toolchain: Toolchain | None = None,
) -> None:
    """Fix-up stage: exact float constants in the exported .futil."""
    consts = exact_float_consts(ws.find("calyx").read_text())
    text = in_path.read_text()
    try:
        fixed = float_consts_to_bits(text, consts)
        if 'import "primitives/float/divSqrtFN.futil";' in fixed:
            tc = toolchain if toolchain is not None else load_toolchain()
            fixed = dynamic_divsqrt_import(fixed, tc.path("calyx_lib"))
    except ValueError as e:
        raise StageError("futil", str(e), log) from e
    n = len(_FUTIL_FLOAT_CONST_RE.findall(text))
    log.write_text(f"replaced {n} std_float_const with exact std_const bits\n")
    out_path.write_text(fixed)


def dynamic_divsqrt_import(futil: str, calyx_lib: Path) -> str:
    """Remove the divider's fixed interval before Calyx can promote its control.

    Inline its extern declaration with a config-resolved RTL path. HardFloat
    and the installed Calyx tree are untouched; prepare_rtl repairs the wrapper.
    """
    directive = 'import "primitives/float/divSqrtFN.futil";'
    if directive not in futil:
        return futil
    path = calyx_lib / "primitives" / "float" / "divSqrtFN.futil"
    declaration = path.read_text()
    digest = sha256(re.sub(r"\s+", "", declaration).encode()).hexdigest()
    if digest != "78b50771d1dcc3c3ed0ff08d4b24f1e6a18a18b4eda0b0797b4b57ddedebda04":
        raise ValueError("unrecognized Calyx divider declaration; review its timing contract")
    declaration = declaration.replace("@interval(32) ", "")
    rtl_path = json.dumps(str(path.with_suffix(".sv").resolve()))
    declaration = declaration.replace('"divSqrtFN.sv"', rtl_path)
    futil = futil.replace(directive, "")
    imports = list(re.finditer(r'^import "[^"\n]+";\s*', futil, re.MULTILINE))
    position = imports[-1].end() if imports else 0
    return futil[:position] + declaration + "\n" + futil[position:]


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


_DIVSQRT_RE = re.compile(r"\bmodule\s+std_divSqrtFN\s*#\(.*?\bendmodule\b", re.DOTALL)
# Calyx 3e595cefd8d4f47c4e9b51feae56651b1874ee94, after morty expands macros.
# Ignore formatting/comments, but fail closed on any unexpected code change.
_DIVSQRT_ORIGINAL_SHA = "f1e3c834a363bdf1ec7c5610ea76a3622778b7f0f38afc6007bf03803ce82cd0"


def _sv_code(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL))


def repair_divsqrt_handshake(sv: str) -> tuple[str, int]:
    """Replace only the pinned faulty Calyx divider wrapper, leaving HardFloat intact.

    The replacement accepts one request, latches output/flags on outValid, and
    retains done until go falls. It is shared by simulation and synthesis.
    Reapplying is safe; a changed upstream wrapper requires explicit review.
    """
    replacement = (
        resources.files("oasis.backend")
        .joinpath("templates", "divsqrt_handshake.sv")
        .read_text()
        .strip()
    )
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        code = _sv_code(match.group())
        if code == _sv_code(replacement):
            return match.group()
        if sha256(code.encode()).hexdigest() != _DIVSQRT_ORIGINAL_SHA:
            raise ValueError(
                "unrecognized std_divSqrtFN wrapper; review the pinned Calyx "
                "divider handshake before generating RTL"
            )
        count += 1
        return replacement

    fixed = _DIVSQRT_RE.sub(replace, sv)
    if re.search(r"\bmodule\s+std_divSqrtFN\b", sv) and not _DIVSQRT_RE.search(sv):
        raise ValueError("unrecognized std_divSqrtFN module declaration")
    return fixed, count


def prepare_rtl(sv: str) -> tuple[str, str]:
    """Apply the same correctness/compatibility repairs to both RTL outputs."""
    fixed, dividers = repair_divsqrt_handshake(sv)
    fixed, ports = drop_port_redeclarations(fixed)
    return fixed, (
        f"repaired {dividers} Calyx divider handshake(s)\n"
        f"removed {ports} redundant port re-declaration(s) (wire X; for port X)\n"
    )


# Comments and strings are skipped so `module` inside them doesn't count as a keyword.
_TOKEN_RE = re.compile(
    r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\\n])*"|\b(endmodule)\b|\b(module)\s+([A-Za-z_]\w*)',
    re.DOTALL,
)


def sort_modules(sv: str) -> str:
    """Rewrite RTL in a fixed order: text between modules first, then modules by name.

    Calyx bundles its primitives (HardFloat) with morty, which emits them in a different
    order on every run. Yosys's LUT mapping depends on that order (gemm: 2772 to 2957 LUTs
    over five runs of the same design), so synthesis RTL gets a fixed order. The text
    between modules (HardFloat's `define constants, comments) moves to the top, identical
    copies once, in sorted order: macros are global, so defining them first is safe.
    morty's `// Compiled by morty ... <timestamp>` line is dropped.
    """
    modules: list[tuple[str, str]] = []
    between: list[str] = []
    start, name, module_start = 0, None, 0
    for m in _TOKEN_RE.finditer(sv):
        if m.group(2) and name is None:  # `module NAME`
            between.append(sv[start : m.start()])
            name, module_start = m.group(3), m.start()
        elif m.group(1) and name is not None:  # `endmodule`
            modules.append((name, sv[module_start : m.end()]))
            start, name = m.end(), None
    if not modules:
        return sv
    between.append(sv[start:])
    kept = set()
    for text in between:
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("// Compiled by morty")
        ).strip()
        if text:
            kept.add(text)
    header = "".join(f"{text}\n\n" for text in sorted(kept))
    return header + "\n\n".join(text for _, text in sorted(modules)) + "\n"


def vivado_compat_sv(
    in_path: Path,
    out_path: Path,
    ws: Workspace,
    log: Path,
    toolchain: Toolchain | None = None,
) -> None:
    """Legacy stage name: shared RTL correctness and Vivado compatibility repairs."""
    try:
        fixed, message = prepare_rtl(in_path.read_text())
    except ValueError as e:
        raise StageError("design", str(e), log) from e
    log.write_text(message)
    out_path.write_text(fixed)


FIXUPS = {"calyx_float_consts": calyx_float_consts, "vivado_compat_sv": vivado_compat_sv}
