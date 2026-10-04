"""numpy <-> hex memory files read by $readmemh and written by $writememh.

Each memory is a flat array of `width`-bit words (memories are flattened before Calyx), one
hex word per line. f32 is stored as its raw IEEE-754 bits, so no precision is lost; integers
as two's complement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def to_words(array: np.ndarray, dtype: str, width: int = 32) -> np.ndarray:
    """Flatten (row-major) and convert to unsigned `width`-bit words."""
    flat = np.ascontiguousarray(array).reshape(-1)
    if dtype == "f32":
        if width != 32:
            raise ValueError(f"f32 needs 32-bit words, memory has {width}")
        return flat.astype(np.float32).view(np.uint32).astype(np.uint64)
    if dtype in ("i32", "i64", "i16", "i8"):
        return flat.astype(np.int64).astype(np.uint64) & np.uint64((1 << width) - 1)
    raise ValueError(f"unsupported dtype '{dtype}'")


def from_words(words: np.ndarray, dtype: str, width: int = 32) -> np.ndarray:
    """Inverse of to_words: unsigned words -> flat array of `dtype`."""
    words = np.asarray(words, dtype=np.uint64)
    if dtype == "f32":
        return words.astype(np.uint32).view(np.float32)
    if dtype in ("i32", "i64", "i16", "i8"):
        signed = words.astype(np.int64)
        sign_bit = 1 << (width - 1)
        return np.where(signed >= sign_bit, signed - (1 << width), signed).astype(
            {"i32": np.int32, "i64": np.int64, "i16": np.int16, "i8": np.int8}[dtype]
        )
    raise ValueError(f"unsupported dtype '{dtype}'")


def write_dat(path: Path, array: np.ndarray, dtype: str, width: int = 32) -> None:
    """Write `array` as one zero-padded hex word per line."""
    digits = (width + 3) // 4
    words = to_words(array, dtype, width)
    path.write_text("".join(f"{int(w):0{digits}x}\n" for w in words))


def read_words(path: Path, size: int) -> np.ndarray:
    """Read a $writememh dump into `size` unsigned words.

    Skips blank lines and // comments; honours `@<hex address>` lines (some simulators emit
    them). Raises on X/Z values or words that were never written.
    """
    words = np.zeros(size, dtype=np.uint64)
    seen = np.zeros(size, dtype=bool)
    addr = 0
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        for token in line.split():
            if token.startswith("@"):
                addr = int(token[1:], 16)
                continue
            if any(c in token.lower() for c in "xz"):
                raise ValueError(f"{path}:{lineno}: undefined value '{token}' at address {addr}")
            if addr >= size:
                raise ValueError(f"{path}:{lineno}: address {addr} beyond memory size {size}")
            words[addr] = int(token, 16)
            seen[addr] = True
            addr += 1
    if not seen.all():
        raise ValueError(f"{path}: only {int(seen.sum())} of {size} words present")
    return words


def read_out(path: Path, shape: tuple[int, ...], dtype: str, width: int = 32) -> np.ndarray:
    """Read a memory dump and return it as an array of `shape` and `dtype`."""
    size = int(np.prod(shape))
    return from_words(read_words(path, size), dtype, width).reshape(shape)
