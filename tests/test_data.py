"""Hex memory files: exact round trips and robust parsing of $writememh dumps."""

import numpy as np
import pytest

from oasis.data import from_words, read_out, read_words, to_words, write_dat


def test_f32_round_trip_is_bit_exact(tmp_path):
    x = np.array([[0.0, -0.0, 1.5, -2.25], [np.inf, 1e-38, 3.4e38, -7.0]], dtype=np.float32)
    path = tmp_path / "mem.dat"
    write_dat(path, x, "f32")
    assert path.read_text().splitlines()[:3] == ["00000000", "80000000", "3fc00000"]
    back = read_out(path, x.shape, "f32")
    np.testing.assert_array_equal(back.view(np.uint32), x.view(np.uint32))


def test_i32_negative_two_complement(tmp_path):
    x = np.array([-1, -2147483648, 2147483647, 0], dtype=np.int32)
    path = tmp_path / "mem.dat"
    write_dat(path, x, "i32")
    assert path.read_text().split() == ["ffffffff", "80000000", "7fffffff", "00000000"]
    np.testing.assert_array_equal(read_out(path, (4,), "i32"), x)


def test_words_round_trip():
    x = np.arange(-5, 5, dtype=np.int32)
    np.testing.assert_array_equal(from_words(to_words(x, "i32"), "i32"), x)


def test_read_words_handles_comments_and_addresses(tmp_path):
    path = tmp_path / "mem.out"
    path.write_text("// memory dump\n@0\n00000001 00000002\n\n@2\n00000003 // trailing\n0004\n")
    np.testing.assert_array_equal(read_words(path, 4), [1, 2, 3, 4])


def test_read_words_rejects_x_and_missing(tmp_path):
    path = tmp_path / "mem.out"
    path.write_text("00000001\nxxxxxxxx\n")
    with pytest.raises(ValueError, match="undefined"):
        read_words(path, 2)
    path.write_text("00000001\n")
    with pytest.raises(ValueError, match="only 1 of 2"):
        read_words(path, 2)
