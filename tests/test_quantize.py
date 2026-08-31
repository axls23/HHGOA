import numpy as np

from faceanchor.evidence.quantize import BOUND, SCALE, quantize, shift_unsigned


def test_quantize_maps_unit_range_to_bound():
    e = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    q = quantize(e)
    assert q == [BOUND, -BOUND, 0]


def test_quantize_clamps_out_of_range_values():
    e = np.array([2.0, -2.0], dtype=np.float32)
    q = quantize(e)
    assert q == [BOUND, -BOUND]


def test_quantize_rounds_to_nearest_int():
    e = np.array([0.6 / SCALE, 0.4 / SCALE], dtype=np.float32)
    q = quantize(e)
    assert q == [1, 0]


def test_shift_unsigned_is_in_valid_range():
    quantized = [-BOUND, 0, BOUND]
    shifted = shift_unsigned(quantized)
    assert shifted == [0, BOUND, 2 * BOUND]
    assert all(0 <= s <= 2 * BOUND for s in shifted)


def test_shift_unsigned_round_trips():
    quantized = [-100, 0, 4096, -4096, 1234]
    shifted = shift_unsigned(quantized)
    unshifted = [s - BOUND for s in shifted]
    assert unshifted == quantized
