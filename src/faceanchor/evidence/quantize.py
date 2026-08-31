"""int16 fixed-point quantization for the ZK circuit (PRD §7.1).

SFace embeddings are L2-normalized (post faceanchor.vision.embed), so every
component is in [-1, 1]. `q = round(x * 2^12)` maps that cleanly into
[-4096, 4096], matching the circuit's SignedRangeCheck bound. Measured
cosine drift from this quantization is <0.001 — well inside the 0.363
decision margin.
"""

from __future__ import annotations

import numpy as np

SCALE = 2**12
BOUND = 4096


def quantize(embedding: np.ndarray) -> list[int]:
    """Unit-norm float32[128] -> int16-range Python ints, clamped to [-BOUND, BOUND]."""
    scaled = np.round(embedding * SCALE).astype(np.int64)
    clamped = np.clip(scaled, -BOUND, BOUND)
    return clamped.tolist()


def shift_unsigned(quantized: list[int]) -> list[int]:
    """circom signals are unsigned field elements — shift into [0, 2*BOUND]
    for the circuit's SignedRangeCheck input."""
    return [q + BOUND for q in quantized]
