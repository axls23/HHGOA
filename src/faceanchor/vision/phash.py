"""Perceptual hash — catches exact re-uploads without invoking the recognizer.

PRD §5.2 scoring cascade step 1: `hamming(phash_probe, phash_cand) <= 8` is an
immediate high-confidence match, skipping the (much more expensive) embed step.
"""

from __future__ import annotations

import cv2
import numpy as np

_hasher = cv2.img_hash.PHash_create()


def phash(image_bgr: np.ndarray) -> bytes:
    """Returns the 64-bit perceptual hash as 8 raw bytes."""
    return _hasher.compute(image_bgr).tobytes()


def hamming_distance(hash_a: bytes, hash_b: bytes) -> int:
    a = np.frombuffer(hash_a, dtype=np.uint8)
    b = np.frombuffer(hash_b, dtype=np.uint8)
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())
