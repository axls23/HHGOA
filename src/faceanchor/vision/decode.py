"""Fast JPEG decode — PyTurboJPEG when the native libturbojpeg is available,
transparent fallback to cv2.imdecode otherwise (PRD §5.1 lists TurboJPEG as a
~2x speedup on the candidate fetch path, not a hard dependency).
"""

from __future__ import annotations

import numpy as np

try:
    from turbojpeg import TurboJPEG

    _turbo = TurboJPEG()
except Exception:
    _turbo = None

import cv2


def decode_jpeg_bytes(data: bytes) -> np.ndarray:
    """Returns an HxWx3 uint8 BGR array."""
    if _turbo is not None:
        try:
            return _turbo.decode(data)
        except Exception:
            pass
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image bytes")
    return img
