"""Stage 1 · 5-pt similarity alignment + crop to the 112x112 SFace input size."""

from __future__ import annotations

import numpy as np

from faceanchor.vision.detect import Face
from faceanchor.vision.models import get_sface_recognizer


def align_and_crop(image_bgr: np.ndarray, face: Face) -> np.ndarray:
    """Returns a 112x112x3 uint8 BGR crop, similarity-aligned on the 5 landmarks."""
    recognizer = get_sface_recognizer()
    return recognizer.alignCrop(image_bgr, face.raw_row)
