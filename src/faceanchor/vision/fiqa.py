"""Stage 1 · learned face image quality (eDifFIQA-T, MIT).

Laplacian variance answers "does this image contain high-frequency detail",
which is not the question. It rises with a busy background, falls on a
perfectly sharp but smooth face, and moves with resolution and JPEG quality —
so a threshold on it is a threshold on the photograph, not on the face.

A face image quality assessment model answers the question that matters:
*how well will a recognizer do on this face*, trained against that outcome
directly. eDifFIQA-T is the tiny variant (6.9MB, ~2ms on CPU) of a distilled
diffusion-based estimator; it takes the same aligned 112x112 crop SFace does,
and returns one scalar where higher is better.

It does not replace the cheap checks — bbox size and Laplacian still run
first, because they cost microseconds and catch the obvious. It replaces the
*decision*.
"""

from __future__ import annotations

import cv2
import numpy as np

from faceanchor.vision.models import EDIFFIQA_PATH, get_fiqa_session

# Measured, not guessed. Over the 195 faces in the reference corpus the scores
# run min -0.02, p10 0.20, median 0.56, p90 0.73, max 0.80. Deliberately
# degrading the ten best faces moves them from 0.79 to: 0.52 (Gaussian k=11),
# 0.61 (JPEG q=10), 0.27 (Gaussian k=21), 0.21 (8x downscale-upscale).
#
# 0.35 sits above the heavy degradations and below mild blur, and rejects 23%
# of casual timeline portraits — which is the right severity for a *probe*, a
# cooperative capture the operator can simply retake, and not something ever
# applied to candidates. Both real probes in this repo score 0.65-0.68, so the
# gate has ~0.3 of headroom on the images it is actually meant to admit.
# See tests/test_vision.py::test_fiqa_ranks_degraded_faces_lower.
MIN_FIQA_SCORE = 0.35


def available() -> bool:
    return EDIFFIQA_PATH.exists()


def score(aligned_crop_bgr: np.ndarray) -> float:
    """Quality of an aligned 112x112 BGR crop (align.align_and_crop's output).

    Preprocessing is pinned to the reference implementation: RGB, scaled to
    [-1, 1] as (x - 127.5) / 127.5, NCHW.
    """
    session = get_fiqa_session()
    blob = cv2.dnn.blobFromImage(
        aligned_crop_bgr, scalefactor=1.0 / 127.5, size=(112, 112), mean=(127.5, 127.5, 127.5), swapRB=True
    )
    output = session.run(None, {session.get_inputs()[0].name: blob})[0]
    return float(np.squeeze(output))
