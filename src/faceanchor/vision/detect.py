"""Stage 1 · face detection (YuNet) — bbox + 5-pt landmarks in one pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faceanchor.vision.models import get_yunet_detector

# cv2.FaceDetectorYN.detect() row layout: x,y,w,h, then 5 landmark (x,y) pairs
# (right eye, left eye, nose tip, right mouth corner, left mouth corner), then score.
_BBOX = slice(0, 4)
_LANDMARKS = slice(4, 14)
_SCORE = 14


@dataclass(frozen=True)
class Face:
    bbox: np.ndarray       # [x, y, w, h]
    landmarks: np.ndarray  # 5x2
    score: float
    raw_row: np.ndarray    # full 15-value row, as cv2's alignCrop expects


def detect_faces(image_bgr: np.ndarray, score_threshold: float = 0.6, nms_threshold: float = 0.3) -> list[Face]:
    h, w = image_bgr.shape[:2]
    detector = get_yunet_detector(w, h, score_threshold, nms_threshold)
    _, faces = detector.detect(image_bgr)
    if faces is None:
        return []
    out = []
    for row in faces:
        out.append(
            Face(
                bbox=row[_BBOX],
                landmarks=row[_LANDMARKS].reshape(5, 2),
                score=float(row[_SCORE]),
                raw_row=row,
            )
        )
    return out
