"""Stage 1 · face detection — bbox + 5 landmarks in one pass, from either backend.

Two detectors, one contract. Both return `Face` with the same 15-value
`raw_row` layout, so `align.align_and_crop`, `pose.estimate`, `quality` and
`liveness` are all indifferent to which one ran:

- **yunet** — OpenCV Zoo's YuNet (Apache-2.0, 232KB). The default, and what
  every calibrated constant in this package was measured against.
- **yolo** — YOLOv8-face (`yolo_face.py`). More robust on small, rotated and
  partially occluded faces; ~9MB and ~10x the inference cost of YuNet; and
  **GPL/AGPL**, which is a licence decision about the whole repo, not a
  detail — see `yolo_face.py`.

Select with `FACEANCHOR_DETECTOR=yunet|yolo`.

The one thing that does not transfer between them is the *scale of a box*.
YOLO's boxes are tighter, so a face YuNet calls 80px YOLO may call ~72px.
Rather than leave `MIN_BBOX_PX` silently meaning two different things,
`bbox_scale()` reports the backend's factor and `quality.py` scales the floor
by it, keeping the gate a statement about the *face* rather than about
whichever model drew the rectangle.

Also true, and the reason a run must not mix backends: the two align slightly
differently, so the same face embeds to a cosine of 0.93-0.98 across them
rather than 1.0. That is far above the 0.363 accept threshold, so a person
still matches themselves — but a marginal candidate can move by a few
hundredths, which is why the probe and the candidates always go through this
one function and the bundle records which detector ran.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from faceanchor.vision.models import get_yunet_detector

# cv2.FaceDetectorYN.detect() row layout: x,y,w,h, then 5 landmark (x,y) pairs
# (right eye, left eye, nose tip, right mouth corner, left mouth corner), then score.
_BBOX = slice(0, 4)
_LANDMARKS = slice(4, 14)
_SCORE = 14

BACKENDS = ("yunet", "yolo")
DEFAULT_BACKEND = "yolo"

# YuNet = 1.0 by definition. The YOLO figure is provisional: measured on the
# three face images this repo ships, where the ratio of the smaller box side
# ran 0.79-1.03 (mean 0.90). One subject is not a population — re-derive it
# over a corpus before treating it as more than a rough correction.
# See tests/test_vision.py::test_backends_agree_on_the_same_face.
_BBOX_SCALE = {"yunet": 1.0, "yolo": 0.90}


def backend() -> str:
    """Which detector this process is using."""
    chosen = os.environ.get("FACEANCHOR_DETECTOR", DEFAULT_BACKEND).strip().lower()
    if chosen not in BACKENDS:
        raise ValueError(f"unknown FACEANCHOR_DETECTOR={chosen!r} (expected one of {', '.join(BACKENDS)})")
    return chosen


def bbox_scale() -> float:
    """How large this backend's boxes are relative to YuNet's, for size gates."""
    return _BBOX_SCALE[backend()]


@dataclass(frozen=True)
class Face:
    bbox: np.ndarray       # [x, y, w, h]
    landmarks: np.ndarray  # 5x2
    score: float
    raw_row: np.ndarray    # full 15-value row, as cv2's alignCrop expects


def _face_from_row(row: np.ndarray) -> Face:
    return Face(
        bbox=row[_BBOX],
        landmarks=row[_LANDMARKS].reshape(5, 2),
        score=float(row[_SCORE]),
        raw_row=row,
    )


def detect_faces(image_bgr: np.ndarray, score_threshold: float = 0.6, nms_threshold: float = 0.3) -> list[Face]:
    if backend() == "yolo":
        from faceanchor.vision import yolo_face

        # nms_threshold is accepted and ignored: this export is NMS-free.
        return [_face_from_row(row) for row in yolo_face.detect_rows(image_bgr, score_threshold)]

    h, w = image_bgr.shape[:2]
    detector = get_yunet_detector(w, h, score_threshold, nms_threshold)
    _, faces = detector.detect(image_bgr)
    if faces is None:
        return []
    return [_face_from_row(row) for row in faces]
