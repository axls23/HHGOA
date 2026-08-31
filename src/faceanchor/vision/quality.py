"""Stage 1 · quality gate — runs pre-search, saves the entire Stage-2 budget on bad probes.

Gates, per PRD §5.1:
- min face bbox >= 80 px (min of width, height)
- Laplacian variance >= 60 (blur)
- landmark asymmetry ratio <= 0.35 (extreme pose)
- single dominant face, else require an explicit --face-index
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceanchor.vision.detect import Face

MIN_BBOX_PX = 80
MIN_LAPLACIAN_VARIANCE = 60.0
MAX_LANDMARK_ASYMMETRY = 0.35

# Landmark row order from cv2.FaceDetectorYN: right eye, left eye, nose tip, right mouth corner, left mouth corner.
_RIGHT_EYE, _LEFT_EYE, _NOSE_TIP = 0, 1, 2


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: str | None
    bbox_min_side: float
    laplacian_variance: float
    landmark_asymmetry: float


def _laplacian_variance(image_bgr: np.ndarray, face: Face) -> float:
    x, y, w, h = face.bbox
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x + w), image_bgr.shape[1]), min(int(y + h), image_bgr.shape[0])
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _landmark_asymmetry(face: Face) -> float:
    nose_x = face.landmarks[_NOSE_TIP][0]
    right_eye_x = face.landmarks[_RIGHT_EYE][0]
    left_eye_x = face.landmarks[_LEFT_EYE][0]
    dist_right = abs(nose_x - right_eye_x)
    dist_left = abs(nose_x - left_eye_x)
    denom = dist_right + dist_left
    if denom == 0:
        return 1.0
    return abs(dist_right - dist_left) / denom


def check_single_face(faces: list[Face], face_index: int | None) -> tuple[Face, str | None]:
    """Resolves which detected face to run the pipeline on.

    Returns (face, error) — error is None on success, else a message explaining
    why the run should abort (ambiguous multi-face probe without --face-index,
    or no face detected at all).
    """
    if not faces:
        return None, "no face detected in probe image"
    if len(faces) == 1:
        return faces[0], None
    if face_index is None:
        return None, f"{len(faces)} faces detected; pass --face-index to disambiguate"
    if not (0 <= face_index < len(faces)):
        return None, f"--face-index {face_index} out of range for {len(faces)} detected faces"
    return faces[face_index], None


def check_quality(image_bgr: np.ndarray, face: Face) -> QualityResult:
    bbox_min_side = float(min(face.bbox[2], face.bbox[3]))
    lap_var = _laplacian_variance(image_bgr, face)
    asymmetry = _landmark_asymmetry(face)

    reason = None
    if bbox_min_side < MIN_BBOX_PX:
        reason = f"face bbox too small ({bbox_min_side:.0f}px < {MIN_BBOX_PX}px)"
    elif lap_var < MIN_LAPLACIAN_VARIANCE:
        reason = f"image too blurry (Laplacian variance {lap_var:.1f} < {MIN_LAPLACIAN_VARIANCE})"
    elif asymmetry > MAX_LANDMARK_ASYMMETRY:
        reason = f"extreme pose (landmark asymmetry {asymmetry:.2f} > {MAX_LANDMARK_ASYMMETRY})"

    return QualityResult(
        ok=reason is None,
        reason=reason,
        bbox_min_side=bbox_min_side,
        laplacian_variance=lap_var,
        landmark_asymmetry=asymmetry,
    )
