"""Stage 1 · quality gate — runs pre-search, saves the entire Stage-2 budget on bad probes.

Gates, cheapest first, so the expensive question is only asked of images that
survive the free ones:
- single dominant face, else require an explicit --face-index
- min face bbox >= 80 px (min of width, height)          [PRD §5.1]
- Laplacian variance >= 60 — a coarse screen for an obviously broken frame,
  not the quality decision it used to be (see `fiqa.py` for why it was a poor
  one: it measures high-frequency detail in the photograph, not usability of
  the face)
- head pose: |yaw| <= 35, |pitch| <= 35, |roll| <= 45 degrees, solved from the
  same five landmarks (`pose.py`). This replaces the old landmark-asymmetry
  ratio, which was blind to pitch and roll entirely; the ratio is still
  computed and reported, and still gates if PnP cannot solve the face.
- learned quality >= 0.35 (`fiqa.py`), when the model is present. Optional by
  design: absent weights degrade to the heuristics above and the caller is
  told so, rather than the gate silently becoming weaker than it claims.

Liveness is deliberately *not* here — see `liveness.py`. "Is this image good
enough" and "is this a real person in front of a camera" are different
questions, and the second one is only answerable of a frame that came from a
camera this process opened.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceanchor.vision import align, detect, fiqa, pose
from faceanchor.vision.detect import Face
from faceanchor.vision.pose import HeadPose

MIN_BBOX_PX = 80
MIN_LAPLACIAN_VARIANCE = 60.0
# Fallback only, for the rare face PnP cannot solve. See pose.py.
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
    # None when PnP could not solve the face (pose) or the model is not
    # installed (fiqa) — distinct from a value that failed, and reported as
    # such rather than being quietly treated as a pass.
    head_pose: HeadPose | None = None
    fiqa: float | None = None


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
    head_pose = pose.estimate(image_bgr, face)

    # The floor is a statement about the face, not about the rectangle the
    # detector happened to draw around it, so it follows the backend's scale.
    floor = MIN_BBOX_PX * detect.bbox_scale()

    reason = None
    if bbox_min_side < floor:
        reason = f"face bbox too small ({bbox_min_side:.0f}px < {floor:.0f}px)"
    elif lap_var < MIN_LAPLACIAN_VARIANCE:
        reason = f"image too blurry (Laplacian variance {lap_var:.1f} < {MIN_LAPLACIAN_VARIANCE})"
    elif head_pose is None:
        if asymmetry > MAX_LANDMARK_ASYMMETRY:
            reason = f"extreme pose (landmark asymmetry {asymmetry:.2f} > {MAX_LANDMARK_ASYMMETRY})"
    elif abs(head_pose.yaw) > pose.MAX_YAW_DEG:
        reason = f"turned too far from the camera (yaw {head_pose.yaw:+.0f}deg, limit {pose.MAX_YAW_DEG:.0f})"
    elif abs(head_pose.pitch) > pose.MAX_PITCH_DEG:
        reason = f"chin too high or low (pitch {head_pose.pitch:+.0f}deg, limit {pose.MAX_PITCH_DEG:.0f})"
    elif abs(head_pose.roll) > pose.MAX_ROLL_DEG:
        reason = f"head tilted too far (roll {head_pose.roll:+.0f}deg, limit {pose.MAX_ROLL_DEG:.0f})"

    # Last, because it is the only one that costs a model inference — and only
    # of a face that already has the size, sharpness and pose to be worth
    # asking about.
    quality_score = None
    if reason is None and fiqa.available():
        quality_score = fiqa.score(align.align_and_crop(image_bgr, face))
        if quality_score < fiqa.MIN_FIQA_SCORE:
            reason = f"face quality too low for recognition ({quality_score:.2f} < {fiqa.MIN_FIQA_SCORE})"

    return QualityResult(
        ok=reason is None,
        reason=reason,
        bbox_min_side=bbox_min_side,
        laplacian_variance=lap_var,
        landmark_asymmetry=asymmetry,
        head_pose=head_pose,
        fiqa=quality_score,
    )
