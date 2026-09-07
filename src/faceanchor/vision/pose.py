"""Stage 1 · head pose from the 5 landmarks YuNet already gives us.

The gate this replaces compared the nose's horizontal distance to each eye and
called the ratio "pose". That is a yaw proxy and nothing else: it cannot see
**pitch** (a chin lifted to the ceiling keeps a perfectly symmetric face) and
it cannot see **roll** (a head tilted 40 degrees is symmetric about its own
axis). Both wreck a 5-point similarity alignment, and neither was measurable.

Solving PnP against a canonical 3D face recovers all three angles in degrees
from the same five points, with no extra model and no extra dependency —
OpenCV is already here. The camera is approximated the usual way (focal length
= image width, principal point = image centre, no distortion); an uncalibrated
webcam gives angles good to a few degrees, which is far inside the tolerance a
gate at 30 degrees needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceanchor.vision.detect import Face

# Canonical face, millimetres, nose tip at the origin. OpenCV image convention:
# +x to the image right, +y down, +z away from the camera. Row order matches
# YuNet's landmarks (subject's right eye first — it lands on the image *left*).
#
# The vertical offsets are calibrated, not looked up: a textbook anthropometric
# model left a systematic bias, because what matters here is not real face
# geometry but the geometry of the five points *YuNet reports*, which are not
# quite eye centres and mouth corners. Measured over the 195 faces in the
# reference corpus, this model puts the population medians at yaw -3.7,
# pitch -0.9, roll +0.4 degrees — i.e. a crowd of ordinary portraits reads as
# facing the camera, which is the only way an absolute threshold means
# anything. Re-run tests/test_vision.py::test_pose_is_centred_on_a_real_population
# after touching these numbers.
_MODEL_POINTS = np.array(
    [
        (-32.0, -42.0, -40.0),  # right eye centre
        (+32.0, -42.0, -40.0),  # left eye centre
        (0.0, 0.0, 0.0),        # nose tip
        (-26.0, +34.0, -29.0),  # right mouth corner
        (+26.0, +34.0, -29.0),  # left mouth corner
    ],
    dtype=np.float64,
)

# Yaw and pitch are what actually break recognition: a similarity transform on
# five points corrects roll and scale exactly, but cannot un-rotate a profile.
# So roll is gated loosely — only enough to catch a tilt that truncates the
# crop — and yaw/pitch at the 90th percentile of a real population (|yaw| 35,
# |pitch| 33, |roll| 16 over the reference corpus).
#
# Together these reject 14% of that corpus. The landmark-asymmetry gate they
# replace rejected 31% of the same faces — so this is both a looser gate and a
# better one: it lets through the head-tilted portrait the ratio failed for
# no reason, and stops the chin-to-the-ceiling frame the ratio could not see
# at all.
MAX_YAW_DEG = 35.0
MAX_PITCH_DEG = 35.0
MAX_ROLL_DEG = 45.0


# Known limitation: six degrees of freedom solved from five noisy points
# couples the axes. Rotating an image by 20 degrees moves roll by ~15 (right)
# but also leaks a few degrees into yaw, because the warp resamples the
# landmarks the solution is built from. It is accurate enough for a gate at 35
# degrees and should not be read as a measurement.
# See tests/test_vision.py::test_pose_tracks_image_rotation.


@dataclass(frozen=True)
class HeadPose:
    yaw: float    # degrees, + turning to the image right
    pitch: float  # degrees, + chin up
    roll: float   # degrees, + head tilted clockwise in the image

    @property
    def worst(self) -> float:
        """The angle that will hurt alignment most — what a gate should read."""
        return max(abs(self.yaw), abs(self.pitch), abs(self.roll))


def _normalize(angle: float) -> float:
    """decomposeProjectionMatrix reports the same rotation as ±180 away.

    A frontal face routinely comes back as pitch ≈ 178 rather than ≈ -2, so
    every angle is folded into (-90, 90]. A real head is never outside that
    range in a frame a detector found a face in.
    """
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return angle


def estimate(image_bgr: np.ndarray, face: Face) -> HeadPose | None:
    """Yaw/pitch/roll in degrees, or None if PnP could not solve this face."""
    h, w = image_bgr.shape[:2]
    focal = float(w)
    camera_matrix = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    image_points = np.ascontiguousarray(face.landmarks, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix,
        np.zeros((4, 1)),
        # EPNP over ITERATIVE: with only five points and no initial guess, the
        # iterative solver converges to a mirrored pose often enough to matter.
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(np.hstack([rotation, tvec]))
    pitch, yaw, roll = (_normalize(float(a)) for a in euler.flatten())
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)
