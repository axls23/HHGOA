"""Stage 1 · presentation-attack detection (Silent-Face MiniFASNet, Apache-2.0).

Without this, a printed photograph held up to the webcam walks the whole
pipeline: it detects, it passes the quality gate, it embeds, it matches the
person in the photo, and it anchors a bundle that says this pipeline observed
that face at that time. Every downstream guarantee still holds — the bundle is
tamper-evident, the score is proven, the chain is honest — and all of them are
guarding a lie told at the lens. Tamper-evidence downstream of a spoofed
capture is the wrong shape of assurance, so the check belongs here.

The Silent-Face ensemble is two small CNNs over two differently-scaled crops of
the same box (2.7x for MiniFASNetV2, 4.0x for MiniFASNetV1SE — the wider crop
is what sees a phone bezel or the edge of a sheet of paper). Each returns a
3-class softmax; class 1 is a live face. Their probabilities are averaged, the
way the reference implementation does.

What it is not: a guarantee. It is a single-frame RGB classifier trained
largely on print and replay attacks, so it is beaten by a good mask, by an
attack it has not seen, and by a synthetic frame injected below the camera API
(this reads whatever V4L2 hands over, and cannot tell a real sensor from a
virtual one). It raises the cost of the easy attack from "hold up your phone"
to something a demo audience cannot do in the room, and that is the honest
claim for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from faceanchor.vision.detect import Face
from faceanchor.vision.models import MINIFASNET_V1SE_PATH, MINIFASNET_V2_PATH, get_antispoof_sessions

# The class the models call a genuine face; 0 and 2 are the two attack families
# (print and replay) they were trained to separate from it.
PRINT_CLASS, LIVE_CLASS, REPLAY_CLASS = 0, 1, 2

# argmax alone would call 0.34/0.33/0.33 a live face. Requiring the live class
# to also clear a majority makes an undecided ensemble read as "not live",
# which is the right default for a gate that exists to refuse things.
MIN_LIVE_PROBABILITY = 0.5


class Verdict(str, Enum):
    LIVE = "live"
    PRINT_ATTACK = "print_attack"
    REPLAY_ATTACK = "replay_attack"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class LivenessResult:
    """What the ensemble saw. Not what the caller should do about it.

    The two are separated on purpose: a development capture may legitimately
    want to accept a face on a screen (it is the only way to iterate without
    sitting in front of the camera), and that is a *policy* decision belonging
    to whoever is capturing — not something this module should express by
    lying about what it detected. So `verdict` always reports the model's
    finding, and the caller decides which findings it will accept.
    """

    is_live: bool
    verdict: Verdict
    live_probability: float
    reason: str | None


def available() -> bool:
    return MINIFASNET_V2_PATH.exists() and MINIFASNET_V1SE_PATH.exists()


def _crop(image_bgr: np.ndarray, face: Face, scale: float) -> np.ndarray:
    """The reference crop: box centre, widened by `scale`, clamped to the frame.

    The scale is part of the weights — each model expects the face to fill a
    particular fraction of its 80x80 input — so it is not a tunable here.
    """
    src_h, src_w = image_bgr.shape[:2]
    x, y, box_w, box_h = (float(v) for v in face.bbox)
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x + box_w / 2, y + box_h / 2

    x1, y1 = max(0, int(cx - new_w / 2)), max(0, int(cy - new_h / 2))
    x2, y2 = min(src_w - 1, int(cx + new_w / 2)), min(src_h - 1, int(cy + new_h / 2))
    return cv2.resize(image_bgr[y1 : y2 + 1, x1 : x2 + 1], (80, 80))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def check(image_bgr: np.ndarray, face: Face) -> LivenessResult:
    """Is this frame a live face, or a photo/screen of one?

    Raw 0-255 BGR in NCHW, no mean subtraction — the reference `to_tensor` does
    nothing but transpose and cast, and matching it exactly is what makes the
    published thresholds mean anything.
    """
    probabilities = np.zeros(3, dtype=np.float64)
    for session, scale in get_antispoof_sessions():
        crop = _crop(image_bgr, face, scale)
        blob = crop.transpose(2, 0, 1)[None].astype(np.float32)
        logits = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
        probabilities += _softmax(np.asarray(logits, dtype=np.float64))
    probabilities /= len(get_antispoof_sessions())

    live_probability = float(probabilities[LIVE_CLASS])
    predicted = int(np.argmax(probabilities))
    if predicted == PRINT_CLASS:
        return LivenessResult(
            False, Verdict.PRINT_ATTACK, live_probability,
            f"looks like a print attack — a photograph held to the camera (live p={live_probability:.2f})",
        )
    if predicted == REPLAY_CLASS:
        return LivenessResult(
            False, Verdict.REPLAY_ATTACK, live_probability,
            f"looks like a replay attack — a face on a screen (live p={live_probability:.2f})",
        )
    if live_probability < MIN_LIVE_PROBABILITY:
        return LivenessResult(
            False, Verdict.UNDECIDED, live_probability, f"liveness undecided (live p={live_probability:.2f})"
        )
    return LivenessResult(True, Verdict.LIVE, live_probability, None)
