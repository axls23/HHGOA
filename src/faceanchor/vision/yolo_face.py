"""Stage 1 · YOLOv8-face detector backend (an alternative to YuNet).

Same contract as `detect.py`'s YuNet path: bbox, five landmarks, a score, and
the 15-value row `cv2.FaceRecognizerSF.alignCrop` expects — so alignment,
pose and everything downstream are unchanged by the swap.

The keypoint order is the same convention YuNet uses (verified against it on
real images, not assumed): image-left eye, image-right eye, nose tip,
image-left mouth corner, image-right mouth corner.

Two things differ from YuNet and both matter:

- **The boxes are tighter, and not by a fixed amount.** Measured on the three
  face images in this repo, YOLO's box ran 0.79x to 1.03x of YuNet's on the
  same face (mean of the smaller side ~0.90). Every constant calibrated
  against a box — the 80px floor, the Laplacian crop, the 2.7x/4.0x liveness
  crop — is therefore measuring something slightly different. `detect.py`
  scales the size floor by a rough factor to compensate; that factor is
  provisional, from n=3 images of one subject, and wants re-deriving over a
  corpus before it is trusted to two decimal places.
- **No NMS step.** This export is the end-to-end variant: it emits a fixed 300
  candidate rows already de-duplicated, so `nms_threshold` has nothing to act
  on and is accepted only to keep one signature across both backends.

Licence, stated plainly because the rest of the stack does not need saying:
the weights are YOLOv8-derived and therefore **GPL-3.0/AGPL-3.0** territory,
unlike YuNet/SFace (Apache-2.0), MiniFASNet (Apache-2.0) and eDifFIQA (MIT).
A detector sits on the default path and cannot be scoped away the way the
GPL Groth16Verifier is. See README "Licences".
"""

from __future__ import annotations

import cv2
import numpy as np

from faceanchor.vision.models import get_yolo_face_session

INPUT_SIZE = 640
PAD_VALUE = 114  # Ultralytics' letterbox grey; the weights were trained against it


def _letterbox(image_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize into a 640x640 canvas. Returns (canvas, scale, pad_x, pad_y)."""
    h, w = image_bgr.shape[:2]
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    pad_y, pad_x = (INPUT_SIZE - new_h) // 2, (INPUT_SIZE - new_w) // 2

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = cv2.resize(
        image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, pad_x, pad_y


def detect_rows(image_bgr: np.ndarray, score_threshold: float) -> list[np.ndarray]:
    """Faces as YuNet-shaped 15-value rows: x, y, w, h, 5x(lx, ly), score.

    Producing YuNet's exact row layout is the point — it is what lets
    `alignCrop` and every landmark consumer stay untouched by the swap.
    """
    canvas, scale, pad_x, pad_y = _letterbox(image_bgr)
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    session = get_yolo_face_session()
    # [1, 300, 21] -> x1, y1, x2, y2, score, class, then 5 x (kx, ky, kconf)
    detections = session.run(None, {session.get_inputs()[0].name: blob})[0][0]

    height, width = image_bgr.shape[:2]

    def to_image(x: float, y: float) -> tuple[float, float]:
        return (x - pad_x) / scale, (y - pad_y) / scale

    rows: list[np.ndarray] = []
    for detection in detections:
        score = float(detection[4])
        if score < score_threshold:
            continue
        x1, y1 = to_image(float(detection[0]), float(detection[1]))
        x2, y2 = to_image(float(detection[2]), float(detection[3]))
        # Clamp to the frame: a box running off the edge would make the
        # liveness crop and the Laplacian crop read out of bounds.
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, float(width)), min(y2, float(height))
        if x2 <= x1 or y2 <= y1:
            continue

        row = np.zeros(15, dtype=np.float32)
        row[0:4] = (x1, y1, x2 - x1, y2 - y1)
        for index, (kx, ky, _kconf) in enumerate(detection[6:].reshape(5, 3)):
            row[4 + index * 2], row[5 + index * 2] = to_image(float(kx), float(ky))
        row[14] = score
        rows.append(row)

    # The export emits its 300 rows in confidence order already, but nothing in
    # the contract promises that, and callers take `max(..., key=score)`.
    rows.sort(key=lambda r: float(r[14]), reverse=True)
    return rows
