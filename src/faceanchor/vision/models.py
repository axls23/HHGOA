"""Shared model-cache paths and lazy singletons for the vision stage."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import onnx

MODEL_CACHE = Path(os.environ.get("FACEANCHOR_MODEL_CACHE", Path.home() / ".cache" / "faceanchor" / "models"))

YUNET_PATH = MODEL_CACHE / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_CACHE / "face_recognition_sface_2021dec.onnx"
SFACE_DYNBATCH_PATH = MODEL_CACHE / "face_recognition_sface_2021dec.dynbatch.onnx"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `models/fetch.sh` first.")
    return path


@lru_cache(maxsize=8)
def get_yunet_detector(input_w: int, input_h: int, score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 5000):
    return cv2.FaceDetectorYN.create(
        str(_require(YUNET_PATH)), "", (input_w, input_h), score_threshold, nms_threshold, top_k
    )


@lru_cache(maxsize=1)
def get_sface_recognizer():
    """CPU-path recognizer for single-image (probe) embedding — cv2's own DNN backend.

    PRD §5.4: a GPU round trip costs more than it saves on one image, so the
    probe embed stays on the same code path OpenCV Zoo's reference uses.
    """
    return cv2.FaceRecognizerSF.create(str(_require(SFACE_PATH)), "")


def ensure_sface_dynbatch_onnx() -> Path:
    """Derives a batch-dynamic SFace graph from the fixed-batch-1 export, cached on disk.

    The upstream ONNX Runtime export from OpenCV Zoo hardcodes batch=1. Relaxing
    dim 0 of the input/output to a symbolic axis is numerically identical
    (verified: <1e-6 max abs diff vs per-item inference) and is what lets the
    candidate-batch embed step actually use CUDA (PRD §5.4's one real GPU win).
    """
    if SFACE_DYNBATCH_PATH.exists():
        return SFACE_DYNBATCH_PATH

    model = onnx.load(str(_require(SFACE_PATH)))
    for vi in model.graph.input:
        if vi.name == "data":
            vi.type.tensor_type.shape.dim[0].Clear()
            vi.type.tensor_type.shape.dim[0].dim_param = "batch"
    for vi in model.graph.output:
        if vi.name == "fc1":
            vi.type.tensor_type.shape.dim[0].Clear()
            vi.type.tensor_type.shape.dim[0].dim_param = "batch"

    tmp_path = SFACE_DYNBATCH_PATH.with_suffix(".tmp.onnx")
    onnx.save(model, str(tmp_path))
    tmp_path.rename(SFACE_DYNBATCH_PATH)
    return SFACE_DYNBATCH_PATH
