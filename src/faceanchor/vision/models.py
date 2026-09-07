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
# v2: also strips initializers out of graph.input (see ensure_sface_dynbatch_onnx).
# The suffix is the cache-busting mechanism — a stale v1 file on disk would
# otherwise be reused forever, since the derivation only runs when it's absent.
SFACE_DYNBATCH_PATH = MODEL_CACHE / "face_recognition_sface_2021dec.dynbatch.v2.onnx"

# Capture-stage models. Both are permissively licensed (PRD §G5): the
# MiniFASNet pair is Apache-2.0 (minivision-ai/Silent-Face-Anti-Spoofing),
# eDifFIQA is MIT. Neither is required by the base pipeline — code that uses
# them degrades to the heuristic gate and says so, rather than failing.
MINIFASNET_V2_PATH = MODEL_CACHE / "MiniFASNetV2.onnx"
MINIFASNET_V1SE_PATH = MODEL_CACHE / "MiniFASNetV1SE.onnx"
EDIFFIQA_PATH = MODEL_CACHE / "ediffiqa_t.onnx"

# Alternative detector backend. Unlike every other weight here this one is not
# permissively licensed — YOLOv8-derived, so GPL/AGPL — and a detector cannot be
# scoped to an opt-in path the way the GPL Groth16Verifier is. Selected with
# FACEANCHOR_DETECTOR; see vision/yolo_face.py.
YOLO_FACE_PATH = MODEL_CACHE / "yolov8n-face.onnx"


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


def _cpu_session(path: Path):
    """These two are small (1.7MB, 6.9MB) and run on one crop at a time, so a
    CUDA context would cost more than the inference. CPU on purpose."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(_require(path)), sess_options=options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def get_antispoof_sessions():
    """The Silent-Face ensemble: two models over two differently-scaled crops.

    Returns ((session, scale), ...). The scales are the ones the weights were
    trained with (2.7 for V2, 4.0 for V1SE) — they are part of the model, not a
    tunable, since each expects the face to occupy a particular fraction of its
    80x80 input.
    """
    return (
        (_cpu_session(MINIFASNET_V2_PATH), 2.7),
        (_cpu_session(MINIFASNET_V1SE_PATH), 4.0),
    )


@lru_cache(maxsize=1)
def get_yolo_face_session():
    """YOLOv8-face, 640x640 in, [1, 300, 21] out (box, score, class, 5 keypoints)."""
    return _cpu_session(YOLO_FACE_PATH)


@lru_cache(maxsize=1)
def get_fiqa_session():
    """eDifFIQA-T: aligned 112x112 in, one scalar quality score out."""
    return _cpu_session(EDIFFIQA_PATH)


def have_capture_models() -> tuple[bool, bool]:
    """(liveness, fiqa) — which capture-stage models are actually on disk."""
    return (
        MINIFASNET_V2_PATH.exists() and MINIFASNET_V1SE_PATH.exists(),
        EDIFFIQA_PATH.exists(),
    )


def ensure_sface_dynbatch_onnx() -> Path:
    """Derives a batch-dynamic SFace graph from the fixed-batch-1 export, cached on disk.

    The upstream ONNX Runtime export from OpenCV Zoo hardcodes batch=1. Relaxing
    dim 0 of the input/output to a symbolic axis is numerically identical
    (verified: <1e-6 max abs diff vs per-item inference) and is what lets the
    candidate-batch embed step actually use CUDA (PRD §5.4's one real GPU win).

    The same rewrite also drops initializers from `graph.input`. That export
    lists all ~300 weights as graph inputs (a pre-IR-4 convention), so ORT
    treats them as overridable, declines to const-fold them, and prints one
    warning per weight — 349 lines of stderr in front of anything the pipeline
    actually says. Dropping them is precisely what ORT's own
    `remove_initializer_from_input.py` does, and it re-enables the folding.
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

    initializers = {init.name for init in model.graph.initializer}
    kept_inputs = [vi for vi in model.graph.input if vi.name not in initializers]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)

    tmp_path = SFACE_DYNBATCH_PATH.with_suffix(".tmp.onnx")
    onnx.save(model, str(tmp_path))
    tmp_path.rename(SFACE_DYNBATCH_PATH)
    return SFACE_DYNBATCH_PATH
