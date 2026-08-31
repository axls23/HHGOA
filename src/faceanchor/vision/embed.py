"""Stage 1 · SFace 128-d embedding.

Two paths, per PRD §5.4's own benchmark finding — GPU only pays for itself on
wide batches, not the single-probe hot path:

- `embed_one`: CPU, via cv2.FaceRecognizerSF (OpenCV Zoo's own reference path).
- `embed_batch`: CUDA EP via raw ONNX Runtime on a batch-dynamic export of the
  same weights. Preprocessing is empirically pinned to match cv2's C++
  implementation exactly (verified cosine=1.0 / max abs diff ~0 against
  cv2.FaceRecognizerSF.feature() on a real aligned crop):
  RGB (not BGR) order, raw [0, 255] float32, HWC -> CHW, no mean/scale.

Both paths L2-normalize the output — cv2's raw `.feature()` is NOT unit-norm
(observed ||e|| ~= 12.1, not 1.0), but downstream cosine scoring (PRD §5.2)
and the ZK quantization (PRD §7.1, values assumed in [-1, 1]) both require it.
"""

from __future__ import annotations

import threading

import numpy as np
import onnxruntime as ort

from faceanchor.vision import _cuda_libs  # noqa: F401 — preloads cuDNN/cuBLAS before ORT touches CUDA EP
from faceanchor.vision.models import ensure_sface_dynbatch_onnx

EMBED_DIM = 128

_batch_session: ort.InferenceSession | None = None
_batch_session_lock = threading.Lock()
_cpu_session: ort.InferenceSession | None = None
_cpu_session_lock = threading.Lock()


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return x / norm


def _crop_to_chw(crop_bgr: np.ndarray) -> np.ndarray:
    return crop_bgr[..., ::-1].astype(np.float32).transpose(2, 0, 1)  # BGR->RGB, HWC->CHW, raw [0,255]


def _tuned_session_options() -> ort.SessionOptions:
    # PRD §5.4: pin intra-op threads to P-cores only (see taskset -c 0-15 at
    # process launch) — letting the scheduler place work on E-cores makes one
    # straggler gate the whole op (2-3x worse p99, unstable benchmarks).
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


def _get_cpu_session() -> ort.InferenceSession:
    """CPU-EP session for the single-probe hot path — raw ONNX Runtime, not
    cv2.FaceRecognizerSF (measured ~38ms p50 for cv2's DNN backend on this
    machine vs ~10-25ms for a thread-tuned raw ORT session — see M1 bench notes).
    """
    global _cpu_session
    if _cpu_session is None:
        with _cpu_session_lock:
            if _cpu_session is None:
                sess = ort.InferenceSession(
                    str(ensure_sface_dynbatch_onnx()), sess_options=_tuned_session_options(), providers=["CPUExecutionProvider"]
                )
                dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
                sess.run(None, {"data": dummy})
                _cpu_session = sess
    return _cpu_session


def embed_one(aligned_crop_bgr: np.ndarray) -> np.ndarray:
    """CPU embed for a single 112x112 BGR crop. Returns a unit-norm 128-d vector."""
    session = _get_cpu_session()
    x = _crop_to_chw(aligned_crop_bgr)[None, ...]
    out = session.run(None, {"data": x})[0][0]
    return _l2_normalize(out)


def _get_batch_session() -> ort.InferenceSession:
    global _batch_session
    if _batch_session is None:
        with _batch_session_lock:
            if _batch_session is None:
                model_path = str(ensure_sface_dynbatch_onnx())
                sess = ort.InferenceSession(
                    model_path, sess_options=_tuned_session_options(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                # pre-warm: first call pays cuDNN algo-search / CUDA context init cost
                dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
                sess.run(None, {"data": dummy})
                _batch_session = sess
    return _batch_session


def embed_batch(aligned_crops_bgr: list[np.ndarray]) -> np.ndarray:
    """Batch embed via CUDA EP. Returns (N, 128) unit-norm vectors.

    Falls back to the CPU path transparently (same session, ORT picks
    CPUExecutionProvider) if CUDA EP failed to initialize.
    """
    if not aligned_crops_bgr:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    session = _get_batch_session()
    batch = np.stack([_crop_to_chw(crop) for crop in aligned_crops_bgr])
    out = session.run(None, {"data": batch})[0]
    return _l2_normalize(out)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both `a` and `b` are assumed unit-norm (as returned by embed_one/embed_batch)."""
    return float(np.dot(a, b))
