"""scripts/capture_probe.py — the frame-selection behaviour, without a camera.

The interesting claim is that the shutter keeps the *best* buffered frame
rather than the first one that scraped past the gate. A scripted fake camera
proves that deterministically, which a webcam never could.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from faceanchor.vision import detect, quality
from faceanchor.vision.decode import decode_jpeg_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "capture_frame.jpg"


@pytest.fixture(scope="module")
def capture_module():
    spec = importlib.util.spec_from_file_location("capture_probe", REPO_ROOT / "scripts" / "capture_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["capture_probe"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frames() -> list[np.ndarray]:
    """Three gate-passing camera frames of the same face, worst first.

    Worst first is the whole point: under the old first-past-the-post rule the
    blown-out frame is what got written.

    The ladder is exposure rather than blur, for two reasons. A blur heavy
    enough to matter fails the Laplacian screen outright and never reaches the
    buffer, which would make "best of N" trivially true. And exposure makes the
    same point the pixelation test does from the other direction: clipping the
    highlights *raises* Laplacian variance (205 -> 482 here, because clipped
    regions have hard edges) while the frame gets strictly worse. A ranker
    reading sharpness alone would climb this ladder backwards.
    """
    frame = decode_jpeg_bytes(FIXTURE.read_bytes())
    sequence = [
        cv2.convertScaleAbs(frame, alpha=2.2),
        cv2.convertScaleAbs(frame, alpha=1.6),
        frame,
    ]

    for index, candidate in enumerate(sequence):
        faces = detect.detect_faces(candidate)
        assert faces, f"frame {index} lost its face — the test needs all three to pass the gate"
        assert quality.check_quality(candidate, faces[0]).ok, f"frame {index} does not pass the gate"
    return sequence


class _FakeCamera:
    """Cycles a fixed sequence forever, like a camera pointed at a still scene."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._index = 0

    def isOpened(self) -> bool:  # noqa: N802 — matching cv2's API
        return True

    def set(self, *_args) -> bool:
        return True

    def read(self):
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return True, frame.copy()

    def release(self) -> None:
        return None


def _closest(saved: np.ndarray, frames: list[np.ndarray]) -> int:
    return int(np.argmin([np.mean(np.abs(saved.astype(float) - f.astype(float))) for f in frames]))


def test_capture_keeps_the_best_buffered_frame_not_the_first(capture_module, frames, tmp_path, monkeypatch):
    out = tmp_path / "probe.jpg"
    monkeypatch.setattr(capture_module.cv2, "VideoCapture", lambda *_a, **_k: _FakeCamera(frames))
    monkeypatch.setattr(capture_module, "HEADLESS_BURST_S", 0.05)
    monkeypatch.setattr(
        sys, "argv",
        ["capture_probe.py", "--out", str(out), "--no-preview", "--warmup", "0", "--timeout", "10"],
    )

    assert capture_module.main() == 0
    assert out.exists()

    saved = decode_jpeg_bytes(out.read_bytes())
    assert _closest(saved, frames) == 2, "kept a blurred frame when a sharp one was in the buffer"


def test_capture_ranks_a_sharp_frame_above_a_blurred_one(capture_module, frames):
    """The ranking itself, independent of the loop that uses it."""
    scores = []
    for frame in frames:
        face = detect.detect_faces(frame)[0]
        scores.append(capture_module._score(frame, face, quality.check_quality(frame, face)))
    assert scores[2] > scores[1] > scores[0]


def test_capture_penalises_a_blown_out_face(capture_module, frames):
    """Exposure is invisible to both the Laplacian and the quality model's crop,
    so the ranker measures it directly."""
    clean, blown = frames[2], frames[0]
    face = detect.detect_faces(clean)[0]
    assert capture_module._clipped_fraction(clean, face) < 0.10
    assert capture_module._clipped_fraction(blown, detect.detect_faces(blown)[0]) > 0.5

    # ...and the metric it replaced moves the wrong way on the same pair.
    sharpness = [quality.check_quality(f, detect.detect_faces(f)[0]).laplacian_variance for f in (clean, blown)]
    assert sharpness[1] > sharpness[0], "the blown-out frame should read as *sharper* to a Laplacian"


class _ForcedVerdict:
    """Stands in for the anti-spoof ensemble with a fixed finding."""

    def __init__(self, verdict):
        from faceanchor.vision import liveness

        self.result = liveness.LivenessResult(
            is_live=False, verdict=verdict, live_probability=0.04, reason=f"forced {verdict.value}"
        )

    def check(self, *_args):
        return self.result


def _run_capture(capture_module, frames, out, monkeypatch, extra_argv):
    monkeypatch.setattr(capture_module.cv2, "VideoCapture", lambda *_a, **_k: _FakeCamera(frames))
    monkeypatch.setattr(capture_module, "HEADLESS_BURST_S", 0.05)
    monkeypatch.setattr(
        sys, "argv",
        ["capture_probe.py", "--out", str(out), "--no-preview", "--warmup", "0", "--timeout", "2", *extra_argv],
    )
    return capture_module.main()


def test_replay_is_refused_by_default(capture_module, frames, tmp_path, monkeypatch):
    from faceanchor.vision import liveness

    out = tmp_path / "probe.jpg"
    monkeypatch.setattr(capture_module.liveness, "check", _ForcedVerdict(liveness.Verdict.REPLAY_ATTACK).check)
    assert _run_capture(capture_module, frames, out, monkeypatch, []) == 1
    assert not out.exists(), "a face on a screen was written as the probe"


def test_allow_replay_accepts_a_screen(capture_module, frames, tmp_path, monkeypatch):
    """The development escape hatch, and only for the finding it names."""
    from faceanchor.vision import liveness

    out = tmp_path / "probe.jpg"
    monkeypatch.setattr(capture_module.liveness, "check", _ForcedVerdict(liveness.Verdict.REPLAY_ATTACK).check)
    assert _run_capture(capture_module, frames, out, monkeypatch, ["--allow-replay"]) == 0
    assert out.exists()


def test_allow_replay_does_not_also_allow_a_printed_photo(capture_module, frames, tmp_path, monkeypatch):
    """Widening one hole must not widen the other: --allow-replay exists so you
    can iterate against a monitor, not so anti-spoofing becomes advisory."""
    from faceanchor.vision import liveness

    out = tmp_path / "probe.jpg"
    monkeypatch.setattr(capture_module.liveness, "check", _ForcedVerdict(liveness.Verdict.PRINT_ATTACK).check)
    assert _run_capture(capture_module, frames, out, monkeypatch, ["--allow-replay"]) == 1
    assert not out.exists()
