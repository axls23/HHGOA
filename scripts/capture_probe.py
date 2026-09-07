"""Capture the probe image straight from a webcam, with Stage 1's own gate
running live on the preview (PRD §5.1).

The gate is the same code the pipeline runs — `detect_faces` +
`check_single_face` + `check_quality` — so the frame you keep is a frame the
pipeline has already agreed to accept. That's the point of capturing here
rather than hunting for a photo file: the demo can't stall on a probe Stage 1
was always going to reject.

Two things happen here that the pipeline's own gate cannot do, because they
are only answerable of a live camera:

- **Best of N, not first-past-the-post.** Every gate-passing frame goes into a
  short rolling buffer; the shutter keeps the *best* frame in it, not the one
  that happened to be on screen when you pressed the key. Frames are free and
  a probe is used forever, so picking the best of a dozen costs nothing and
  removes the worst source of a mediocre probe: human reaction time.
- **Liveness.** A printed photo held to the lens passes every quality check
  there is, and anchors an evidence bundle that says this pipeline saw that
  face. `vision/liveness.py` is what makes that harder; it runs on the frame
  that is about to be written, and refuses it.

  python scripts/capture_probe.py --out evidence/probe.jpg
  python scripts/capture_probe.py --out probe.jpg --no-preview   # headless

Exit codes: 0 captured, 2 aborted by the operator, 1 anything else.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from faceanchor.vision import detect, fiqa, liveness, quality
from faceanchor.vision.detect import Face
from faceanchor.vision.quality import QualityResult

JPEG_QUALITY = 95
GREEN, RED, AMBER, WHITE = (110, 220, 120), (80, 80, 235), (60, 190, 240), (245, 245, 245)

# ~0.4s at 30fps. Long enough to span a blink and a breath, short enough that
# the frame you keep is still the moment you meant to capture.
BURST_FRAMES = 12
# Headless has no shutter, so it collects for this long after the first
# gate-passing frame and then keeps the best of what it saw.
HEADLESS_BURST_S = 1.0
# Liveness on the preview is for feedback only; the check that decides runs on
# the single frame about to be saved. Every 5th frame keeps the preview at
# full rate on a laptop CPU.
LIVENESS_PREVIEW_EVERY = 5


@dataclass
class Shot:
    """A gate-passing frame, kept as a candidate for the shutter."""

    frame: np.ndarray
    face: Face
    quality: QualityResult
    score: float


def _clipped_fraction(image_bgr: np.ndarray, face: Face) -> float:
    """How much of the face is blown out or crushed to black.

    Neither the quality model nor the Laplacian sees exposure the way it
    matters here: a backlit face can be perfectly sharp and perfectly posed
    and still carry no recoverable detail in the shadows.
    """
    x, y, w, h = (int(v) for v in face.bbox)
    crop = image_bgr[max(y, 0) : y + h, max(x, 0) : x + w]
    if crop.size == 0:
        return 1.0
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean((grey >= 250) | (grey <= 5)))


def _score(image_bgr: np.ndarray, face: Face, q: QualityResult) -> float:
    """Rank one gate-passing frame against another.

    This decides nothing — every frame it ranks has already passed the gate —
    so it only has to order them. Learned quality leads where it is available
    (it is the only term trained against what actually matters) and Laplacian
    stands in when the model is not installed; exposure is a penalty, because
    a face clipped to white carries no detail either metric can see.

    **Pose is deliberately not a term here**, though it is very much a gate.
    Ranking on it inverts the result: blurring an image moves the landmarks the
    pose is solved from, and it moves them toward the mean, so a degraded frame
    reads as *more* frontal than a sharp one. Measured on the test fixture, a
    3x3 Gaussian takes the estimate from 32 degrees to 8 — which would make the
    blurriest frame in every burst look like the best-posed one. Within a burst
    the subject's real pose barely changes anyway; what changes is the noise.
    See tests/test_capture.py::test_capture_ranks_a_sharp_frame_above_a_blurred_one.
    """
    base = q.fiqa if q.fiqa is not None else min(q.laplacian_variance / 400.0, 1.0)
    exposure_penalty = 1.0 - _clipped_fraction(image_bgr, face)
    return float(base * exposure_penalty)


def _gate(frame: np.ndarray, face_index: int | None):
    """(face, quality_result, message) for one frame — the pipeline's own gate."""
    faces = detect.detect_faces(frame)
    face, err = quality.check_single_face(faces, face_index)
    if err:
        return None, None, err
    q = quality.check_quality(frame, face)
    return face, q, (q.reason or "ready")


def _stats_line(q: QualityResult | None) -> str:
    if q is None:
        return "no usable face in frame"
    parts = [f"face {q.bbox_min_side:.0f}px (>={quality.MIN_BBOX_PX})"]
    if q.head_pose is not None:
        parts.append(f"yaw {q.head_pose.yaw:+.0f} pitch {q.head_pose.pitch:+.0f} roll {q.head_pose.roll:+.0f}")
    else:
        parts.append(f"sharpness {q.laplacian_variance:.0f}")
    if q.fiqa is not None:
        parts.append(f"quality {q.fiqa:.2f} (>={fiqa.MIN_FIQA_SCORE})")
    return "  ".join(parts)


def _overlay(frame, face, q, message, live, buffered: int, allow_replay: bool = False) -> np.ndarray:
    """Selfie-mirrored preview. Never touches the frame that gets saved."""
    view = cv2.flip(frame, 1)
    h, w = view.shape[:2]
    passing = q is not None and q.ok

    if face is not None:
        x, y, bw, bh = (int(v) for v in face.bbox)
        x = w - x - bw  # mirror the box to match the mirrored view
        cv2.rectangle(view, (x, y), (x + bw, y + bh), GREEN if passing else AMBER, 2)

    cv2.rectangle(view, (0, h - 84), (w, h), (28, 28, 28), -1)
    if live is not None and not live.is_live and not (allow_replay and live.verdict is liveness.Verdict.REPLAY_ATTACK):
        banner, colour = f"NOT READY — {live.reason}", RED
    elif passing:
        banner, colour = f"READY — SPACE to capture (best of {buffered})", GREEN
    else:
        banner, colour = f"NOT READY — {message}", RED
    cv2.putText(view, banner, (14, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2)
    cv2.putText(view, _stats_line(q), (14, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1)

    if live is not None:
        live_text = f"liveness {live.live_probability:.2f} {live.verdict.value}"
        if not live.is_live:
            live_text += "  (allowed: --allow-replay)" if allow_replay and live.verdict is liveness.Verdict.REPLAY_ATTACK else "  SPOOF"
    else:
        live_text = "liveness not installed — run models/fetch.sh"
    cv2.putText(view, live_text, (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, GREEN if (live and live.is_live) else AMBER, 1)

    cv2.putText(view, "ESC quit   F force-capture anyway", (w - 340, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1)
    return view


def _save(path: Path, shot: Shot, face_index: int | None) -> bool:
    """Write the JPEG, then re-gate the bytes actually on disk.

    JPEG quantisation moves the numbers a little, and the pipeline reads the
    file, not this in-memory array — so the gate that counts is the one applied
    after the round trip.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), shot.frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        print(f"ERROR: could not write {path}", file=sys.stderr)
        return False
    _, q, message = _gate(cv2.imread(str(path)), face_index)
    if q is None or not q.ok:
        print(f"  saved frame no longer passes the gate after JPEG encode ({message}) — keep going", file=sys.stderr)
        return False
    print(f"captured {path}  {_stats_line(q)}", file=sys.stderr)
    return True


def _keep_best(path: Path, buffer, face_index: int | None, check_liveness: bool, allow_replay: bool = False) -> bool:
    """Pick the best buffered frame, prove it is a live face, write it."""
    if not buffer:
        return False
    best = max(buffer, key=lambda shot: shot.score)
    print(f"  best of {len(buffer)} buffered frames (rank {best.score:.3f})", file=sys.stderr)

    if check_liveness:
        result = liveness.check(best.frame, best.face)
        if not result.is_live:
            if allow_replay and result.verdict is liveness.Verdict.REPLAY_ATTACK:
                # Development only. Iterating on the capture loop means pointing
                # the webcam at a photo on a monitor a hundred times, and a
                # working replay detector refuses every one of them. Nothing is
                # relaxed silently: it is opt-in per run, it names what it let
                # through, and it does not touch print-attack or undecided.
                print(f"  ALLOWED ANYWAY (--allow-replay): {result.reason}", file=sys.stderr)
            else:
                print(f"  REFUSED: {result.reason}", file=sys.stderr)
                print("  a photo or screen of a face is not a capture of a person. Use the real thing,", file=sys.stderr)
                print("  --allow-replay to develop against a screen, or --no-liveness to skip the check.", file=sys.stderr)
                return False
        else:
            print(f"  liveness ok (live p={result.live_probability:.2f})", file=sys.stderr)
    return _save(path, best, face_index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="where to write the captured JPEG")
    ap.add_argument("--camera", type=int, default=0, help="V4L2 device index [0]")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-preview", action="store_true", help="headless: keep the best gate-passing frame of a short burst")
    ap.add_argument("--timeout", type=float, default=60.0, help="headless: give up after N seconds [60]")
    ap.add_argument("--warmup", type=int, default=15, help="frames to discard while auto-exposure settles [15]")
    ap.add_argument("--face-index", type=int, default=None, help="disambiguate when more than one face is in frame")
    ap.add_argument("--burst", type=int, default=BURST_FRAMES, help=f"frames to choose the best from [{BURST_FRAMES}]")
    ap.add_argument("--no-liveness", action="store_true", help="do not refuse a frame the anti-spoof model calls a print/replay attack")
    ap.add_argument(
        "--allow-replay",
        action="store_true",
        help="development: accept a face on a screen (replay attack) while still refusing printed photos. "
             "For iterating on the capture loop without sitting in front of the camera",
    )
    args = ap.parse_args()

    check_liveness = not args.no_liveness and liveness.available()
    if args.allow_replay and check_liveness:
        print("WARNING: --allow-replay is on. A face displayed on a screen will be accepted as the", file=sys.stderr)
        print("         probe. This is a development convenience and nothing downstream can tell:", file=sys.stderr)
        print("         the evidence bundle records the probe's hash, not how it was captured, so a", file=sys.stderr)
        print("         run made this way is indistinguishable from one made from a live person.", file=sys.stderr)
        print("         Do not use it for anything you intend to anchor as evidence.", file=sys.stderr)
    if not args.no_liveness and not liveness.available():
        print("WARNING: anti-spoofing weights not installed (run models/fetch.sh) — capturing WITHOUT a", file=sys.stderr)
        print("         liveness check. A photograph held to the camera would be accepted.", file=sys.stderr)
    if not fiqa.available():
        print("WARNING: quality model not installed (run models/fetch.sh) — falling back to the", file=sys.stderr)
        print("         Laplacian heuristic, which is a weaker judge of a usable face.", file=sys.stderr)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)  # let OpenCV pick a backend
    if not cap.isOpened():
        print(f"ERROR: no camera at index {args.camera} (try --camera 1, or check /dev/video*)", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    headless = args.no_preview
    window = "face-anchor · probe capture"
    if not headless:
        try:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        except cv2.error:
            print("no display available — falling back to headless capture", file=sys.stderr)
            headless = True

    if headless:
        print(f"headless capture: best gate-passing frame of a {HEADLESS_BURST_S:.0f}s burst (up to {args.timeout:.0f}s)", file=sys.stderr)

    buffer: deque[Shot] = deque(maxlen=max(args.burst, 1))
    try:
        deadline = time.monotonic() + args.timeout
        burst_ends: float | None = None
        seen, last_message, live_result = 0, "", None
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: camera stopped returning frames", file=sys.stderr)
                return 1
            seen += 1
            if seen <= args.warmup:
                continue

            face, q, message = _gate(frame, args.face_index)
            if q is not None and q.ok:
                buffer.append(Shot(frame.copy(), face, q, _score(frame, face, q)))

            if headless:
                reason_key = message.split(" (")[0]
                if reason_key != last_message:
                    print(f"  {message}", file=sys.stderr)
                    last_message = reason_key
                if buffer and burst_ends is None:
                    burst_ends = time.monotonic() + HEADLESS_BURST_S
                if burst_ends is not None and time.monotonic() >= burst_ends:
                    if _keep_best(args.out, buffer, args.face_index, check_liveness, args.allow_replay):
                        return 0
                    # Nothing in this burst survived: start a fresh one rather
                    # than retrying the same frames, which would fail the same way.
                    buffer.clear()
                    burst_ends = None
                if time.monotonic() > deadline:
                    print(f"ERROR: {args.timeout:.0f}s elapsed without a usable frame ({message})", file=sys.stderr)
                    return 1
                continue

            if check_liveness and face is not None and seen % LIVENESS_PREVIEW_EVERY == 0:
                live_result = liveness.check(frame, face)
            elif face is None:
                live_result = None

            cv2.imshow(window, _overlay(frame, face, q, message, live_result, len(buffer), args.allow_replay))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("capture aborted", file=sys.stderr)
                return 2
            if key in (32, 13):
                if not buffer:
                    print(f"  not yet: {message}", file=sys.stderr)
                elif _keep_best(args.out, buffer, args.face_index, check_liveness, args.allow_replay):
                    return 0
                else:
                    buffer.clear()
            elif key == ord("f"):
                # Escape hatch for a stubborn webcam. The pipeline's gate is
                # unchanged and will still reject it — that's a real outcome,
                # not a demo bug.
                print(f"  forcing a capture that failed the gate: {message}", file=sys.stderr)
                forced = Shot(frame, face, q, 0.0) if q is not None else None
                if forced is None:
                    print("  ...but there is no face in frame at all, so there is nothing to write", file=sys.stderr)
                    continue
                _save(args.out, forced, args.face_index)
                return 0
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
