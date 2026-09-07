from pathlib import Path

import cv2
import numpy as np
import pytest

from faceanchor.vision import align, detect, embed, phash, quality
from faceanchor.vision.decode import decode_jpeg_bytes

FIXTURE = Path(__file__).parent / "fixtures" / "probe_sample.jpg"


@pytest.fixture(scope="module")
def probe_image() -> np.ndarray:
    return decode_jpeg_bytes(FIXTURE.read_bytes())


@pytest.fixture(scope="module")
def probe_face(probe_image: np.ndarray) -> detect.Face:
    faces = detect.detect_faces(probe_image)
    assert len(faces) == 1
    return faces[0]


def test_detect_finds_one_face(probe_image: np.ndarray):
    faces = detect.detect_faces(probe_image)
    assert len(faces) == 1
    assert faces[0].score > 0.5


def test_align_produces_112x112_crop(probe_image: np.ndarray, probe_face: detect.Face):
    crop = align.align_and_crop(probe_image, probe_face)
    assert crop.shape == (112, 112, 3)
    assert crop.dtype == np.uint8


def test_embed_one_is_deterministic_and_unit_norm(probe_image: np.ndarray, probe_face: detect.Face):
    crop = align.align_and_crop(probe_image, probe_face)
    e1 = embed.embed_one(crop)
    e2 = embed.embed_one(crop)
    assert e1.shape == (128,)
    np.testing.assert_allclose(e1, e2)
    np.testing.assert_allclose(np.linalg.norm(e1), 1.0, atol=1e-5)


def test_embed_batch_matches_embed_one(probe_image: np.ndarray, probe_face: detect.Face):
    crop = align.align_and_crop(probe_image, probe_face)
    single = embed.embed_one(crop)
    batch = embed.embed_batch([crop, crop])
    assert batch.shape == (2, 128)
    cos = embed.cosine_similarity(single, batch[0])
    assert cos > 0.999


def test_phash_hamming_zero_for_identical_image(probe_image: np.ndarray):
    h1 = phash.phash(probe_image)
    h2 = phash.phash(probe_image)
    assert phash.hamming_distance(h1, h2) == 0


def test_phash_hamming_nonzero_for_different_image(probe_image: np.ndarray):
    flipped = cv2.flip(probe_image, 1)
    h1 = phash.phash(probe_image)
    h2 = phash.phash(flipped)
    assert phash.hamming_distance(h1, h2) > 0


def test_quality_gate_passes_on_upscaled_fixture(probe_image: np.ndarray):
    # The raw fixture is 112x112 with a face bbox of ~79px — 1px under the
    # min-bbox floor, so it exercises the reject path (see the min-bbox test
    # below) rather than a clean pass. Upscale 2x to get comfortably above the
    # 80px floor and confirm the gate passes a good-quality probe.
    upscaled = cv2.resize(probe_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    faces = detect.detect_faces(upscaled)
    assert len(faces) == 1
    result = quality.check_quality(upscaled, faces[0])
    assert result.ok, result.reason


def test_quality_gate_rejects_fixture_min_bbox(probe_image: np.ndarray, probe_face: detect.Face):
    # At native 112x112 the fixture's face bbox (~79px) sits just under the
    # 80px floor — documents the exact boundary the gate enforces.
    result = quality.check_quality(probe_image, probe_face)
    assert not result.ok
    assert "too small" in result.reason


def test_quality_gate_rejects_tiny_face(probe_image: np.ndarray, probe_face: detect.Face):
    import dataclasses

    tiny = dataclasses.replace(probe_face, bbox=np.array([0, 0, 40, 40], dtype=np.float32))
    result = quality.check_quality(probe_image, tiny)
    assert not result.ok
    assert "too small" in result.reason


def test_check_single_face_no_detection():
    face, err = quality.check_single_face([], face_index=None)
    assert face is None
    assert "no face detected" in err


def test_check_single_face_ambiguous_multiface():
    fake = [
        detect.Face(bbox=np.zeros(4), landmarks=np.zeros((5, 2)), score=0.9, raw_row=np.zeros(15)),
        detect.Face(bbox=np.zeros(4), landmarks=np.zeros((5, 2)), score=0.9, raw_row=np.zeros(15)),
    ]
    face, err = quality.check_single_face(fake, face_index=None)
    assert face is None
    assert "face-index" in err

    face, err = quality.check_single_face(fake, face_index=1)
    assert face is fake[1]
    assert err is None


# --- Stage 1 capture models: head pose, learned quality, liveness -----------


@pytest.fixture(scope="module")
def upscaled(probe_image: np.ndarray) -> np.ndarray:
    """The fixture at 2x, where the face clears the 80px floor and the gate passes."""
    return cv2.resize(probe_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def test_pose_tracks_image_rotation(upscaled: np.ndarray):
    """Roll is the one angle a synthetic transform can produce exactly.

    Rotating the image rotates the head in it, so roll must follow and yaw must
    not. This is what the old landmark-asymmetry ratio could not see at all.
    """
    from faceanchor.vision import pose

    h, w = upscaled.shape[:2]
    baseline = pose.estimate(upscaled, detect.detect_faces(upscaled)[0])
    assert baseline is not None

    measured = {}
    for degrees in (-20, 20):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
        rotated = cv2.warpAffine(upscaled, matrix, (w, h))
        faces = detect.detect_faces(rotated)
        assert faces, f"detector lost the face at {degrees} degrees"
        result = pose.estimate(rotated, faces[0])
        assert result is not None
        measured[degrees] = result

    # Opposite rotations must land on opposite sides of the baseline roll, and
    # each must move by most of the angle applied.
    assert measured[-20].roll > baseline.roll + 8
    assert measured[20].roll < baseline.roll - 8

    # Yaw should not move much — rolling the image does not turn the head — but
    # it does move some: solving six degrees of freedom from five noisy points
    # couples the axes, and warping the image resamples the landmarks it solves
    # from. The honest claim is that roll absorbs most of the change, not that
    # yaw is untouched.
    for degrees, result in measured.items():
        roll_delta = abs(result.roll - baseline.roll)
        yaw_delta = abs(result.yaw - baseline.yaw)
        assert yaw_delta < roll_delta, f"at {degrees} deg the rotation leaked into yaw ({yaw_delta:.1f} vs roll {roll_delta:.1f})"


def test_pose_is_centred_on_a_real_population():
    """The calibration claim in pose.py, asserted.

    An absolute threshold in degrees only means something if an ordinary
    portrait reads as ~0. Skips where the reference corpus isn't built.
    """
    import json

    from faceanchor.vision import pose

    db = Path(__file__).parent.parent / "evidence" / "corpus-db"
    manifest = db / "manifest.jsonl"
    if not manifest.exists():
        pytest.skip("reference corpus not built (faceanchor corpus-build --db evidence/corpus-db)")

    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()][1:]
    angles = []
    for row in rows:
        raw = (db / "images" / row["sha256"][:2] / row["sha256"]).read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        faces = detect.detect_faces(image)
        if not faces:
            continue
        result = pose.estimate(image, max(faces, key=lambda f: f.score))
        if result is not None:
            angles.append((result.yaw, result.pitch, result.roll))

    assert len(angles) > 50, "not enough faces to say anything about the population"
    yaw, pitch, roll = (np.median([a[i] for a in angles]) for i in range(3))
    assert abs(yaw) < 8, f"yaw is biased: median {yaw:.1f} degrees"
    assert abs(pitch) < 8, f"pitch is biased: median {pitch:.1f} degrees"
    assert abs(roll) < 8, f"roll is biased: median {roll:.1f} degrees"


def test_fiqa_ranks_degraded_faces_lower(upscaled: np.ndarray):
    """The threshold in fiqa.py is only defensible if the score responds to the
    degradations it is supposed to catch — and does not collapse on a good face."""
    from faceanchor.vision import fiqa

    if not fiqa.available():
        pytest.skip("eDifFIQA weights not installed (models/fetch.sh)")

    crop = align.align_and_crop(upscaled, detect.detect_faces(upscaled)[0])
    good = fiqa.score(crop)
    blurred = fiqa.score(cv2.GaussianBlur(crop, (21, 21), 0))
    small = cv2.resize(crop, (14, 14))
    pixelated = fiqa.score(cv2.resize(small, (112, 112), interpolation=cv2.INTER_NEAREST))

    assert good > fiqa.MIN_FIQA_SCORE, f"a clean fixture scored {good:.2f}, below the gate"
    assert blurred < good
    assert pixelated < good
    assert min(blurred, pixelated) < fiqa.MIN_FIQA_SCORE, "the gate would pass a face nobody could recognise"


def test_liveness_accepts_a_real_camera_frame():
    """Liveness needs the *frame*, not a face crop.

    MiniFASNet widens the detection box 2.7-4x before it looks — the wider crop
    is what sees a phone bezel or the edge of a sheet of paper. On a 112x112
    crop that widening is clamped to the image border, the model gets none of
    the context it was trained on, and it calls a genuine face an attack.
    That is why this uses the camera-frame fixture, and why liveness runs at
    capture time on a full frame rather than inside the quality gate.
    """
    from faceanchor.vision import liveness

    if not liveness.available():
        pytest.skip("anti-spoofing weights not installed (models/fetch.sh)")

    frame = decode_jpeg_bytes((Path(__file__).parent / "fixtures" / "capture_frame.jpg").read_bytes())
    result = liveness.check(frame, detect.detect_faces(frame)[0])
    assert result.is_live, f"a real camera frame was called an attack: {result.reason}"
    assert result.live_probability > liveness.MIN_LIVE_PROBABILITY


def test_liveness_refuses_an_undecided_ensemble(upscaled: np.ndarray, monkeypatch):
    """The gate must not read argmax alone: a three-way tie is not a live face."""
    from faceanchor.vision import liveness, models

    class _FlatSession:
        def get_inputs(self):
            class _Input:
                name = "input"

            return [_Input()]

        def run(self, _outputs, _feed):
            # 0.34 / 0.35 / 0.31 after softmax-ish: class 1 wins, but weakly.
            return [np.array([[0.02, 0.05, 0.0]], dtype=np.float32)]

    monkeypatch.setattr(models, "get_antispoof_sessions", lambda: ((_FlatSession(), 2.7), (_FlatSession(), 4.0)))
    monkeypatch.setattr(liveness, "get_antispoof_sessions", models.get_antispoof_sessions)

    result = liveness.check(upscaled, detect.detect_faces(upscaled)[0])
    assert not result.is_live
    assert "undecided" in result.reason


def test_quality_gate_reports_pose_and_learned_quality(upscaled: np.ndarray):
    """Both new signals reach the caller, so a rejection can say which one failed."""
    from faceanchor.vision import fiqa

    result = quality.check_quality(upscaled, detect.detect_faces(upscaled)[0])
    assert result.ok, result.reason
    assert result.head_pose is not None
    if fiqa.available():
        assert result.fiqa is not None and result.fiqa > fiqa.MIN_FIQA_SCORE


def test_laplacian_and_learned_quality_disagree_on_a_pixelated_face(upscaled: np.ndarray):
    """Why the decision moved off Laplacian variance, demonstrated.

    Nearest-neighbour pixelation destroys a face while *raising* the metric the
    old gate judged with: the block edges are high-frequency energy. Laplacian
    reads an order of magnitude above its threshold and calls this fine; the
    learned score, trained against recognition outcomes, drops sharply. Neither
    number is wrong — they answer different questions, and only one of them is
    the question this gate is asking.
    """
    from faceanchor.vision import fiqa

    if not fiqa.available():
        pytest.skip("eDifFIQA weights not installed (models/fetch.sh)")

    h, w = upscaled.shape[:2]
    pixelated = cv2.resize(
        cv2.resize(upscaled, (w // 6, h // 6)), (w, h), interpolation=cv2.INTER_NEAREST
    )
    faces = detect.detect_faces(pixelated)
    assert faces, "the detector should still find this face — that is the point"

    clean = quality.check_quality(upscaled, detect.detect_faces(upscaled)[0])
    wrecked = quality.check_quality(pixelated, faces[0])

    # Laplacian is not merely fooled, it is fooled *upwards*.
    assert wrecked.laplacian_variance > clean.laplacian_variance
    assert wrecked.laplacian_variance > 10 * quality.MIN_LAPLACIAN_VARIANCE
    # The learned score moves the way a human would.
    assert wrecked.fiqa < clean.fiqa - 0.05


# --- the YOLOv8-face backend ------------------------------------------------


def test_yolo_backend_returns_the_same_shaped_face(yolo_backend, upscaled: np.ndarray):
    """The contract, not the numbers: whatever runs, downstream cannot tell."""
    from faceanchor.vision import detect as detect_module

    assert detect_module.backend() == "yolo"
    faces = detect_module.detect_faces(upscaled)
    assert len(faces) == 1
    face = faces[0]
    assert face.raw_row.shape == (15,)
    assert face.landmarks.shape == (5, 2)
    assert 0.0 < face.score <= 1.0
    # alignCrop reads raw_row directly — if the layout were wrong this raises
    # or produces garbage rather than a 112x112 crop.
    assert align.align_and_crop(upscaled, face).shape == (112, 112, 3)


def test_backends_agree_on_the_same_face(yolo_backend, upscaled: np.ndarray, monkeypatch):
    """Probe and candidates must go through one backend, and this is why.

    The two detectors align slightly differently, so the same face does not
    embed identically across them. The agreement is high enough that a person
    still matches themselves, and low enough that mixing backends across the
    probe/candidate boundary would move scores near the threshold.
    """
    from faceanchor.vision import detect as detect_module

    yolo_face = detect_module.detect_faces(upscaled)[0]
    yolo_embedding = embed.embed_one(align.align_and_crop(upscaled, yolo_face))

    monkeypatch.setenv("FACEANCHOR_DETECTOR", "yunet")
    yunet_face = detect_module.detect_faces(upscaled)[0]
    yunet_embedding = embed.embed_one(align.align_and_crop(upscaled, yunet_face))

    agreement = float(np.dot(yolo_embedding, yunet_embedding))
    assert agreement > 0.85, f"the backends disagree about this face ({agreement:.3f})"
    assert agreement < 0.999, "identical embeddings would mean the backends are not actually different"

    # ...and the boxes differ, which is what bbox_scale() compensates for.
    yolo_side = min(yolo_face.bbox[2], yolo_face.bbox[3])
    yunet_side = min(yunet_face.bbox[2], yunet_face.bbox[3])
    assert 0.7 < yolo_side / yunet_side < 1.1


def test_size_floor_follows_the_backend(yolo_backend):
    """A gate at "80px" must mean the same face size whichever model drew the box."""
    from faceanchor.vision import detect as detect_module

    assert detect_module.bbox_scale() < 1.0
    assert detect_module.bbox_scale() > 0.5


def test_unknown_backend_is_refused(monkeypatch):
    from faceanchor.vision import detect as detect_module

    monkeypatch.setenv("FACEANCHOR_DETECTOR", "retinaface")
    with pytest.raises(ValueError, match="retinaface"):
        detect_module.backend()
