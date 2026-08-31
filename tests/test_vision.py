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
