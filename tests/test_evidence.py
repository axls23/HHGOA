import copy

from faceanchor.evidence.bundle import ProbeInfo, build_bundle, bundle_digest, sha256_hex
from faceanchor.evidence.cid import compute_cid
from faceanchor.evidence.jcs import canonicalize, digest
from faceanchor.search.models import Candidate, ScoredMatch


def _sample_bundle() -> dict:
    probe = ProbeInfo(
        image_sha256="a" * 64,
        image_phash="deadbeefcafebabe",
        embed_model="sface_2021dec",
        embed_sha256="b" * 64,
        consent_digest="c" * 64,
    )
    match = ScoredMatch(
        candidate=Candidate(
            platform="commons",
            image_url="https://upload.wikimedia.org/x.jpg",
            post_uri="https://commons.wikimedia.org/wiki/File:x.jpg",
            author=None,
            record_cid=None,
        ),
        score=0.5123,
        metric="cosine",
        image_bytes_sha256="d" * 64,
    )
    return build_bundle(
        pipeline_commit="a1b2c3d",
        probe=probe,
        match=match,
        match_text_sha256="e" * 64,
        match_image_url_sha256="f" * 64,
        run_id="01JFIXEDRUNID000000000000",
        observed_at="2026-08-31T09:12:44Z",
    )


def test_canonicalize_sorts_keys_deterministically():
    a = canonicalize({"b": 1, "a": 2})
    b = canonicalize({"a": 2, "b": 1})
    assert a == b
    assert a == b'{"a":2,"b":1}'


def test_jcs_digest_is_32_bytes_and_deterministic():
    bundle = _sample_bundle()
    d1 = digest(bundle)
    d2 = digest(copy.deepcopy(bundle))
    assert len(d1) == 32
    assert d1 == d2


def test_bundle_digest_changes_on_tamper():
    bundle = _sample_bundle()
    d_original = bundle_digest(bundle)

    tampered = copy.deepcopy(bundle)
    tampered["score"]["value"] = 0.9999
    d_tampered = bundle_digest(tampered)

    assert d_original != d_tampered


def test_bundle_excludes_raw_biometric_fields():
    bundle = _sample_bundle()
    serialized = str(bundle)
    # only hashes/digests of biometric material may appear — never raw vectors
    assert "embed_sha256" in bundle["probe"]
    assert "embedding" not in serialized.lower()
    assert "vector" not in serialized.lower()


def test_compute_cid_is_deterministic_and_versioned():
    data = b"evidence bundle bytes"
    cid1 = compute_cid(data)
    cid2 = compute_cid(data)
    assert cid1 == cid2
    assert cid1.startswith("bafkrei")  # CIDv1, raw codec, base32


def test_compute_cid_differs_for_different_bytes():
    assert compute_cid(b"a") != compute_cid(b"b")


def test_sha256_hex_helper():
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
