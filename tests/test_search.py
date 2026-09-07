from faceanchor.search.bluesky import _extract_image_candidates as bsky_extract
from faceanchor.search.fanout import sha256_hex
from faceanchor.search.mastodon import _extract_image_candidates as mastodon_extract
from faceanchor.search.models import Candidate
from faceanchor.vision.phash import hamming_distance


def test_bluesky_extract_image_candidates():
    post = {
        "uri": "at://did:plc:abc123/app.bsky.feed.post/xyz789",
        "cid": "bafyreiabc123",
        "author": {"handle": "alice.bsky.social"},
        "record": {"text": "hello world"},
        "embed": {
            "images": [
                {"fullsize": "https://cdn.bsky.app/img/feed_fullsize/plain/abc/def.jpg", "alt": "a photo"},
            ]
        },
    }
    candidates = bsky_extract(post)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.platform == "bsky"
    assert c.image_url == "https://cdn.bsky.app/img/feed_fullsize/plain/abc/def.jpg"
    assert c.author == "alice.bsky.social"
    assert c.record_cid == "bafyreiabc123"
    assert c.text == "hello world"


def test_bluesky_extract_handles_record_with_media_embed():
    post = {
        "uri": "at://did:plc:abc/app.bsky.feed.post/1",
        "cid": "bafyreidef",
        "author": {"handle": "bob.bsky.social"},
        "record": {"text": "quote post"},
        "embed": {"media": {"images": [{"fullsize": "https://cdn.bsky.app/img/2.jpg"}]}},
    }
    candidates = bsky_extract(post)
    assert len(candidates) == 1
    assert candidates[0].image_url == "https://cdn.bsky.app/img/2.jpg"


def test_bluesky_extract_no_images_returns_empty():
    post = {"uri": "at://x", "cid": "y", "author": {"handle": "z"}, "record": {"text": "no images here"}}
    assert bsky_extract(post) == []


def test_mastodon_extract_image_candidates():
    status = {
        "url": "https://mastodon.social/@alice/12345",
        "account": {"acct": "alice"},
        "content": "<p>a post</p>",
        "media_attachments": [
            {"type": "image", "url": "https://files.mastodon.social/media/1.jpg", "description": "a face"},
            {"type": "video", "url": "https://files.mastodon.social/media/2.mp4"},
        ],
    }
    candidates = mastodon_extract(status)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.platform == "mastodon"
    assert c.image_url == "https://files.mastodon.social/media/1.jpg"
    assert c.author == "alice"
    assert c.extra["description"] == "a face"


def test_mastodon_extract_ignores_non_image_media():
    status = {
        "url": "https://mastodon.social/@bob/1",
        "account": {"acct": "bob"},
        "content": "",
        "media_attachments": [{"type": "audio", "url": "https://files.mastodon.social/media/a.mp3"}],
    }
    assert mastodon_extract(status) == []


def test_sha256_hex_matches_known_vector():
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hamming_distance_zero_for_identical_hashes():
    h = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    assert hamming_distance(h, h) == 0


def test_hamming_distance_counts_bit_differences():
    a = bytes([0b00000000])
    b = bytes([0b00000011])
    assert hamming_distance(a, b) == 2


# --- handle routes (added after a live audit found searchPosts blocked and
# --- Mastodon's anonymous status search auth-gated; both arms grew an
# --- account-lookup route that works without credentials) ---


def test_bluesky_parse_actor_accepts_handles_and_dids():
    from faceanchor.search.bluesky import _parse_actor

    assert _parse_actor("alice.bsky.social") == "alice.bsky.social"
    assert _parse_actor("@alice.bsky.social") == "alice.bsky.social"
    assert _parse_actor("did:plc:z72i7hdynmk6r22z27h6tvur") == "did:plc:z72i7hdynmk6r22z27h6tvur"


def test_bluesky_parse_actor_rejects_non_handles():
    from faceanchor.search.bluesky import _parse_actor

    assert _parse_actor("portrait") is None            # bare word — not an actor
    assert _parse_actor("official portrait photo") is None  # phrase
    assert _parse_actor("") is None


def test_bluesky_avatar_candidate():
    from faceanchor.search.bluesky import _avatar_candidate

    c = _avatar_candidate(
        {"handle": "alice.bsky.social", "did": "did:plc:abc", "avatar": "https://cdn.bsky.app/img/avatar/a.jpg",
         "description": "hi"}
    )
    assert len(c) == 1
    assert c[0].platform == "bsky"
    assert c[0].extra["kind"] == "avatar"
    assert c[0].post_uri == "at://did:plc:abc"
    assert _avatar_candidate({"handle": "no-avatar.bsky.social"}) == []


def test_mastodon_parse_handle_shapes():
    from faceanchor.search.mastodon import _parse_handle

    assert _parse_handle("@Gargron") == ("Gargron", None)
    assert _parse_handle("Gargron@mastodon.social") == ("Gargron", "mastodon.social")
    assert _parse_handle("@Gargron@mastodon.social") == ("Gargron", "mastodon.social")
    assert _parse_handle("portrait") == ("portrait", None)   # ambiguous: tried as both account and tag
    assert _parse_handle("two words") is None


def test_mastodon_avatar_candidate():
    from faceanchor.search.mastodon import _avatar_candidate

    c = _avatar_candidate({"acct": "alice", "url": "https://mastodon.social/@alice",
                           "avatar": "https://files.mastodon.social/a.png", "note": "bio"})
    assert len(c) == 1 and c[0].extra["kind"] == "avatar" and c[0].author == "alice"
    assert _avatar_candidate({"acct": "alice"}) == []


def test_source_report_distinguishes_blocked_from_empty():
    """The whole point of SourceReport: 0-because-blocked must not read the
    same as 0-because-no-hits."""
    from faceanchor.search.fanout import FanoutResult, SourceReport

    blocked = SourceReport("bsky", 0, 12.0, error="HTTP 403")
    empty = SourceReport("mastodon", 0, 30.0)
    assert blocked.describe() == "bsky FAILED (HTTP 403)"
    assert empty.describe() == "mastodon 0"

    result = FanoutResult(candidates=[], reports=[blocked, empty])
    assert result.degraded is True
    assert result.summary() == "bsky FAILED (HTTP 403), mastodon 0"

    assert FanoutResult(candidates=[], reports=[empty]).degraded is False


async def test_run_source_reports_http_status_error():
    import httpx

    from faceanchor.search.fanout import _run_source

    async def blocked():
        request = httpx.Request("GET", "https://example.invalid/x")
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("403", request=request, response=response)

    candidates, report = await _run_source("bsky", blocked(), timeout=1.0)
    assert candidates == []
    assert report.error == "HTTP 403"
    assert report.count == 0


async def test_run_source_reports_timeout():
    import asyncio

    from faceanchor.search.fanout import _run_source

    async def slow():
        await asyncio.sleep(5)
        return []

    candidates, report = await _run_source("slow", slow(), timeout=0.05)
    assert candidates == []
    assert "timeout" in report.error


# --- reverse-image (by-face) corpus discovery ---


def test_corpus_spec_parsing():
    from faceanchor.search.corpus import parse_spec

    s = parse_spec("mastodon:tag/selfie")
    assert (s.platform, s.kind, s.value, s.instance) == ("mastodon", "tag", "selfie", None)

    s = parse_spec("mastodon:tag/portrait@mstdn.social")
    assert (s.value, s.instance) == ("portrait", "mstdn.social")
    assert str(s) == "mastodon:tag/portrait@mstdn.social"

    s = parse_spec("bsky:feed/whats-hot")
    assert (s.platform, s.kind, s.value) == ("bsky", "feed", "whats-hot")

    s = parse_spec("bsky:author/alice.bsky.social")
    assert (s.platform, s.kind, s.value) == ("bsky", "author", "alice.bsky.social")


def test_corpus_spec_rejects_unsupported():
    import pytest

    from faceanchor.search.corpus import CorpusSpecError, parse_spec

    for bad in ["instagram:everything", "mastodon:tag", "nonsense", "bsky:dm/alice", "mastodon:tag/"]:
        with pytest.raises(CorpusSpecError):
            parse_spec(bad)


def test_corpus_prefers_bounded_variant():
    """Corpus scanning must fetch the platform's small variant, not the
    multi-MB original the 512KB cap would truncate."""
    from faceanchor.search.corpus import _prefer_bounded_variant

    c = Candidate(
        platform="mastodon",
        image_url="https://files.mastodon.social/original/huge.jpg",
        post_uri="https://mastodon.social/@a/1",
        extra={"preview_url": "https://files.mastodon.social/small/huge.jpg"},
    )
    out = _prefer_bounded_variant(c)
    assert out.image_url == "https://files.mastodon.social/small/huge.jpg"
    assert out.extra["variant"] == "preview"
    assert out.extra["fullsize_url"] == "https://files.mastodon.social/original/huge.jpg"

    bare = Candidate(platform="ddg", image_url="https://x/y.jpg", post_uri="https://x/y")
    assert _prefer_bounded_variant(bare) is bare


def test_truncated_response_is_dropped_not_decoded():
    import httpx

    from faceanchor.search.fanout import CANDIDATE_BYTE_CAP, _is_truncated

    request = httpx.Request("GET", "https://x/y.png")
    partial = httpx.Response(
        206,
        request=request,
        headers={"content-range": f"bytes 0-{CANDIDATE_BYTE_CAP-1}/{CANDIDATE_BYTE_CAP*6}"},
        content=b"x" * 16,
    )
    assert _is_truncated(partial) is True

    whole = httpx.Response(
        206,
        request=request,
        headers={"content-range": f"bytes 0-1023/1024"},
        content=b"x" * 1024,
    )
    assert _is_truncated(whole) is False


def test_by_face_retrieval_finds_the_probe_in_a_corpus():
    """Positive control for the by-face path with no network: a corpus that
    contains the probe image itself must retrieve it, on the face/phash
    cascade alone, with no text query anywhere in the call."""
    from pathlib import Path

    from faceanchor.search import score
    from faceanchor.vision import detect, quality
    from faceanchor.vision.decode import decode_jpeg_bytes

    fixture = Path(__file__).parent / "fixtures" / "probe_sample.jpg"
    raw = fixture.read_bytes()
    image = decode_jpeg_bytes(raw)
    face, err = quality.check_single_face(detect.detect_faces(image), None)
    assert err is None

    probe_ctx = score.build_probe_context(image, face)
    corpus = [
        (Candidate(platform="mastodon", image_url="https://x/decoy.jpg", post_uri="https://x/1"), b"not an image"),
        (Candidate(platform="mastodon", image_url="https://x/hit.jpg", post_uri="https://x/2"), raw),
    ]
    matches = score.score_candidates(probe_ctx, corpus)
    assert matches, "the probe's own image must be retrieved from the corpus"
    assert matches[0].candidate.post_uri == "https://x/2"


def test_candidates_are_detected_with_the_selected_backend(yolo_backend):
    """The probe/candidate cross-match runs one detector, not two.

    score.py re-detects every candidate before embedding it, so whichever
    backend the probe used is the one the candidates get. Mixing them would
    compare faceprints from differently-aligned crops — the same face reads
    ~0.95 across backends, which is enough to move a marginal candidate.
    """
    from pathlib import Path

    import cv2

    from faceanchor.search import score as score_module
    from faceanchor.vision import detect as detect_module
    from faceanchor.vision.decode import decode_jpeg_bytes

    fixture = Path(__file__).parent / "fixtures" / "capture_frame.jpg"
    image = decode_jpeg_bytes(fixture.read_bytes())
    probe_face = detect_module.detect_faces(image)[0]
    probe = score_module.build_probe_context(image, probe_face)

    # A candidate that is the same face at a different scale: the cascade must
    # detect it with the active backend, embed it, and score it as a match.
    resized = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    raw = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()

    from faceanchor.search.models import Candidate

    candidate = Candidate(
        platform="test", image_url="test://same-face.jpg", post_uri="test://same-face",
    )
    trace: list[dict] = []
    matches = score_module.score_candidates(probe, [(candidate, raw)], trace=trace)

    assert detect_module.backend() == "yolo"
    assert matches, f"the same face did not survive the cascade: {trace}"
    assert matches[0].score >= score_module.COSINE_ACCEPT_THRESHOLD
