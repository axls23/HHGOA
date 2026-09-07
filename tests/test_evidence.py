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


def test_candidate_log_ties_every_url_to_the_probe(tmp_path):
    """The development log must answer 'what did this run look at, and why did
    each URL lose', and must be unmistakably keyed to one probe image."""
    import json

    from faceanchor.cli import _write_candidate_log
    from faceanchor.search.fanout import SourceReport

    log = tmp_path / "candidates.jsonl"
    written = _write_candidate_log(
        path=log,
        probe_path=tmp_path / "probe.jpg",
        probe_sha256="a" * 64,
        probe_phash="6505ba791b613c93",
        discovery={"mode": "by_face", "corpus": ["mastodon:tag/selfie"], "corpus_images": 2},
        source_reports=[SourceReport("mastodon:tag/selfie", 2, 40.0), SourceReport("bsky:feed/x", 0, 5.0, "HTTP 403")],
        fetch_trace=[
            {"platform": "mastodon", "image_url": "https://x/1.jpg", "post_uri": "https://x/p1", "author": "a",
             "variant": "preview", "fetch": "ok", "status": 206, "bytes": 4000},
            {"platform": "mastodon", "image_url": "https://x/2.png", "post_uri": "https://x/p2", "author": "b",
             "variant": "preview", "fetch": "truncated", "status": 206, "bytes": 524288},
        ],
        score_trace=[
            {"platform": "mastodon", "image_url": "https://x/1.jpg", "post_uri": "https://x/p1", "author": "a",
             "verdict": "below_threshold", "cosine": 0.0465},
        ],
    )
    assert written == 2

    lines = log.read_text().splitlines()
    header = json.loads(lines[0])
    assert header["probe"]["sha256"] == "a" * 64
    assert header["discovery"]["mode"] == "by_face"
    assert header["sources"][1]["error"] == "HTTP 403"   # a blocked source stays visible in the log

    rows = {json.loads(x)["image_url"]: json.loads(x) for x in lines[1:]}
    assert rows["https://x/1.jpg"]["verdict"] == "below_threshold"   # fetch + score merged on one line
    assert rows["https://x/1.jpg"]["cosine"] == 0.0465
    assert rows["https://x/1.jpg"]["fetch"] == "ok"
    assert rows["https://x/2.png"]["verdict"] == "not_scored"        # never reached the cascade
    assert rows["https://x/2.png"]["fetch"] == "truncated"


def test_bundle_records_discovery_mode():
    """A by-face hit and a text-seeded hit are different claims; the anchored
    bundle must not let them look alike, and must never carry the raw query."""
    probe = ProbeInfo(image_sha256="a"*64, image_phash="deadbeefcafebabe", embed_model="sface_2021dec",
                      embed_sha256="b"*64, consent_digest="c"*64)
    match = ScoredMatch(
        candidate=Candidate(platform="mastodon", image_url="https://x/y.jpg", post_uri="https://x/p"),
        score=0.51, metric="cosine", image_bytes_sha256="d"*64,
    )
    kwargs = dict(pipeline_commit="dev", probe=probe, match=match,
                  match_text_sha256="e"*64, match_image_url_sha256="f"*64,
                  run_id="01JFIXEDRUNID000000000000", observed_at="2026-09-01T00:00:00Z")

    by_face = build_bundle(**kwargs, discovery={"mode": "by_face", "corpus": ["mastodon:tag/selfie"], "corpus_images": 243})
    text = build_bundle(**kwargs, discovery={"mode": "text_seeded", "query_sha256": "9"*64})
    plain = build_bundle(**kwargs)

    assert by_face["run"]["discovery"]["mode"] == "by_face"
    assert text["run"]["discovery"]["query_sha256"] == "9"*64
    assert "discovery" not in plain["run"]
    assert bundle_digest(by_face) != bundle_digest(text)   # the mode is part of what is anchored


def test_candidate_log_redacts_signed_urls():
    """A signed CDN URL is a bearer token with an expiry. The log needs to say
    which image was seen, not carry the grant that let us see it."""
    from faceanchor.cli import redact_url

    meta = ("https://scontent.cdninstagram.com/v/t51.29350-15/123_n.jpg"
            "?_nc_ht=scontent.cdninstagram.com&_nc_ohc=AbCdEf&oh=00_AYBxyz&oe=6712ABCD&ig_cache_key=xyz")
    out = redact_url(meta)
    assert "oh=" not in out and "oe=" not in out and "_nc_ohc" not in out
    assert out.startswith("https://scontent.cdninstagram.com/v/t51.29350-15/123_n.jpg")
    assert "ig_cache_key=xyz" in out          # identifying params survive
    assert "redacted=signature" in out        # and the redaction is visible, not silent

    plain = "https://files.mastodon.social/media_attachments/files/1/small/a.jpg"
    assert redact_url(plain) == plain         # untouched when there is nothing to strip

    s3 = "https://x/y.jpg?X-Amz-Signature=deadbeef&X-Amz-Credential=AKIA"
    assert "deadbeef" not in redact_url(s3)


def test_contact_sheet_inlines_images_and_marks_the_hit(tmp_path):
    """The sheet exists to answer 'is this match real?', so it must inline the
    bytes (links rot), order by score, and visually distinguish what cleared
    the threshold."""
    from pathlib import Path

    from faceanchor.review import write_contact_sheet

    fixture = (Path(__file__).parent / "fixtures" / "probe_sample.jpg").read_bytes()
    cands = [
        (Candidate(platform="mastodon", image_url="https://x/low.jpg", post_uri="https://x/low", author="lo"), fixture),
        (Candidate(platform="bsky", image_url="https://x/high.jpg", post_uri="https://x/high", author="hi"), fixture),
        (Candidate(platform="bsky", image_url="https://x/broken.jpg", post_uri="https://x/broken"), b"not an image"),
    ]
    trace = [
        {"image_url": "https://x/low.jpg", "verdict": "below_threshold", "cosine": 0.11},
        {"image_url": "https://x/high.jpg", "verdict": "match_cosine", "cosine": 0.52},
    ]
    out = tmp_path / "sheet.html"
    shown = write_contact_sheet(
        path=out, probe_bytes=fixture, probe_label="probe.jpg", fetched=cands,
        score_trace=trace, threshold=0.363,
        discovery={"mode": "by_face", "corpus": ["mastodon:tag/selfie"]},
    )
    assert shown == 3
    page = out.read_text()

    assert page.count("data:image/jpeg;base64,") >= 3      # probe + both decodable candidates inlined
    assert "https://x/broken" in page and "no decodable image" in page  # undecodable still listed, not dropped
    assert 'class="hit"' in page                            # the above-threshold one is marked
    assert page.index("0.5200") < page.index("0.1100")      # ordered by score, best first
    assert "accept threshold 0.363" in page
