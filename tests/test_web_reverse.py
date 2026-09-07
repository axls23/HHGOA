"""Genuine reverse-image discovery: provider, normalization, failure modes,
social filtering, face verification, evidence provenance, and tamper detection.

Nothing here touches a live Google API. The live path is exercised by the
opt-in integration test at the bottom, which skips unless real credentials
are present — a unit suite that needs a paid third party is a unit suite that
stops being run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from faceanchor.evidence.bundle import ProbeInfo, build_bundle, bundle_digest
from faceanchor.evidence.jcs import canonicalize
from faceanchor.search.models import Candidate, ScoredMatch
from faceanchor.search.web_reverse import (
    GoogleWebDetectionProvider,
    MalformedResponseError,
    PlatformMatcher,
    ProviderAuthError,
    ProviderConfigError,
    ProviderHTTPError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ReplayProvider,
    ReverseImageError,
    ReverseImageProvider,
    ReverseImageResult,
    available_providers,
    dedupe_results,
    filter_social,
    get_provider,
)
from faceanchor.search.web_reverse.google_vision import normalize_web_detection

FIXTURES = Path(__file__).parent / "fixtures"
GOOGLE_FIXTURE = FIXTURES / "google_web_detection.json"
PROBE = FIXTURES / "probe_sample.jpg"


def _payload() -> dict:
    return json.loads(GOOGLE_FIXTURE.read_text())


def _provider_with(handler, **kwargs) -> tuple[GoogleWebDetectionProvider, httpx.AsyncClient]:
    provider = GoogleWebDetectionProvider(api_key="test-key-not-a-real-credential", **kwargs)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider, client


# ── 1. the provider interface ────────────────────────────────────────────


def test_provider_interface_is_abstract_and_registered():
    with pytest.raises(TypeError):
        ReverseImageProvider()  # search() is abstract — a provider must implement it

    assert "google" in available_providers()
    google = get_provider("google")
    assert isinstance(google, ReverseImageProvider)
    assert google.name == "google"
    assert google.provider_id == "google_web_detection"


def test_unknown_provider_is_a_config_error_not_a_silent_default():
    with pytest.raises(ProviderConfigError) as e:
        get_provider("definitely-not-a-provider")
    assert "available" in str(e.value)


def test_mock_provider_can_never_masquerade_as_a_real_one():
    """The replay double must be selectable only by name, and must leave a
    permanent mark in whatever it produces."""
    from faceanchor.search.web_reverse import REAL_PROVIDERS

    replay = get_provider("mock")
    assert replay.provider_id == "mock_replay"
    assert replay.name not in REAL_PROVIDERS

    results = normalize_web_detection(_payload(), provider=replay.provider_id)
    assert {r.provider for r in results} == {"mock_replay"}


async def test_mock_provider_refuses_to_invent_results():
    replay = ReplayProvider(response_path=None)
    with pytest.raises(ProviderConfigError):
        await replay.search(PROBE)

    with pytest.raises(ProviderConfigError):
        await ReplayProvider(response_path="/nonexistent/response.json").search(PROBE)


async def test_mock_provider_replays_through_the_production_parser():
    replay = ReplayProvider(response_path=GOOGLE_FIXTURE)
    results = await replay.search(PROBE)
    assert results
    assert all(r.provider == "mock_replay" for r in results)


# ── 2. Google response normalization ─────────────────────────────────────


def test_google_normalization_maps_every_bucket():
    results = normalize_web_detection(_payload())
    by_key = {r.key: r for r in results}

    # a full match found on a page keeps both URLs and the strongest claim
    bsky_page = ("https://bsky.app/profile/alice.bsky.social/post/3kabcdef",
                 "https://cdn.bsky.app/img/feed_fullsize/plain/did:plc:abc/full.jpg")
    assert by_key[bsky_page].match_type == "full"
    assert by_key[bsky_page].provider == "google_web_detection"
    assert by_key[bsky_page].metadata["page_title"] == "alice on Bluesky"

    masto_page = ("https://mastodon.social/@bob/109123456789012345",
                  "https://files.mastodon.social/media_attachments/files/2/original/crop.jpg")
    assert by_key[masto_page].match_type == "partial"

    # top-level buckets have no page URL
    assert by_key[("", "https://cdn.bsky.app/img/feed_fullsize/plain/did:plc:abc/similar.jpg")].match_type == "similar"
    assert by_key[("", "https://files.mastodon.social/media_attachments/files/1/original/partial.jpg")].match_type == "partial"

    # a page hit with no named image is kept, not silently lost
    assert ("https://mstdn.social/@carol/109999999999999999", "") in by_key


def test_google_normalization_discards_web_entities():
    """Web entities are text labels. Turning them into a query would make this
    a text search wearing a reverse-image search's name."""
    results = normalize_web_detection(_payload())
    blob = json.dumps([r.metadata for r in results])
    assert "webEntities" not in blob and "Hair" not in blob


def test_normalization_dedupes_onto_the_strongest_claim():
    results = dedupe_results(
        [
            ReverseImageResult("p", "https://x/a.jpg", "https://page/1", "similar"),
            ReverseImageResult("p", "https://x/a.jpg", "https://page/1", "full"),
            ReverseImageResult("p", "https://x/a.jpg", "https://page/1", "partial"),
            ReverseImageResult("p", None, None, "full"),  # no locator at all — dropped
        ]
    )
    assert len(results) == 1
    assert results[0].match_type == "full"


def test_result_normalizes_an_unknown_match_type():
    assert ReverseImageResult("p", "https://x/a.jpg", None, "wat").match_type == "unknown"


def test_empty_but_wellformed_response_is_no_results_not_an_error():
    assert normalize_web_detection({"responses": [{}]}) == []


# ── 3. malformed provider responses ──────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        {},                                   # no `responses`
        {"responses": []},                    # empty `responses`
        {"responses": ["not an object"]},     # wrong element type
        {"responses": [{"webDetection": []}]},  # webDetection is not an object
        [],                                   # not an object at all
    ],
)
def test_malformed_responses_raise_rather_than_returning_nothing(payload):
    with pytest.raises(MalformedResponseError):
        normalize_web_detection(payload)


async def test_non_json_body_is_a_malformed_response():
    provider, client = _provider_with(lambda request: httpx.Response(200, content=b"<html>oops</html>"))
    async with client:
        with pytest.raises(MalformedResponseError):
            await provider.search(PROBE, client=client)


def test_inline_annotate_error_is_a_provider_error_not_an_empty_result():
    """Vision reports per-image failures inside a 200 body. That is still a
    failure, and must not read as 'nothing found'."""
    with pytest.raises(ProviderHTTPError):
        normalize_web_detection({"responses": [{"error": {"code": 3, "message": "Bad image data"}}]})

    with pytest.raises(ProviderRateLimitError):
        normalize_web_detection({"responses": [{"error": {"code": 429, "message": "quota"}}]})


# ── 4. authentication failure ────────────────────────────────────────────


async def test_missing_credentials_fail_loudly_with_no_fallback(monkeypatch):
    for env in ("GOOGLE_VISION_API_KEY", "GOOGLE_CLOUD_VISION_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.delenv(env, raising=False)

    provider = GoogleWebDetectionProvider()
    with pytest.raises(ProviderAuthError) as e:
        await provider.search(PROBE)
    assert "no offline fallback" in str(e.value)


async def test_credentials_come_from_the_environment_never_the_source(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "env-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.url.params.get("key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_payload())

    provider = GoogleWebDetectionProvider()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await provider.search(PROBE, client=client)

    assert seen["key"] == "env-key"
    assert seen["body"]["requests"][0]["features"][0]["type"] == "WEB_DETECTION"
    # the image itself is what is submitted — this is what makes it reverse-image search
    assert seen["body"]["requests"][0]["image"]["content"]


def test_no_credentials_are_hardcoded_anywhere_in_the_provider():
    source = (
        Path(__file__).parent.parent / "src" / "faceanchor" / "search" / "web_reverse" / "google_vision.py"
    ).read_text()
    assert "AIza" not in source          # Google API keys all start AIza
    assert "-----BEGIN" not in source    # no embedded private key
    assert "GOOGLE_VISION_API_KEY" in source


@pytest.mark.parametrize("status", [401, 403])
async def test_rejected_credentials_are_an_auth_error(status):
    provider, client = _provider_with(lambda request: httpx.Response(status, json={"error": {"message": "nope"}}))
    async with client:
        with pytest.raises(ProviderAuthError):
            await provider.search(PROBE, client=client)


# ── 5. timeout / HTTP / rate-limit handling ──────────────────────────────


async def test_provider_timeout_is_its_own_error():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    provider, client = _provider_with(handler, timeout_s=0.01)
    async with client:
        with pytest.raises(ProviderTimeoutError):
            await provider.search(PROBE, client=client)


async def test_rate_limit_is_distinguishable_from_a_generic_http_error():
    provider, client = _provider_with(lambda request: httpx.Response(429, text="rate limited"))
    async with client:
        with pytest.raises(ProviderRateLimitError) as e:
            await provider.search(PROBE, client=client)
    assert e.value.status_code == 429


async def test_server_error_is_a_provider_error():
    provider, client = _provider_with(lambda request: httpx.Response(500, text="boom"))
    async with client:
        with pytest.raises(ProviderHTTPError) as e:
            await provider.search(PROBE, client=client)
    assert e.value.status_code == 500


async def test_transport_failure_is_a_provider_error_not_an_empty_result():
    def handler(request):
        raise httpx.ConnectError("dns", request=request)

    provider, client = _provider_with(handler)
    async with client:
        with pytest.raises(ProviderHTTPError):
            await provider.search(PROBE, client=client)


async def test_every_provider_failure_shares_one_base_class():
    """`cli.py` catches ReverseImageError to report PROVIDER_ERROR. If a
    failure mode escaped that hierarchy it would crash the run instead."""
    for exc in (ProviderAuthError, ProviderTimeoutError, MalformedResponseError, ProviderConfigError):
        assert issubclass(exc, ReverseImageError)
    assert issubclass(ProviderRateLimitError, ProviderHTTPError)
    assert issubclass(ProviderHTTPError, ReverseImageError)


async def test_empty_probe_is_refused_before_upload():
    provider = GoogleWebDetectionProvider(api_key="k")
    empty = Path(__file__).parent / "fixtures" / "_empty_probe.bin"
    empty.write_bytes(b"")
    try:
        with pytest.raises(MalformedResponseError):
            await provider.search(empty)
    finally:
        empty.unlink()


# ── 6/7. candidate URL extraction + social-platform filtering ────────────


def test_social_filter_keeps_only_platforms_the_pipeline_can_handle():
    report = filter_social(normalize_web_detection(_payload()))

    assert {c.platform for c in report.candidates} == {"bsky", "mastodon"}
    assert all(c.image_url for c in report.candidates)

    urls = {c.image_url for c in report.candidates}
    assert "https://news.example.com/img/hero.jpg" not in urls
    assert "https://example.com/not-social/similar.jpg" not in urls

    reasons = {entry["reason"] for entry in report.skipped}
    assert "not_social" in reasons     # the news site
    assert "no_image_url" in reasons   # carol's page, no named image
    assert "duplicate" in reasons      # the bsky full match, seen twice


def test_social_candidates_carry_page_and_match_provenance():
    report = filter_social(normalize_web_detection(_payload()))
    bsky = next(c for c in report.candidates if c.platform == "bsky")
    assert bsky.post_uri == "https://bsky.app/profile/alice.bsky.social/post/3kabcdef"
    assert bsky.extra["match_type"] == "full"
    assert bsky.extra["provider"] == "google_web_detection"
    assert bsky.extra["discovery"] == "web_reverse"


def test_candidate_without_a_page_url_still_gets_a_locator():
    """A top-level full match has no page. The image URL is then the only
    locator there is; an empty `uri` in an evidence bundle would be a lie."""
    report = filter_social([ReverseImageResult("p", "https://cdn.bsky.app/img/x.jpg", None, "full")])
    assert report.candidates[0].post_uri == "https://cdn.bsky.app/img/x.jpg"
    assert report.candidates[0].extra["page_url"] == ""


def test_platform_matcher_classifies_by_domain_and_subdomain():
    m = PlatformMatcher.build(env={})
    assert m.classify("https://bsky.app/profile/a/post/1", None) == "bsky"
    assert m.classify(None, "https://cdn.bsky.app/img/x.jpg") == "bsky"
    assert m.classify("https://mastodon.social/@a/1", None) == "mastodon"
    assert m.classify("https://news.example.com/story", None) is None
    assert m.classify(None, None) is None
    assert m.classify("not a url at all", None) is None


def test_platform_matcher_heuristic_covers_unknown_mastodon_instances():
    """Mastodon is thousands of domains; the URL layout is Mastodon's own."""
    m = PlatformMatcher.build(env={})
    assert m.classify("https://social.invalid.example/@dave/109876543210987654", None) == "mastodon"
    assert m.classify(None, "https://media.invalid.example/system/media_attachments/files/1/a.jpg") == "mastodon"

    strict = PlatformMatcher.build(env={}, heuristics=False)
    assert strict.classify("https://social.invalid.example/@dave/109876543210987654", None) is None


def test_platform_domains_are_configurable_from_cli_and_env():
    from faceanchor.search.web_reverse.social import ENV_EXTRA_DOMAINS

    m = PlatformMatcher.build(["mastodon:corp.example"], env={}, heuristics=False)
    assert m.classify("https://corp.example/anything", None) == "mastodon"

    m = PlatformMatcher.build(env={ENV_EXTRA_DOMAINS: "bsky:pds.example,mastodon:other.example"}, heuristics=False)
    assert m.classify("https://pds.example/x", None) == "bsky"
    assert m.classify("https://other.example/x", None) == "mastodon"

    # defaults survive the merge rather than being replaced
    assert m.classify("https://bsky.app/profile/a/post/1", None) == "bsky"


def test_bad_social_domain_override_is_rejected():
    with pytest.raises(ValueError):
        PlatformMatcher.build(["justadomain.example"], env={})


# ── 8. deduplication ─────────────────────────────────────────────────────


def test_same_image_reached_from_two_pages_is_fetched_once():
    results = [
        ReverseImageResult("p", "https://cdn.bsky.app/img/same.jpg", "https://bsky.app/profile/a/post/1", "full"),
        ReverseImageResult("p", "https://cdn.bsky.app/img/same.jpg", "https://bsky.app/profile/b/post/2", "partial"),
    ]
    report = filter_social(results)
    assert len(report.candidates) == 1
    assert [e["reason"] for e in report.skipped] == ["duplicate"]


# ── 9. candidate downloading (reuses the existing bounded fetch) ─────────


async def test_reverse_image_candidates_use_the_existing_bounded_fetch():
    """No second download path: the same 512KB range-capped fetch, so the
    size limit and truncation guard apply to reverse-image candidates too."""
    from faceanchor.search.fanout import CANDIDATE_BYTE_CAP, fetch_all_candidates

    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers[str(request.url)] = request.headers.get("range")
        if "oversized" in str(request.url):
            return httpx.Response(
                206,
                headers={"content-range": f"bytes 0-{CANDIDATE_BYTE_CAP-1}/{CANDIDATE_BYTE_CAP*4}"},
                content=b"x" * 32,
            )
        if "gone" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, content=PROBE.read_bytes())

    report = filter_social(
        [
            ReverseImageResult("p", "https://cdn.bsky.app/img/ok.jpg", "https://bsky.app/profile/a/post/1", "full"),
            ReverseImageResult("p", "https://cdn.bsky.app/img/oversized.jpg", "https://bsky.app/profile/a/post/2", "full"),
            ReverseImageResult("p", "https://cdn.bsky.app/img/gone.jpg", "https://bsky.app/profile/a/post/3", "full"),
        ]
    )
    trace: list[dict] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetched = await fetch_all_candidates(client, report.candidates, trace=trace)

    assert [c.image_url for c, _ in fetched] == ["https://cdn.bsky.app/img/ok.jpg"]
    assert all(h == f"bytes=0-{CANDIDATE_BYTE_CAP - 1}" for h in seen_headers.values())
    outcomes = {r["image_url"].rsplit("/", 1)[-1]: r["fetch"] for r in trace}
    assert outcomes == {"ok.jpg": "ok", "oversized.jpg": "truncated", "gone.jpg": "http_error"}


# ── 10. face verification is what turns a provider hit into a match ──────


def _probe_context():
    from faceanchor.search import score
    from faceanchor.vision import detect, quality
    from faceanchor.vision.decode import decode_jpeg_bytes

    raw = PROBE.read_bytes()
    image = decode_jpeg_bytes(raw)
    face, err = quality.check_single_face(detect.detect_faces(image), None)
    assert err is None
    return score.build_probe_context(image, face), raw


def test_a_provider_hit_is_not_a_match_until_a_face_verifies():
    """The acceptance criterion: the reverse-image provider proposes, SFace
    disposes. A 'full' match on an image with no matching face is dropped."""
    from faceanchor.search import score

    probe_ctx, raw = _probe_context()
    report = filter_social(
        [
            # the undecodable one first: an exact re-upload short-circuits the
            # cascade, so a candidate ordered after it would never be traced
            ReverseImageResult("p", "https://cdn.bsky.app/img/noface.jpg", "https://bsky.app/profile/a/post/2", "full"),
            ReverseImageResult("p", "https://cdn.bsky.app/img/hit.jpg", "https://bsky.app/profile/a/post/1", "full"),
        ]
    )
    by_url = {"https://cdn.bsky.app/img/hit.jpg": raw, "https://cdn.bsky.app/img/noface.jpg": b"not an image"}
    fetched = [(c, by_url[c.image_url]) for c in report.candidates]

    trace: list[dict] = []
    matches = score.score_candidates(probe_ctx, fetched, trace=trace)

    assert [m.candidate.post_uri for m in matches] == ["https://bsky.app/profile/a/post/1"]
    assert {r["image_url"]: r["verdict"] for r in trace}["https://cdn.bsky.app/img/noface.jpg"] == "undecodable"
    assert matches[0].score >= score.COSINE_ACCEPT_THRESHOLD or matches[0].metric == "phash"


def test_face_verification_uses_the_existing_threshold_policy():
    """No separate reverse-image threshold: the same SFace policy decides."""
    from faceanchor.search import score

    assert score.COSINE_ACCEPT_THRESHOLD == 0.363
    probe_ctx, raw = _probe_context()
    matches = score.score_candidates(
        probe_ctx,
        [(Candidate(platform="bsky", image_url="https://cdn.bsky.app/i.jpg", post_uri="https://bsky.app/p/1",
                    extra={"discovery": "web_reverse", "match_type": "full", "provider": "google_web_detection"}), raw)],
    )
    assert matches
    assert matches[0].metric in ("phash", "cosine")


# ── 11/12/13/14. evidence provenance, canonicalization, anchor, verify ───


def _web_reverse_bundle(**overrides) -> dict:
    probe = ProbeInfo(
        image_sha256="a" * 64,
        image_phash="deadbeefcafebabe",
        embed_model="sface_2021dec",
        embed_sha256="b" * 64,
        consent_digest="c" * 64,
    )
    candidate = Candidate(
        platform="bsky",
        image_url="https://cdn.bsky.app/img/feed_fullsize/plain/did:plc:abc/full.jpg",
        post_uri="https://bsky.app/profile/alice.bsky.social/post/3kabcdef",
        extra={"discovery": "web_reverse", "provider": "google_web_detection", "match_type": "full"},
    )
    match = ScoredMatch(candidate=candidate, score=0.4812, metric="cosine", image_bytes_sha256="d" * 64)
    return build_bundle(
        pipeline_commit="dev",
        probe=probe,
        match=match,
        match_text_sha256=hashlib.sha256(b"").hexdigest(),
        match_image_url_sha256=hashlib.sha256(candidate.image_url.encode()).hexdigest(),
        run_id="01JFIXEDRUNID000000000000",
        observed_at="2026-09-02T00:00:00Z",
        discovery={
            "mode": "web_reverse",
            "provider": "google_web_detection",
            "provider_results": 7,
            "social_candidates": 3,
            "platforms": ["bsky", "mastodon"],
        },
        match_extra={
            "reverse_image": {
                "provider": "google_web_detection",
                "match_type": "full",
                "page_url": candidate.post_uri,
                "image_url": candidate.image_url,
            }
        },
        **overrides,
    )


def test_bundle_records_every_field_a_later_reader_needs():
    bundle = _web_reverse_bundle()
    assert bundle["run"]["discovery"] == {
        "mode": "web_reverse",
        "provider": "google_web_detection",
        "provider_results": 7,
        "social_candidates": 3,
        "platforms": ["bsky", "mastodon"],
    }
    assert bundle["probe"]["image_sha256"] == "a" * 64          # probe image hash
    assert bundle["probe"]["embed_model"] == "sface_2021dec"    # face encoder
    assert bundle["probe"]["consent_digest"] == "c" * 64        # consent
    assert bundle["match"]["platform"] == "bsky"
    assert bundle["match"]["uri"].startswith("https://bsky.app/")           # candidate page URL
    assert bundle["match"]["image_sha256"] == "d" * 64                       # candidate image hash
    assert bundle["match"]["reverse_image"]["image_url"].startswith("https://cdn.bsky.app/")
    assert bundle["match"]["reverse_image"]["match_type"] == "full"
    assert bundle["score"]["value"] == 0.4812                                # similarity
    assert bundle["score"]["threshold"] == 0.363                             # threshold
    assert bundle["run"]["observed_at"] == "2026-09-02T00:00:00Z"            # timestamp


def test_bundle_still_carries_no_biometric_material():
    blob = json.dumps(_web_reverse_bundle()).lower()
    assert "embedding" not in blob
    assert "vector" not in blob
    assert "base64" not in blob


def test_match_extra_is_additive_and_leaves_old_bundles_byte_identical():
    """The schema change must not move an existing bundle's digest."""
    probe = ProbeInfo(image_sha256="a"*64, image_phash="d"*16, embed_model="sface_2021dec",
                      embed_sha256="b"*64, consent_digest="c"*64)
    match = ScoredMatch(
        candidate=Candidate(platform="commons", image_url="https://x/y.jpg", post_uri="https://x/p"),
        score=0.5, metric="cosine", image_bytes_sha256="d"*64,
    )
    kwargs = dict(pipeline_commit="dev", probe=probe, match=match, match_text_sha256="e"*64,
                  match_image_url_sha256="f"*64, run_id="01JFIXEDRUNID000000000000",
                  observed_at="2026-09-01T00:00:00Z")
    assert canonicalize(build_bundle(**kwargs)) == canonicalize(build_bundle(**kwargs, match_extra=None))
    assert canonicalize(build_bundle(**kwargs)) == canonicalize(build_bundle(**kwargs, match_extra={}))
    assert canonicalize(build_bundle(**kwargs)) != canonicalize(
        build_bundle(**kwargs, match_extra={"reverse_image": {"provider": "google_web_detection"}})
    )


def test_canonicalization_is_deterministic_for_a_web_reverse_bundle():
    bundle = _web_reverse_bundle()
    assert canonicalize(bundle) == canonicalize(json.loads(json.dumps(bundle)))
    assert len(bundle_digest(bundle)) == 32


async def test_anchor_then_verify_round_trips_through_the_existing_chain_path():
    """Reuses the real ChainAdapter interface — no second anchoring system."""
    from faceanchor.chain.base import AnchorReceipt, ChainAdapter, VerifyResult

    class InMemoryChain(ChainAdapter):
        def __init__(self):
            self.state: dict[bytes, int] = {}

        async def anchor(self, digest: bytes, cid: str) -> AnchorReceipt:
            assert len(digest) == 32
            self.state[digest] = 1788262102
            return AnchorReceipt(tx_hash="0xdeadbeef", block_timestamp=1788262102)

        async def verify(self, digest: bytes) -> VerifyResult:
            return VerifyResult(ok=digest in self.state, timestamp=self.state.get(digest, 0))

        async def wait_for_inclusion(self, receipt: AnchorReceipt) -> None:
            return None

    chain = InMemoryChain()
    bundle = _web_reverse_bundle()
    digest = bundle_digest(bundle)

    receipt = await chain.anchor(digest, "bafkreitest")
    await chain.wait_for_inclusion(receipt)
    assert (await chain.verify(bundle_digest(bundle))).ok is True

    # nothing but the 32-byte commitment ever reached the chain
    assert list(chain.state) == [digest]
    return chain


# ── 15-18. tamper detection on every reverse-image field ─────────────────


def _tamper(mutate) -> tuple[bytes, bytes]:
    bundle = _web_reverse_bundle()
    before = bundle_digest(bundle)
    tampered = copy.deepcopy(bundle)
    mutate(tampered)
    return before, bundle_digest(tampered)


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("candidate page url (match.uri)", lambda b: b["match"].__setitem__("uri", "https://bsky.app/profile/mallory.bsky.social/post/9")),
        ("candidate page url (reverse_image)", lambda b: b["match"]["reverse_image"].__setitem__("page_url", "https://bsky.app/profile/mallory.bsky.social/post/9")),
        ("candidate image url", lambda b: b["match"]["reverse_image"].__setitem__("image_url", "https://cdn.bsky.app/img/other.jpg")),
        ("candidate image hash", lambda b: b["match"].__setitem__("image_sha256", "9" * 64)),
        ("candidate image url hash", lambda b: b["match"].__setitem__("image_url_sha256", "9" * 64)),
        ("similarity", lambda b: b["score"].__setitem__("value", 0.9999)),
        ("threshold", lambda b: b["score"].__setitem__("threshold", 0.1)),
        ("provider (match block)", lambda b: b["match"]["reverse_image"].__setitem__("provider", "some_other_provider")),
        ("provider (discovery block)", lambda b: b["run"]["discovery"].__setitem__("provider", "mock_replay")),
        ("discovery mode", lambda b: b["run"]["discovery"].__setitem__("mode", "by_face")),
        ("match type", lambda b: b["match"]["reverse_image"].__setitem__("match_type", "similar")),
        ("platform", lambda b: b["match"].__setitem__("platform", "mastodon")),
        ("consent digest", lambda b: b["probe"].__setitem__("consent_digest", "9" * 64)),
        ("probe image hash", lambda b: b["probe"].__setitem__("image_sha256", "9" * 64)),
        ("face encoder", lambda b: b["probe"].__setitem__("embed_model", "some_other_model")),
    ],
)
def test_tampering_with_any_recorded_field_breaks_the_anchor(name, mutate):
    before, after = _tamper(mutate)
    assert before != after, f"tampering with {name} did not change the digest"


async def test_a_tampered_bundle_fails_verification_against_the_chain():
    chain = await test_anchor_then_verify_round_trips_through_the_existing_chain_path()
    tampered = _web_reverse_bundle()
    tampered["match"]["reverse_image"]["image_url"] = "https://cdn.bsky.app/img/swapped.jpg"
    assert (await chain.verify(bundle_digest(tampered))).ok is False


# ── 19/20. NO_RESULTS vs PROVIDER_ERROR ──────────────────────────────────


async def test_no_results_returns_empty_and_raises_nothing():
    provider, client = _provider_with(lambda request: httpx.Response(200, json={"responses": [{}]}))
    async with client:
        assert await provider.search(PROBE, client=client) == []


async def test_provider_error_never_looks_like_no_results():
    """The distinction the whole error hierarchy exists for."""
    failures = [
        (lambda request: httpx.Response(500, text="boom"), ProviderHTTPError),
        (lambda request: httpx.Response(403, text="denied"), ProviderAuthError),
        (lambda request: httpx.Response(429, text="slow down"), ProviderRateLimitError),
        (lambda request: httpx.Response(200, content=b"not json"), MalformedResponseError),
    ]
    for handler, expected in failures:
        provider, client = _provider_with(handler)
        async with client:
            with pytest.raises(expected):
                await provider.search(PROBE, client=client)


def test_cli_uses_distinct_exit_codes_for_the_two_outcomes():
    from faceanchor.cli import EXIT_NO_RESULTS, EXIT_PROVIDER_ERROR

    assert EXIT_NO_RESULTS != EXIT_PROVIDER_ERROR


def test_no_social_candidates_is_reported_separately_from_no_results():
    """34 web hits and none social is a different fact from 0 web hits."""
    report = filter_social([ReverseImageResult("p", "https://news.example.com/a.jpg", "https://news.example.com/a", "full")])
    assert report.candidates == []
    assert report.skipped[0]["reason"] == "not_social"
    assert report.summary() == "none"


# ── 21/22/23. the existing arms are untouched ────────────────────────────


def test_existing_discovery_flags_still_resolve_the_way_they_did():
    import typer

    from faceanchor.cli import _resolve_discovery_mode

    assert _resolve_discovery_mode(None, "some words", False) == "text_seeded"
    assert _resolve_discovery_mode(None, None, True) == "by_face"

    # the original mutual exclusivity is unchanged
    for query, by_face in [(None, False), ("words", True)]:
        with pytest.raises(typer.BadParameter):
            _resolve_discovery_mode(None, query, by_face)


def test_discovery_flag_names_the_same_three_arms():
    import typer

    from faceanchor.cli import _resolve_discovery_mode

    assert _resolve_discovery_mode("web-reverse", None, False) == "web_reverse"
    assert _resolve_discovery_mode("web_reverse", None, False) == "web_reverse"
    assert _resolve_discovery_mode("text", "words", False) == "text_seeded"
    assert _resolve_discovery_mode("by-face", None, True) == "by_face"
    assert _resolve_discovery_mode("by-face", None, False) == "by_face"

    with pytest.raises(typer.BadParameter):
        _resolve_discovery_mode("web-reverse", "words", False)   # not the same operation
    with pytest.raises(typer.BadParameter):
        _resolve_discovery_mode("web-reverse", None, True)       # nor this one
    with pytest.raises(typer.BadParameter):
        _resolve_discovery_mode("text", None, False)             # text needs a query
    with pytest.raises(typer.BadParameter):
        _resolve_discovery_mode("nonsense", None, True)


def test_consent_gate_still_blocks_every_discovery_mode():
    from faceanchor.consent import MissingConsentError, load_consent_digest

    with pytest.raises(MissingConsentError):
        load_consent_digest(None)
    with pytest.raises(MissingConsentError):
        load_consent_digest("/nonexistent/consent.json")


def test_web_reverse_run_requires_consent_before_anything_is_uploaded():
    """Ordering matters: the consent gate is upstream of Stage 1, which is
    upstream of the upload notice."""
    from typer.testing import CliRunner

    from faceanchor.cli import app

    result = CliRunner().invoke(
        app,
        ["run", "--probe", str(PROBE), "--discovery", "web-reverse", "--provider", "google",
         "--accept-external-upload", "--chain", "anvil",
         "--contract-address", "0x0000000000000000000000000000000000000000"],
    )
    assert result.exit_code != 0
    assert "subject-consent is required" in result.output
    assert "NOTICE" not in result.output   # nothing was ever offered for upload


def test_web_reverse_refuses_to_upload_without_acknowledgement(monkeypatch, capsys):
    """Non-interactive and un-acknowledged: the probe must not leave."""
    import typer

    import faceanchor.cli as cli_mod

    monkeypatch.setattr("sys.stdin", type("NoTTY", (), {"isatty": staticmethod(lambda: False)})())
    with pytest.raises(typer.Exit) as e:
        cli_mod._confirm_external_upload("google_web_detection", accepted=False)
    assert e.value.exit_code == 1
    assert "refusing to upload" in capsys.readouterr().err


def test_upload_notice_names_the_trust_boundary(capsys):
    import faceanchor.cli as cli_mod

    cli_mod._confirm_external_upload("google_web_detection", accepted=True)
    out = capsys.readouterr().out
    assert "google_web_detection" in out
    assert "PROBE IMAGE" in out
    assert "trust boundary" in out


# ── opt-in live integration test ─────────────────────────────────────────


@pytest.mark.skipif(
    not (__import__("os").environ.get("FACEANCHOR_LIVE_VISION")
         and (__import__("os").environ.get("GOOGLE_VISION_API_KEY")
              or __import__("os").environ.get("GOOGLE_APPLICATION_CREDENTIALS"))),
    reason="live Google Vision test — set FACEANCHOR_LIVE_VISION=1 plus real credentials to run (costs money)",
)
async def test_live_google_web_detection():
    """The only test that spends money and leaves the machine. Opt-in twice
    over: the credential AND the explicit flag, so it can never run by
    accident in CI."""
    provider = GoogleWebDetectionProvider()
    async with httpx.AsyncClient(http2=True) as client:
        results = await provider.search(PROBE, client=client)
    assert isinstance(results, list)
    for r in results:
        assert r.provider == "google_web_detection"
        assert r.image_url or r.page_url
