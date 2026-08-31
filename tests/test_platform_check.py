import hashlib

import httpx
import pytest

from faceanchor.evidence.platform_check import (
    MutationVerdict,
    _parse_at_uri,
    check_bsky_record,
    check_platform_mutation,
)


def _bundle(record_cid: str, text: str, platform: str = "bsky") -> dict:
    return {
        "match": {
            "platform": platform,
            "uri": "at://did:plc:abc123/app.bsky.feed.post/xyz789",
            "record_cid": record_cid,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    }


def test_parse_at_uri():
    did, collection, rkey = _parse_at_uri("at://did:plc:abc123/app.bsky.feed.post/xyz789")
    assert did == "did:plc:abc123"
    assert collection == "app.bsky.feed.post"
    assert rkey == "xyz789"


@pytest.mark.asyncio
async def test_digest_mismatch_short_circuits_to_our_evidence_altered():
    bundle = _bundle("bafyreiabc", "hello")
    async with httpx.AsyncClient() as client:
        result = await check_platform_mutation(client, bundle, digest_verified=False)
    assert result.verdict == MutationVerdict.OUR_EVIDENCE_ALTERED


@pytest.mark.asyncio
async def test_non_bsky_platform_is_not_applicable():
    bundle = _bundle("", "hello", platform="commons")
    async with httpx.AsyncClient() as client:
        result = await check_platform_mutation(client, bundle, digest_verified=True)
    assert result.verdict == MutationVerdict.NOT_APPLICABLE


@pytest.mark.asyncio
async def test_record_intact_when_cid_and_text_match():
    bundle = _bundle("bafyreiabc", "hello world")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"cid": "bafyreiabc", "value": {"text": "hello world"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await check_bsky_record(client, bundle)
    assert result.verdict == MutationVerdict.RECORD_INTACT


@pytest.mark.asyncio
async def test_platform_content_changed_when_cid_diverges():
    bundle = _bundle("bafyreiORIGINAL", "hello world")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"cid": "bafyreiCHANGED", "value": {"text": "edited text"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await check_bsky_record(client, bundle)
    assert result.verdict == MutationVerdict.PLATFORM_CONTENT_CHANGED


@pytest.mark.asyncio
async def test_post_deleted_when_record_not_found():
    bundle = _bundle("bafyreiabc", "hello")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "RecordNotFound"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await check_bsky_record(client, bundle)
    assert result.verdict == MutationVerdict.POST_DELETED
