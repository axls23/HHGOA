"""Async fanout across Arm A sources + candidate image fetch (PRD §5.2/§5.3).

HTTP/2, a shared keep-alive pool capped at 64 connections, per-arm timeouts so
one slow source can't stall the others, and a 512KB range-capped fetch per
candidate image to bound tail latency on large media.
"""

from __future__ import annotations

import asyncio
import hashlib

import httpx

from faceanchor.search import bluesky, commons, mastodon
from faceanchor.search.models import Candidate

MAX_CONNECTIONS = 64
CANDIDATE_FETCH_CONCURRENCY = 64
CANDIDATE_BYTE_CAP = 512 * 1024
ARM_A_TIMEOUT_S = 3.0


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        http2=True,
        limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_CONNECTIONS),
        headers={"User-Agent": "faceanchor/0.1 (+https://github.com/axls23/face-anchor)"},
    )


async def arm_a_fanout(client: httpx.AsyncClient, query: str) -> list[Candidate]:
    """Bluesky + Mastodon + Commons, concurrently, each bounded by its own deadline.

    A source that times out or errors contributes zero candidates rather than
    failing the whole fanout — the same degrade-cleanly posture as Arm B.
    """

    async def _bounded(coro):
        try:
            return await asyncio.wait_for(coro, timeout=ARM_A_TIMEOUT_S)
        except Exception:
            return []

    results = await asyncio.gather(
        _bounded(bluesky.search_posts(client, query)),
        _bounded(mastodon.search_posts(client, query)),
        _bounded(commons.search_images(client, query)),
    )
    return [c for group in results for c in group]


async def fetch_candidate_bytes(client: httpx.AsyncClient, candidate: Candidate) -> tuple[Candidate, bytes | None]:
    try:
        resp = await client.get(
            candidate.image_url,
            headers={"Range": f"bytes=0-{CANDIDATE_BYTE_CAP - 1}"},
            timeout=5.0,
            follow_redirects=True,
        )
        if resp.status_code not in (200, 206):
            return candidate, None
        return candidate, resp.content
    except httpx.HTTPError:
        return candidate, None


async def fetch_all_candidates(
    client: httpx.AsyncClient, candidates: list[Candidate]
) -> list[tuple[Candidate, bytes]]:
    sem = asyncio.Semaphore(CANDIDATE_FETCH_CONCURRENCY)

    async def _fetch_one(c: Candidate):
        async with sem:
            return await fetch_candidate_bytes(client, c)

    results = await asyncio.gather(*[_fetch_one(c) for c in candidates])
    return [(c, b) for c, b in results if b is not None]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
