"""Async fanout across Arm A sources + candidate image fetch (PRD §5.2/§5.3).

HTTP/2, a shared keep-alive pool capped at 64 connections, per-arm timeouts so
one slow source can't stall the others, and a 512KB range-capped fetch per
candidate image to bound tail latency on large media.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field

import httpx

from faceanchor.search import bluesky, commons, duckduckgo, mastodon
from faceanchor.search.models import Candidate
from faceanchor.search.safety import SafetyReport

MAX_CONNECTIONS = 64
CANDIDATE_FETCH_CONCURRENCY = 64
CANDIDATE_BYTE_CAP = 512 * 1024
ARM_A_TIMEOUT_S = 3.0
# Arm B carries its own 1.5s deadline internally (duckduckgo.ARM_B_DEADLINE_S);
# this is the outer belt-and-braces bound, so a hang there can't stall Stage 2.
ARM_B_TIMEOUT_S = 3.0


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        http2=True,
        limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_CONNECTIONS),
        headers={"User-Agent": "faceanchor/0.1 (+https://github.com/axls23/face-anchor)"},
    )


@dataclass(frozen=True)
class SourceReport:
    """Why a source contributed what it did.

    Degrading cleanly is right; degrading *silently* is not. Every source here
    can return zero candidates for three very different reasons — it was
    blocked (Bluesky's 403), it answered honestly with nothing (Mastodon's
    auth-gated search returns HTTP 200 and an empty list), or the query simply
    had no hits. Collapsing all three into `[]` is what makes a half-dead
    fanout look like a working one, so each is recorded and surfaced.
    """

    name: str
    count: int
    elapsed_ms: float
    error: str | None = None

    def describe(self) -> str:
        if self.error:
            return f"{self.name} FAILED ({self.error})"
        return f"{self.name} {self.count}"


@dataclass
class FanoutResult:
    candidates: list[Candidate] = field(default_factory=list)
    reports: list[SourceReport] = field(default_factory=list)
    #: What the explicit-content filter removed before anything was fetched.
    safety: SafetyReport = field(default_factory=SafetyReport)

    def summary(self) -> str:
        return ", ".join(r.describe() for r in self.reports)

    @property
    def degraded(self) -> bool:
        return any(r.error for r in self.reports)


async def _run_source(name: str, coro, timeout: float) -> tuple[list[Candidate], SourceReport]:
    started = time.monotonic()
    try:
        out = await asyncio.wait_for(coro, timeout=timeout)
        return out, SourceReport(name, len(out), (time.monotonic() - started) * 1000)
    except asyncio.TimeoutError:
        return [], SourceReport(name, 0, (time.monotonic() - started) * 1000, f"timeout >{timeout}s")
    except httpx.HTTPStatusError as e:
        return [], SourceReport(name, 0, (time.monotonic() - started) * 1000, f"HTTP {e.response.status_code}")
    except Exception as e:
        return [], SourceReport(name, 0, (time.monotonic() - started) * 1000, type(e).__name__)


async def arm_a_fanout(
    client: httpx.AsyncClient, query: str, allow_sensitive: bool = False
) -> FanoutResult:
    """Bluesky + Mastodon + Commons, concurrently, each bounded by its own deadline.

    A source that times out or errors contributes zero candidates rather than
    failing the whole fanout — the same degrade-cleanly posture as Arm B — but
    it says so in its SourceReport.

    Explicit posts are dropped at extraction on both social sources (see
    `search/safety.py`) and counted in `FanoutResult.safety`. Commons needs no
    such filter: it is a curated, moderated media library, not a public feed.
    """
    safety = SafetyReport()
    pairs = await asyncio.gather(
        _run_source(
            "bsky",
            bluesky.search_posts(client, query, allow_sensitive=allow_sensitive, safety=safety),
            ARM_A_TIMEOUT_S,
        ),
        _run_source(
            "mastodon",
            mastodon.search_posts(client, query, allow_sensitive=allow_sensitive, safety=safety),
            ARM_A_TIMEOUT_S,
        ),
        _run_source("commons", commons.search_images(client, query), ARM_A_TIMEOUT_S),
    )
    return FanoutResult(
        candidates=[c for group, _ in pairs for c in group],
        reports=[report for _, report in pairs],
        safety=safety,
    )


async def full_fanout(
    client: httpx.AsyncClient, query: str, seed_query: str | None = None, allow_sensitive: bool = False
) -> FanoutResult:
    """Arm A + Arm B (PRD §5.2 cascade diagram).

    With an explicit `seed_query`, Arm B fires concurrently with Arm A (both
    are independent of Arm A's results in that case). Without one, Arm B is
    seeded from Arm A's first hit's author/text — it corroborates a match
    Arm A already found rather than bootstrapping from the face alone (the
    deliberate capability reduction documented in duckduckgo.py).

    Either way Arm B is bounded by its own 1.5s deadline/rate limiter and a
    throttled or empty Arm B never affects the Arm A result — it only adds.
    """
    if seed_query:
        arm_a_task = asyncio.ensure_future(arm_a_fanout(client, query, allow_sensitive))
        arm_b_task = asyncio.ensure_future(
            _run_source(f"ddg[{seed_query}]", duckduckgo.search_images(seed_query), ARM_B_TIMEOUT_S)
        )
        result, (arm_b_candidates, arm_b_report) = await asyncio.gather(arm_a_task, arm_b_task)
        result.candidates += arm_b_candidates
        result.reports.append(arm_b_report)
        return result

    result = await arm_a_fanout(client, query, allow_sensitive)
    if not result.candidates:
        result.reports.append(SourceReport("ddg", 0, 0.0, "not seeded — Arm A found nothing to corroborate"))
        return result

    seed = duckduckgo.extract_seed_terms(result.candidates[0].author, result.candidates[0].text)
    if not seed:
        result.reports.append(SourceReport("ddg", 0, 0.0, "not seeded — Arm A's best hit had no usable terms"))
        return result

    arm_b_candidates, arm_b_report = await _run_source(
        f"ddg[{seed}]", duckduckgo.search_images(seed), ARM_B_TIMEOUT_S
    )
    result.candidates += arm_b_candidates
    result.reports.append(arm_b_report)
    return result


def _is_truncated(resp: httpx.Response) -> bool:
    """True when the cap cut the file short, per the server's own accounting.

    A 206 carries `Content-Range: bytes 0-524287/3145728` — if the total
    exceeds what we asked for, we are holding a fragment. Decoding a fragment
    is worse than skipping it: libjpeg hands back a corrupted bottom half
    (which can still detect a bogus "face") and libpng fails loudly. So a
    known-partial image is dropped rather than scored.
    """
    content_range = resp.headers.get("content-range", "")
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[-1].strip()
        if total.isdigit() and int(total) > CANDIDATE_BYTE_CAP:
            return True
    return len(resp.content) >= CANDIDATE_BYTE_CAP


async def fetch_candidate_bytes(
    client: httpx.AsyncClient, candidate: Candidate, trace: list[dict] | None = None
) -> tuple[Candidate, bytes | None]:
    """`trace`, when given, gets one record per candidate describing what the
    fetch actually did — for development, where "why was this URL not in the
    results?" is the question and a silently dropped candidate is the answer
    you can't see.
    """

    def _note(**fields):
        if trace is not None:
            trace.append(
                {
                    "platform": candidate.platform,
                    "image_url": candidate.image_url,
                    "post_uri": candidate.post_uri,
                    "author": candidate.author,
                    "variant": candidate.extra.get("variant", "original"),
                    **fields,
                }
            )

    try:
        resp = await client.get(
            candidate.image_url,
            headers={"Range": f"bytes=0-{CANDIDATE_BYTE_CAP - 1}"},
            timeout=5.0,
            follow_redirects=True,
        )
        if resp.status_code not in (200, 206):
            _note(fetch="http_error", status=resp.status_code, bytes=0)
            return candidate, None
        if _is_truncated(resp):
            _note(fetch="truncated", status=resp.status_code, bytes=len(resp.content))
            return candidate, None
        _note(fetch="ok", status=resp.status_code, bytes=len(resp.content))
        return candidate, resp.content
    except httpx.HTTPError as e:
        _note(fetch="error", status=None, bytes=0, detail=type(e).__name__)
        return candidate, None


async def fetch_all_candidates(
    client: httpx.AsyncClient, candidates: list[Candidate], trace: list[dict] | None = None
) -> list[tuple[Candidate, bytes]]:
    sem = asyncio.Semaphore(CANDIDATE_FETCH_CONCURRENCY)

    async def _fetch_one(c: Candidate):
        async with sem:
            return await fetch_candidate_bytes(client, c, trace=trace)

    results = await asyncio.gather(*[_fetch_one(c) for c in candidates])
    return [(c, b) for c, b in results if b is not None]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
