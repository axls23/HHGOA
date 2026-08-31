"""Arm B · DuckDuckGo — seed-driven candidate generator, NOT reverse-image search (PRD §5.2).

`ddgs.images()` is a text query against DDG's (Bing-backed) image index; there
is no upload/image_url parameter anywhere in the surface. So this arm cannot
bootstrap an identity from a face alone — it corroborates and expands around
a match Arm A already found, seeded from that match's author handle/name/
keywords (or an explicit --seed-query). That capability reduction versus true
reverse-image search is deliberate and sits better with PRD §3's subject
policy: face-to-stranger-identity is exactly the mode it exists to prevent.

Rate limiting, not latency, is the binding constraint (PRD §5.2): DDG
throttles aggressively on bursts, so this arm carries its own ~1req/s limiter
and hard-fails at a deadline rather than retrying into the response path. A
throttled Arm B must never degrade Arm A's result.
"""

from __future__ import annotations

import asyncio
import re

from aiolimiter import AsyncLimiter
from ddgs import DDGS

from faceanchor.search.models import Candidate

ARM_B_DEADLINE_S = 1.5
RATE_LIMIT_PER_SECOND = 1
MAX_RESULTS = 20

_limiter = AsyncLimiter(RATE_LIMIT_PER_SECOND, time_period=1.0)

# Handles/hashtags/URLs stripped out; keeps plain words and @mentions as seeds.
_SEED_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOPWORDS = {"the", "and", "for", "with", "this", "that", "from", "have", "https", "http"}


def extract_seed_terms(author_handle: str | None, text: str | None, max_terms: int = 5) -> str:
    """Deterministic regex parsing — no LLM on the hot path (PRD §5.4:
    the one honest use of an LLM here is optional, behind --seed-llm, and
    off by default).
    """
    terms: list[str] = []
    if author_handle:
        # strip a leading @ and any @domain suffix (e.g. mastodon's user@instance)
        handle = author_handle.lstrip("@").split("@")[0]
        if handle:
            terms.append(handle)

    if text:
        for token in _SEED_TOKEN_RE.findall(text):
            lower = token.lower()
            if lower in _STOPWORDS or lower in terms:
                continue
            terms.append(token)
            if len(terms) >= max_terms:
                break

    return " ".join(terms[:max_terms])


def _image_candidates_from_ddgs_results(results: list[dict]) -> list[Candidate]:
    candidates = []
    for r in results:
        image_url = r.get("image")
        if not image_url:
            continue
        candidates.append(
            Candidate(
                platform="ddg",
                image_url=image_url,
                post_uri=r.get("url", image_url),
                text=r.get("title"),
                extra={"source": r.get("source", "")},
            )
        )
    return candidates


async def search_images(seed_query: str, max_results: int = MAX_RESULTS) -> list[Candidate]:
    """Returns [] on throttling or deadline overrun — never raises into the caller.

    ddgs is a synchronous scraper client (no native asyncio support), so the
    blocking call runs in a thread; the limiter and deadline still bound it
    from the async side.
    """

    async def _call():
        async with _limiter:
            return await asyncio.to_thread(DDGS().images, seed_query, max_results=max_results)

    try:
        results = await asyncio.wait_for(_call(), timeout=ARM_B_DEADLINE_S)
    except Exception:
        return []

    return _image_candidates_from_ddgs_results(results)
