"""Arm A · Mastodon — public `/api/v2/search`, no key (PRD §5.2).

Not every instance keeps unauthenticated status search open (some require a
bearer token, having locked it down against scraping/abuse — mastodon.social
itself returns empty `statuses` for anonymous requests, confirmed against the
live instance while building this). The client fans out across a
configurable instance list and treats a per-instance empty/auth-blocked
result as "no candidates from this instance", not a hard failure — the same
degrade-cleanly posture Arm B uses under DDG throttling.
"""

from __future__ import annotations

import httpx

from faceanchor.search.models import Candidate

DEFAULT_INSTANCES = ["mastodon.social"]


def _extract_image_candidates(status: dict) -> list[Candidate]:
    account = status.get("account", {})
    author = account.get("acct")
    url = status.get("url", "")
    text = status.get("content", "")

    candidates = []
    for media in status.get("media_attachments", []):
        if media.get("type") != "image":
            continue
        image_url = media.get("url")
        if not image_url:
            continue
        candidates.append(
            Candidate(
                platform="mastodon",
                image_url=image_url,
                post_uri=url,
                author=author,
                text=text,
                extra={"description": media.get("description") or ""},
            )
        )
    return candidates


async def search_posts(
    client: httpx.AsyncClient, query: str, limit: int = 20, instances: list[str] | None = None
) -> list[Candidate]:
    instances = instances or DEFAULT_INSTANCES
    candidates: list[Candidate] = []

    for instance in instances:
        try:
            resp = await client.get(
                f"https://{instance}/api/v2/search",
                params={"q": query, "type": "statuses", "limit": limit},
                timeout=5.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            continue

        data = resp.json()
        for status in data.get("statuses", []):
            candidates.extend(_extract_image_candidates(status))

    return candidates
