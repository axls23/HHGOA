"""Arm A · Bluesky — unauthenticated public AppView (PRD §5.2).

`public.api.bsky.app` needs no auth and no anti-bot friction under normal
conditions; AT Protocol records already carry a content-addressed CID, which
composes directly with the evidence bundle (PRD §6.3/§7.2).

Note: this endpoint was unreachable from the sandbox this was built in (403
at Bluesky's CDN edge, geo/datacenter filtering — Mastodon and Wikimedia
Commons were reachable from the same host). The client below is written
against the documented XRPC shapes and unit-tested against a captured
response; it has not been exercised against a live response. Verify against
a real network before the demo.
"""

from __future__ import annotations

import httpx

from faceanchor.search.models import Candidate

BASE_URL = "https://public.api.bsky.app"


def _extract_image_candidates(post: dict) -> list[Candidate]:
    """A `searchPosts` result item -> zero or more image Candidates.

    Embedded images live at `post.embed.images[].fullsize` (app.bsky.embed.images#view)
    or, for quote/record-with-media embeds, `post.embed.media.images[]`.
    """
    author = post.get("author", {})
    author_handle = author.get("handle")
    uri = post.get("uri", "")
    cid = post.get("cid")
    text = (post.get("record") or {}).get("text")

    embed = post.get("embed") or {}
    images = embed.get("images")
    if images is None:
        images = (embed.get("media") or {}).get("images", [])

    candidates = []
    for img in images or []:
        fullsize = img.get("fullsize")
        if not fullsize:
            continue
        candidates.append(
            Candidate(
                platform="bsky",
                image_url=fullsize,
                post_uri=uri,
                author=author_handle,
                text=text,
                record_cid=cid,
                extra={"alt": img.get("alt", "")},
            )
        )
    return candidates


async def search_posts(client: httpx.AsyncClient, query: str, limit: int = 25) -> list[Candidate]:
    resp = await client.get(
        f"{BASE_URL}/xrpc/app.bsky.feed.searchPosts",
        params={"q": query, "limit": limit},
        timeout=5.0,
    )
    resp.raise_for_status()
    data = resp.json()

    candidates: list[Candidate] = []
    for post in data.get("posts", []):
        candidates.extend(_extract_image_candidates(post))
    return candidates
