"""Arm A · Bluesky — unauthenticated public AppView (PRD §5.2).

No auth, no key; AT Protocol records already carry a content-addressed CID,
which composes directly with the evidence bundle (PRD §6.3/§7.2).

Two routes, because `searchPosts` is not dependably available to anonymous
callers. Measured, not assumed:

- `app.bsky.feed.searchPosts` answered 200 with real posts, then began
  returning `403 Request forbidden by administrative rules` for this IP —
  on both `public.api.bsky.app` and `api.bsky.app`, and identically across
  HTTP/1.1 and h2, with a browser UA, our own UA, and no UA at all. So it is
  a per-IP block on that one endpoint after a burst of anonymous search, not
  UA sniffing, not a dead host, and not the geo/datacenter filtering an
  earlier revision of this file blamed. Kept as a best-effort route; when it
  is blocked it is *reported* as blocked rather than silently returning [].
- `app.bsky.actor.getProfile` + `app.bsky.feed.getAuthorFeed` kept answering
  200 throughout (30 posts, 8 images, avatar present) from the same IP in the
  same second that searchPosts was 403ing. So a query that *is a handle*
  still resolves to that account's own images without any credentials.
"""

from __future__ import annotations

import asyncio
import re

import httpx

from faceanchor.search.models import Candidate
from faceanchor.search.safety import SafetyReport, bsky_explicit_labels

BASE_URL = "https://public.api.bsky.app"

# "alice.bsky.social", "@alice.bsky.social", or a raw DID — anything else
# (a bare word, a phrase) is not a resolvable actor and skips the handle route.
_HANDLE_RE = re.compile(r"^@?([A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,}|did:[a-z]+:[A-Za-z0-9._%-]+)$")


def _parse_actor(query: str) -> str | None:
    m = _HANDLE_RE.match(query.strip())
    return m.group(1) if m else None


def _extract_image_candidates(
    post: dict, *, allow_sensitive: bool = False, report: SafetyReport | None = None
) -> list[Candidate]:
    """A `searchPosts` result item -> zero or more image Candidates.

    Embedded images live at `post.embed.images[].fullsize` (app.bsky.embed.images#view)
    or, for quote/record-with-media embeds, `post.embed.media.images[]`.

    Posts carrying an explicit AT Protocol label — on the post, the record's
    self-labels, or the author's account — contribute nothing by default, and
    are dropped before any fetch (`search/safety.py`).
    """
    if not allow_sensitive:
        labels = bsky_explicit_labels(post)
        if labels:
            if report is not None:
                report.note_labelled(labels)
            return []

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
                extra={"alt": img.get("alt", ""), "thumb": img.get("thumb") or ""},
            )
        )
    return candidates


def _avatar_candidate(profile: dict, *, allow_sensitive: bool = False) -> list[Candidate]:
    """The account's own avatar — for a handle query, the one image of a person
    a handle reliably points at (mirrors the Mastodon arm's avatar route).
    """
    avatar = profile.get("avatar")
    if not avatar:
        return []
    # An adult account is usually labelled once, at the account level, rather
    # than on each post — so the avatar needs the same check the posts get.
    if not allow_sensitive and bsky_explicit_labels({"author": profile, "labels": profile.get("labels")}):
        return []
    handle = profile.get("handle")
    return [
        Candidate(
            platform="bsky",
            image_url=avatar,
            post_uri=f"at://{profile.get('did', '')}" if profile.get("did") else (handle or ""),
            author=handle,
            text=profile.get("description") or "",
            extra={"kind": "avatar"},
        )
    ]


async def _get_json(client: httpx.AsyncClient, path: str, params: dict, timeout: float = 5.0) -> dict:
    resp = await client.get(f"{BASE_URL}/xrpc/{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


async def _search_route(
    client: httpx.AsyncClient, query: str, limit: int, allow_sensitive: bool, safety: SafetyReport
) -> list[Candidate]:
    data = await _get_json(client, "app.bsky.feed.searchPosts", {"q": query, "limit": limit})
    return [
        c
        for post in data.get("posts", [])
        for c in _extract_image_candidates(post, allow_sensitive=allow_sensitive, report=safety)
    ]


async def _account_route(
    client: httpx.AsyncClient, actor: str, limit: int, allow_sensitive: bool, safety: SafetyReport
) -> list[Candidate]:
    profile = await _get_json(client, "app.bsky.actor.getProfile", {"actor": actor})
    feed = await _get_json(client, "app.bsky.feed.getAuthorFeed", {"actor": actor, "limit": limit})
    posts = [item.get("post") or {} for item in feed.get("feed", [])]
    return _avatar_candidate(profile, allow_sensitive=allow_sensitive) + [
        c for post in posts for c in _extract_image_candidates(post, allow_sensitive=allow_sensitive, report=safety)
    ]


async def search_posts(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 25,
    allow_sensitive: bool = False,
    safety: SafetyReport | None = None,
) -> list[Candidate]:
    """Both routes, concurrently, deduped by image URL.

    Raises only if *every* applicable route failed — so a 403 on the blocked
    search endpoint still surfaces as a failure when it is the only route the
    query shape allows, and is quietly absorbed when the handle route
    delivered candidates anyway.

    Posts carrying an explicit AT Protocol label are excluded unless
    `allow_sensitive`; `safety`, when given, counts what was removed.
    """
    actor = _parse_actor(query)
    safety = safety if safety is not None else SafetyReport()
    routes = [_search_route(client, query, limit, allow_sensitive, safety)]
    if actor:
        routes.append(_account_route(client, actor, limit, allow_sensitive, safety))

    results = await asyncio.gather(*routes, return_exceptions=True)

    candidates: list[Candidate] = []
    seen: set[str] = set()
    errors: list[BaseException] = []
    for r in results:
        if isinstance(r, BaseException):
            errors.append(r)
            continue
        for c in r:
            if c.image_url in seen:
                continue
            seen.add(c.image_url)
            candidates.append(c)

    if not candidates and errors:
        raise errors[0]
    return candidates
