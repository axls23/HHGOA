"""Arm A · Mastodon — anonymous public API, no key (PRD §5.2).

Three routes, because the obvious one is a dead end. Measured against
mastodon.social, unauthenticated:

- `/api/v2/search?type=statuses` -> HTTP 200 with `statuses: []`, always.
  Mastodon gates full-text status search behind a token; an anonymous caller
  gets a well-formed empty result, not an error. Kept because it is correct
  the moment a token is present, but it contributes nothing on its own.
- `/api/v1/accounts/lookup` + `/api/v1/accounts/:id/statuses?only_media=true`
  -> 200, 20 statuses, 23 images. This is the route that matters: it turns a
  query that *is a handle* into that account's own images, plus the avatar.
- `/api/v1/timelines/tag/:tag?only_media=true` -> 200, 10 statuses, 10 images.
  Public hashtag timelines are open to anonymous callers.

So the query shape decides the route: a handle looks up an account, a single
word reads a hashtag timeline, and a multi-word phrase has no anonymous
route at all (v2 search would serve it, if you had a token).
"""

from __future__ import annotations

import asyncio
import re

import httpx

from faceanchor.search.models import Candidate
from faceanchor.search.safety import SafetyReport, mastodon_status_is_sensitive

DEFAULT_INSTANCES = ["mastodon.social"]

# "@user@instance" | "@user" | "user" — a bare word is ambiguous with a hashtag,
# so it is tried as both (each route 404s harmlessly when it doesn't apply).
_HANDLE_RE = re.compile(r"^@?([A-Za-z0-9_]{1,30})(?:@([A-Za-z0-9.\-]+\.[A-Za-z]{2,}))?$")


def _parse_handle(query: str) -> tuple[str, str | None] | None:
    m = _HANDLE_RE.match(query.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def _extract_image_candidates(
    status: dict, *, allow_sensitive: bool = False, report: SafetyReport | None = None
) -> list[Candidate]:
    """Statuses the poster flagged as sensitive contribute nothing by default.

    Dropped here rather than after download, so an explicit post costs no
    request and its URL never reaches the candidate log or a contact sheet
    (`search/safety.py` has the reasoning and the limits).
    """
    if not allow_sensitive and mastodon_status_is_sensitive(status):
        if report is not None:
            report.note_sensitive()
        return []

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
                extra={
                    "description": media.get("description") or "",
                    # Mastodon's own bounded variant (meta.small, typically
                    # ~640px). Corpus scanning prefers it: the originals run to
                    # multiple MB and the 512KB Range cap truncates them
                    # mid-file, which decodes to garbage or not at all.
                    "preview_url": media.get("preview_url") or "",
                },
            )
        )
    return candidates


def _avatar_candidate(account: dict) -> list[Candidate]:
    """An account's own avatar — usually the one image of a person that a
    handle reliably points at, which is exactly what the webcam demo needs.
    """
    avatar = account.get("avatar")
    if not avatar:
        return []
    return [
        Candidate(
            platform="mastodon",
            image_url=avatar,
            post_uri=account.get("url", ""),
            author=account.get("acct"),
            text=account.get("note") or "",
            extra={"kind": "avatar"},
        )
    ]


async def _get_json(client: httpx.AsyncClient, url: str, params: dict, timeout: float = 5.0):
    resp = await client.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


async def _search_statuses(
    client: httpx.AsyncClient, instance: str, query: str, limit: int, allow_sensitive: bool, safety: SafetyReport
) -> list[Candidate]:
    data = await _get_json(
        client, f"https://{instance}/api/v2/search", {"q": query, "type": "statuses", "limit": limit}
    )
    return [
        c
        for s in data.get("statuses", [])
        for c in _extract_image_candidates(s, allow_sensitive=allow_sensitive, report=safety)
    ]


async def _account_media(
    client: httpx.AsyncClient, instance: str, handle: str, limit: int, allow_sensitive: bool, safety: SafetyReport
) -> list[Candidate]:
    account = await _get_json(client, f"https://{instance}/api/v1/accounts/lookup", {"acct": handle})
    account_id = account.get("id")
    if not account_id:
        return []
    statuses = await _get_json(
        client,
        f"https://{instance}/api/v1/accounts/{account_id}/statuses",
        {"limit": limit, "only_media": "true", "exclude_reblogs": "true"},
    )
    return _avatar_candidate(account) + [
        c for s in statuses for c in _extract_image_candidates(s, allow_sensitive=allow_sensitive, report=safety)
    ]


async def _tag_timeline(
    client: httpx.AsyncClient, instance: str, tag: str, limit: int, allow_sensitive: bool, safety: SafetyReport
) -> list[Candidate]:
    statuses = await _get_json(
        client, f"https://{instance}/api/v1/timelines/tag/{tag}", {"limit": limit, "only_media": "true"}
    )
    return [
        c for s in statuses for c in _extract_image_candidates(s, allow_sensitive=allow_sensitive, report=safety)
    ]


async def search_posts(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 20,
    instances: list[str] | None = None,
    allow_sensitive: bool = False,
    safety: SafetyReport | None = None,
) -> list[Candidate]:
    """Fans out every route the query shape allows, across every instance.

    A route that errors, 404s, or is auth-gated contributes zero candidates
    rather than failing the arm — the same degrade-cleanly posture Arm B uses
    under DDG throttling. Duplicate image URLs across routes are collapsed.

    Statuses the poster flagged sensitive are excluded unless
    `allow_sensitive`; `safety`, when given, counts what was removed.
    """
    instances = list(instances or DEFAULT_INSTANCES)
    parsed = _parse_handle(query)
    safety = safety if safety is not None else SafetyReport()

    tasks = []
    for instance in instances:
        tasks.append(_search_statuses(client, instance, query, limit, allow_sensitive, safety))
        if parsed:
            handle, handle_instance = parsed
            # "@user@instance" pins the lookup to that instance; a bare handle
            # is looked up on each configured instance instead.
            target = handle_instance or instance
            tasks.append(
                _account_media(
                    client,
                    target,
                    f"{handle}@{handle_instance}" if handle_instance else handle,
                    limit,
                    allow_sensitive,
                    safety,
                )
            )
            tasks.append(_tag_timeline(client, instance, handle, limit, allow_sensitive, safety))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for r in results:
        if isinstance(r, BaseException):
            continue
        for c in r:
            if c.image_url in seen:
                continue
            seen.add(c.image_url)
            candidates.append(c)
    return candidates
