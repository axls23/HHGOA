"""Corpus ingestion for face-embedding (by-face) discovery — PRD §5.2's
cascade, seeded by a corpus instead of by words about the subject.

This is **not** reverse-image search, and calling it that would overstate what
it does. It never submits the image anywhere: a corpus spec names *where* to
look (a hashtag timeline, a feed, an account), the pipeline embeds every face
it finds there, and the probe's own embedding is the only query. The face is
the query, and the search space is whatever you named.

Genuine reverse-image search — submitting the image itself to a provider that
indexes the open web by image content — is a separate arm, in
`search/web_reverse/`, reached with `--discovery web-reverse`. The two answer
different questions: this one asks "is this face in the corpus I named?", that
one asks "which pages on the web carry this image?".

Routes are the ones measured to work without credentials (see bluesky.py and
mastodon.py for the audit): Mastodon hashtag timelines paginate via `max_id`
(40 statuses/page, ~40 images/page, tested to 5 pages deep on one tag), and
Bluesky's feed generators serve ~50 items/call while `searchPosts` stays
IP-blocked.

Nothing ingested is written to disk. The candidate list lives for one run and
dies with the process — PRD §3's "no persistent embedding store" is what
separates this from the untargeted facial-recognition database that EU AI Act
Art. 5(1)(e) prohibits, so it is a load-bearing property, not an optimisation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import httpx

from faceanchor.search import bluesky, mastodon
from faceanchor.search.models import Candidate
from faceanchor.search.safety import (
    AdultCorpusError,
    SafetyReport,
    assert_corpus_is_not_adult,
)

DEFAULT_PAGES = 3
DEFAULT_MAX_IMAGES = 600
MASTODON_PAGE_LIMIT = 40
BSKY_PAGE_LIMIT = 50

# Bluesky feed generators worth naming — an at:// URI is accepted directly.
BSKY_FEED_ALIASES = {
    "whats-hot": "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot",
}


class CorpusSpecError(ValueError):
    pass


def _prefer_bounded_variant(candidate: Candidate) -> Candidate:
    """Swap in the platform's own small variant for corpus scanning.

    A corpus run fetches hundreds of images under a 512KB Range cap. Social
    originals routinely exceed that (3838x2161 was the first one sampled), and
    a capped fetch of one truncates mid-file — libpng/libjpeg then either fail
    outright or hand back a corrupted bottom half, which is worse. The small
    variant is a complete file, is still far above the 112px the embedder
    needs, and is what the bundle then hashes, so `image_url` continues to
    name exactly the bytes that were observed.
    """
    bounded = candidate.extra.get("preview_url") or candidate.extra.get("thumb")
    if not bounded:
        return candidate
    return replace(
        candidate,
        image_url=bounded,
        extra={**candidate.extra, "variant": "preview", "fullsize_url": candidate.image_url},
    )


@dataclass(frozen=True)
class CorpusSpec:
    """`platform:kind/value` — e.g. `mastodon:tag/selfie@mstdn.social`,
    `bsky:feed/whats-hot`, `bsky:author/alice.bsky.social`.
    """

    platform: str
    kind: str
    value: str
    instance: str | None = None

    def __str__(self) -> str:
        base = f"{self.platform}:{self.kind}/{self.value}"
        return f"{base}@{self.instance}" if self.instance else base


def parse_spec(raw: str) -> CorpusSpec:
    if ":" not in raw or "/" not in raw.split(":", 1)[1]:
        raise CorpusSpecError(
            f"bad corpus spec {raw!r} — expected platform:kind/value, e.g. mastodon:tag/selfie or bsky:feed/whats-hot"
        )
    platform, rest = raw.split(":", 1)
    kind, value = rest.split("/", 1)
    instance = None
    if platform == "mastodon" and "@" in value:
        value, instance = value.rsplit("@", 1)

    platform, kind = platform.strip().lower(), kind.strip().lower()
    valid = {("mastodon", "tag"), ("mastodon", "account"), ("bsky", "feed"), ("bsky", "author")}
    if (platform, kind) not in valid:
        raise CorpusSpecError(
            f"unsupported corpus spec {raw!r} — supported: "
            "mastodon:tag/<tag>[@instance], mastodon:account/<handle>, bsky:feed/<alias|at-uri>, bsky:author/<handle>"
        )
    if not value:
        raise CorpusSpecError(f"corpus spec {raw!r} has an empty value")
    # An adult tag or instance is refused at parse time, before the first
    # request — the run must not happen, rather than happen and be filtered.
    try:
        assert_corpus_is_not_adult(platform, kind, value, instance)
    except AdultCorpusError as e:
        raise CorpusSpecError(str(e)) from e
    return CorpusSpec(platform=platform, kind=kind, value=value, instance=instance)


async def _mastodon_tag(
    client: httpx.AsyncClient, spec: CorpusSpec, pages: int, safety: SafetyReport, allow_sensitive: bool
) -> list[Candidate]:
    instance = spec.instance or mastodon.DEFAULT_INSTANCES[0]
    out: list[Candidate] = []
    max_id: str | None = None
    for _ in range(pages):
        params = {"limit": MASTODON_PAGE_LIMIT, "only_media": "true"}
        if max_id:
            params["max_id"] = max_id
        statuses = await mastodon._get_json(
            client, f"https://{instance}/api/v1/timelines/tag/{spec.value}", params
        )
        if not statuses:
            break
        out += [
            c
            for s in statuses
            for c in mastodon._extract_image_candidates(s, allow_sensitive=allow_sensitive, report=safety)
        ]
        max_id = statuses[-1].get("id")
        if not max_id:
            break
    return out


async def _bsky_feed(
    client: httpx.AsyncClient, spec: CorpusSpec, pages: int, safety: SafetyReport, allow_sensitive: bool
) -> list[Candidate]:
    feed_uri = BSKY_FEED_ALIASES.get(spec.value, spec.value)
    if not feed_uri.startswith("at://"):
        raise CorpusSpecError(
            f"unknown bsky feed {spec.value!r} — pass an at:// feed URI or one of: {', '.join(BSKY_FEED_ALIASES)}"
        )
    out: list[Candidate] = []
    cursor: str | None = None
    for _ in range(pages):
        params: dict = {"feed": feed_uri, "limit": BSKY_PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        data = await bluesky._get_json(client, "app.bsky.feed.getFeed", params)
        items = data.get("feed", [])
        if not items:
            break
        out += [
            c
            for item in items
            for c in bluesky._extract_image_candidates(
                item.get("post") or {}, allow_sensitive=allow_sensitive, report=safety
            )
        ]
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


async def _ingest_one(
    client: httpx.AsyncClient, spec: CorpusSpec, pages: int, safety: SafetyReport, allow_sensitive: bool
) -> list[Candidate]:
    if spec.platform == "mastodon" and spec.kind == "tag":
        return await _mastodon_tag(client, spec, pages, safety, allow_sensitive)
    if spec.platform == "mastodon" and spec.kind == "account":
        return await mastodon.search_posts(
            client,
            spec.value,
            instances=[spec.instance] if spec.instance else None,
            allow_sensitive=allow_sensitive,
            safety=safety,
        )
    if spec.platform == "bsky" and spec.kind == "feed":
        return await _bsky_feed(client, spec, pages, safety, allow_sensitive)
    if spec.platform == "bsky" and spec.kind == "author":
        return await bluesky.search_posts(client, spec.value, allow_sensitive=allow_sensitive, safety=safety)
    raise CorpusSpecError(f"unsupported corpus spec: {spec}")


async def ingest(
    client: httpx.AsyncClient,
    specs: list[CorpusSpec],
    pages: int = DEFAULT_PAGES,
    max_images: int = DEFAULT_MAX_IMAGES,
    allow_sensitive: bool = False,
):
    """Every spec concurrently; deduped by image URL; capped at `max_images`.

    Returns (candidates, reports, safety) — SourceReport is the same type
    Arm A uses, so a corpus that is blocked or empty is reported rather than
    silently contributing nothing, and `safety` counts what the explicit
    filter removed before any of it was fetched (`search/safety.py`).
    """
    from faceanchor.search.fanout import ARM_A_TIMEOUT_S, _run_source

    safety = SafetyReport()
    pairs = await asyncio.gather(
        *[
            _run_source(
                str(spec), _ingest_one(client, spec, pages, safety, allow_sensitive), ARM_A_TIMEOUT_S * pages
            )
            for spec in specs
        ]
    )

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for spec, (group, _) in zip(specs, pairs):
        for c in (_prefer_bounded_variant(x) for x in group):
            if c.image_url in seen:
                continue
            seen.add(c.image_url)
            # Which spec produced this candidate. Without it a corpus database
            # built from three timelines cannot say which one an image came
            # from, and "by_corpus" is a column of empty strings.
            candidates.append(replace(c, extra={**c.extra, "corpus": str(spec)}))
            if len(candidates) >= max_images:
                return candidates, [report for _, report in pairs], safety
    return candidates, [report for _, report in pairs], safety
