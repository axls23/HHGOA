"""Which reverse-image results are social-media posts, and which are noise.

A reverse-image provider answers with the open web: news sites, stock photo
pages, scrapers, aggregators, someone's blog. The pipeline's job is narrower
— find a *social-media post* carrying this face — so results are classified
by domain before anything is downloaded.

The rule is a configurable domain map, not a guess, and it defaults to the
two platforms the rest of this repo already knows how to handle end to end
(Bluesky and Mastodon): their candidates flow into the same fetch/score
cascade, the same `Candidate` type, and — for Bluesky — the same §7.2
content-addressed mutation check at verify time. Adding a platform here
without the handling behind it would produce candidates the evidence bundle
cannot describe honestly.

Mastodon is the awkward one: it is thousands of independent domains, so
there is no list that can be complete. Two mechanisms, both explicit:

- a seed list of well-known instances, extensible with `--social-domain
  mastodon:example.social` or `FACEANCHOR_SOCIAL_DOMAINS`;
- a URL-shape heuristic for the Mastodon status/media URL layout
  (`/@user/<snowflake>`, `/users/<x>/statuses/<n>`, `/system/media_attachments/…`),
  which is stable across instances because it is Mastodon's own routing.

The heuristic can be switched off (`heuristics=False`) when only the explicit
list should count. Nothing here treats "not classified" as a failure — an
unclassified result is reported as skipped, with the URL, so a run that
found plenty of web hits and no social ones says exactly that.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from faceanchor.search.models import Candidate
from faceanchor.search.safety import ADULT_INSTANCES
from faceanchor.search.web_reverse.base import (
    ReverseImageResult,
    match_rank,
)

ENV_EXTRA_DOMAINS = "FACEANCHOR_SOCIAL_DOMAINS"

#: platform -> registrable domains whose hosts (or subdomains) belong to it.
DEFAULT_DOMAINS: dict[str, tuple[str, ...]] = {
    "bsky": ("bsky.app", "bsky.social", "bsky.network", "atproto.com"),
    "mastodon": (
        "mastodon.social", "mastodon.online", "mastodon.world", "mstdn.social",
        "fosstodon.org", "hachyderm.io", "infosec.exchange", "mas.to",
        "techhub.social", "mastodon.art", "c.im", "chaos.social", "toot.community",
        "ioc.exchange", "phpc.social", "hostux.social", "sfba.social", "genomic.social",
    ),
}

# Mastodon's own routes, identical on every instance because they come from
# Mastodon itself rather than from any one operator's configuration.
_MASTODON_PATHS = (
    re.compile(r"^/@[A-Za-z0-9_]+(?:@[A-Za-z0-9.\-]+)?/\d+/?$"),
    re.compile(r"^/users/[A-Za-z0-9_]+/statuses/\d+"),
    re.compile(r"^/system/media_attachments/"),
    re.compile(r"^/media_attachments/files/"),
)


def _host(url: str | None) -> str:
    if not url:
        return ""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def parse_domain_overrides(raw: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """`["mastodon:example.social", "bsky:pds.example"]` -> domain map fragment."""
    out: dict[str, list[str]] = {}
    for item in raw:
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad social-domain override {item!r} — expected platform:domain")
        platform, domain = item.split(":", 1)
        platform, domain = platform.strip().lower(), domain.strip().lower().lstrip(".")
        if not platform or not domain:
            raise ValueError(f"bad social-domain override {item!r} — expected platform:domain")
        out.setdefault(platform, []).append(domain)
    return {k: tuple(v) for k, v in out.items()}


@dataclass(frozen=True)
class PlatformMatcher:
    domains: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: DEFAULT_DOMAINS)
    heuristics: bool = True

    @classmethod
    def build(cls, overrides: Iterable[str] = (), *, heuristics: bool = True, env: Mapping[str, str] | None = None):
        """Defaults + `FACEANCHOR_SOCIAL_DOMAINS` + explicit CLI overrides."""
        env = os.environ if env is None else env
        merged: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in DEFAULT_DOMAINS.items()}
        from_env = env.get(ENV_EXTRA_DOMAINS, "")
        for source in (parse_domain_overrides(from_env.split(",")), parse_domain_overrides(overrides)):
            for platform, domains in source.items():
                merged[platform] = tuple(dict.fromkeys(merged.get(platform, ()) + domains))
        return cls(domains=merged, heuristics=heuristics)

    def _by_domain(self, host: str) -> str | None:
        if not host:
            return None
        for platform, domains in self.domains.items():
            for domain in domains:
                if host == domain or host.endswith("." + domain):
                    return platform
        return None

    def classify(self, page_url: str | None, image_url: str | None = None) -> str | None:
        """The platform this result belongs to, or None if it is not social.

        The page URL decides when it is classifiable — it is the thing a
        verifier would open. The image URL is the fallback, because a CDN host
        (`cdn.bsky.app`, `files.mastodon.social`) still identifies the
        platform when the provider returned an image without its page.
        """
        for url in (page_url, image_url):
            platform = self._by_domain(_host(url))
            if platform:
                return platform
        if not self.heuristics:
            return None
        for url in (page_url, image_url):
            if not url:
                continue
            try:
                path = urlsplit(url).path
            except ValueError:
                continue
            if any(pattern.search(path) for pattern in _MASTODON_PATHS):
                return "mastodon"
        return None


@dataclass
class SocialFilterReport:
    """What the filter kept, what it dropped, and why — per result.

    The skipped list is the honest half. "34 results, 0 social" and "0 results"
    are different facts, and so are "not a social domain" and "social, but the
    provider gave no image URL to fetch".
    """

    candidates: list[Candidate] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    by_platform: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        if not self.by_platform:
            return "none"
        return ", ".join(f"{p} {n}" for p, n in sorted(self.by_platform.items()))

    def skip_summary(self) -> str:
        counts = Counter(entry["reason"] for entry in self.skipped)
        return ", ".join(f"{reason} {n}" for reason, n in sorted(counts.items())) or "none"


def _best_first(results: Iterable[ReverseImageResult]) -> list[ReverseImageResult]:
    """Order results so the most informative copy of each image comes first.

    Providers report the same image more than once: Google lists it under
    `fullMatchingImages` with no page, *and* under `pagesWithMatchingImages`
    with the page it lives on. Deduplicating in arrival order would keep the
    pageless copy and throw away the only URL a human could open — the
    evidence bundle would then locate the match by CDN URL alone. So a result
    carrying a page URL sorts ahead of one that does not, then the stronger
    match type wins, and the original order breaks remaining ties.
    """
    return [
        result
        for _, result in sorted(
            ((i, r) for i, r in enumerate(results)),
            key=lambda pair: (0 if pair[1].page_url else 1, match_rank(pair[1].match_type), pair[0]),
        )
    ]


def filter_social(
    results: Iterable[ReverseImageResult], matcher: PlatformMatcher | None = None
) -> SocialFilterReport:
    """Reverse-image results -> `Candidate`s the existing cascade can score.

    Deduplicated by image URL, which is the stable key the rest of the
    pipeline already dedupes on (`corpus.ingest`, `bluesky.search_posts`), so
    the same image reached from two pages is fetched and embedded once — and
    the copy that survives is the one that names a page (see `_best_first`).
    """
    matcher = matcher or PlatformMatcher.build()
    report = SocialFilterReport()
    seen: set[str] = set()

    for result in _best_first(results):
        # A reverse-image result is a bare URL: none of the per-post
        # sensitivity metadata the other two arms read comes with it. Blocking
        # hosts that exist to serve adult content is the only filter available
        # at this layer, so it is applied before classification.
        if _host(result.page_url) in ADULT_INSTANCES or _host(result.image_url) in ADULT_INSTANCES:
            report.skipped.append(
                {"reason": "adult_instance", "page_url": result.page_url, "image_url": result.image_url}
            )
            continue
        platform = matcher.classify(result.page_url, result.image_url)
        if platform is None:
            report.skipped.append(
                {"reason": "not_social", "page_url": result.page_url, "image_url": result.image_url}
            )
            continue
        if not result.image_url:
            report.skipped.append(
                {"reason": "no_image_url", "page_url": result.page_url, "image_url": None, "platform": platform}
            )
            continue
        if result.image_url in seen:
            report.skipped.append(
                {"reason": "duplicate", "page_url": result.page_url, "image_url": result.image_url,
                 "platform": platform}
            )
            continue
        seen.add(result.image_url)
        report.by_platform[platform] += 1
        report.candidates.append(
            Candidate(
                platform=platform,
                image_url=result.image_url,
                # No page URL means the provider found the image without
                # naming a page. The image URL is then the only locator there
                # is, and pretending otherwise would put an empty `uri` in the
                # evidence bundle.
                post_uri=result.page_url or result.image_url,
                extra={
                    "discovery": "web_reverse",
                    "provider": result.provider,
                    "match_type": result.match_type,
                    "page_url": result.page_url or "",
                    "page_title": result.metadata.get("page_title", ""),
                },
            )
        )
    return report
