"""Shared candidate/match types across every discovery source (PRD §5.2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """One discoverable image, pre-scoring. `platform`-specific identifiers are
    kept alongside the plain image_url so a hit can round-trip into an
    evidence bundle (PRD §6.3) without a second fetch.
    """

    platform: str            # "bsky" | "mastodon" | "commons" | "ddg"
    image_url: str
    post_uri: str             # at:// URI, status URL, or Commons File: page URL
    author: str | None = None
    text: str | None = None
    record_cid: str | None = None  # AT Protocol content-addressed record CID, if available
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredMatch:
    candidate: Candidate
    score: float
    metric: str            # "phash" | "cosine"
    image_bytes_sha256: str
