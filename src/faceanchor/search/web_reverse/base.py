"""Reverse-image provider abstraction (PRD §5.2, new discovery arm).

This is the arm the rest of the repo did not have: an image goes *out* to a
provider that indexes the web by image content, and URLs come back. It is a
different question from anything else in `search/`:

    --query        "where do words about this person point?"
    --by-face      "is this face anywhere in a corpus I named?"
    --discovery web-reverse
                   "which pages on the open web carry this image?"

Only the third is reverse-image search, and it is the only one that submits
the probe image itself to a third party. That transfer is a trust boundary
the other two do not cross, which is why `cli.py` gates it behind an explicit
acknowledgement rather than folding it into the consent artifact.

Two rules this module exists to enforce:

1. Providers return `ReverseImageResult`, never their own JSON. A provider's
   response shape stops at the edge of its own module, so swapping Google for
   something else touches one file.
2. A provider failure is never allowed to look like an empty result. Every
   failure mode raises a `ReverseImageError` subclass; "the provider answered,
   and the answer was nothing" is the *only* thing that returns `[]`. The
   caller reports those as PROVIDER_ERROR and NO_RESULTS respectively, and a
   run must not silently degrade one into the other — a search that broke and
   a search that found nothing are opposite facts about the subject.

Deliberately *not* used: Google's `webEntities` (text labels describing what
is in the image). Turning those into a text query would be exactly the
"pretend a text search is reverse-image search" failure this arm exists to
avoid, so they are read and discarded.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

# Ordered strongest-first; used to collapse duplicates onto their best claim.
MATCH_TYPES = ("full", "partial", "similar", "unknown")
_MATCH_RANK = {name: i for i, name in enumerate(MATCH_TYPES)}


class DiscoveryOutcome(str, Enum):
    """Why a reverse-image discovery ended the way it did.

    `NO_RESULTS` and `PROVIDER_ERROR` are separate members on purpose: the
    first is evidence about the subject, the second is evidence about the
    plumbing, and a tool that conflates them lies about both.
    """

    OK = "ok"
    NO_RESULTS = "no_results"
    NO_SOCIAL_CANDIDATES = "no_social_candidates"
    PROVIDER_ERROR = "provider_error"


class ReverseImageError(Exception):
    """Base for every provider-side failure. Never raised for 'found nothing'."""


class ProviderConfigError(ReverseImageError):
    """Unknown provider name, or a provider configured with nonsense."""


class ProviderAuthError(ReverseImageError):
    """No usable credentials, or the provider rejected the ones given.

    Raised loudly and never swallowed: an unauthenticated run must fail, not
    quietly fall back to some other kind of search that would then be
    mislabelled as reverse-image discovery.
    """


class ProviderTimeoutError(ReverseImageError):
    pass


class ProviderHTTPError(ReverseImageError):
    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        super().__init__(f"provider returned HTTP {status_code}{': ' + detail if detail else ''}")


class ProviderRateLimitError(ProviderHTTPError):
    pass


class MalformedResponseError(ReverseImageError):
    """The provider answered, but not in a shape this adapter can read."""


@dataclass(frozen=True)
class ReverseImageResult:
    """One normalized hit from a reverse-image provider.

    `image_url` is the image the provider says matches; `page_url` is the page
    it found that image on. Either can be absent — a top-level "full matching
    image" has no page, and a page hit whose image URL the provider omits has
    no image — so downstream code must handle both and say which it dropped.
    """

    provider: str
    image_url: str | None
    page_url: str | None
    match_type: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.match_type not in _MATCH_RANK:
            object.__setattr__(self, "match_type", "unknown")

    @property
    def key(self) -> tuple[str, str]:
        return (self.page_url or "", self.image_url or "")


def match_rank(match_type: str) -> int:
    """Sort key: lower is a stronger claim. Unknown types sort last."""
    return _MATCH_RANK.get(match_type, len(MATCH_TYPES))


def dedupe_results(results: Iterable[ReverseImageResult]) -> list[ReverseImageResult]:
    """Collapse duplicate (page, image) pairs onto their strongest match_type.

    Google reports the same URL under more than one bucket routinely — a page
    can appear in `pagesWithMatchingImages` with both a full and a partial
    match for the same image. Keeping both would double-count the candidate
    and let the weaker claim win by ordering accident, so the strongest claim
    for a pair is the one that survives. Insertion order is preserved.
    """
    best: dict[tuple[str, str], ReverseImageResult] = {}
    for result in results:
        if not (result.image_url or result.page_url):
            continue
        existing = best.get(result.key)
        if existing is None or _MATCH_RANK[result.match_type] < _MATCH_RANK[existing.match_type]:
            best[result.key] = result
    return list(best.values())


class ReverseImageProvider(ABC):
    """Submit an image, get back normalized web results.

    `search` is async because every caller here is: the pipeline already owns
    an `httpx.AsyncClient` pool (`fanout.make_client`) and passes it in, so a
    reverse-image call shares connection limits and timeouts with the
    candidate fetches rather than opening a second, unbounded one.
    """

    #: CLI name (`--provider <name>`).
    name: str = "abstract"
    #: Stable identifier recorded in the evidence bundle. Never the CLI name,
    #: so a test double can never leave a record claiming a real provider.
    provider_id: str = "abstract"

    @abstractmethod
    async def search(
        self, image_path: str | os.PathLike[str] | Path, *, client: httpx.AsyncClient | None = None
    ) -> list[ReverseImageResult]:
        """Reverse-image search `image_path`.

        Returns `[]` only when the provider genuinely reported no matches.
        Raises a `ReverseImageError` subclass for every other outcome.
        """
        raise NotImplementedError
