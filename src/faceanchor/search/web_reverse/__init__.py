"""Genuine reverse-image discovery: an image goes out, web URLs come back.

Distinct from both existing discovery arms (see `base.py` for the three-way
distinction). Providers are registered here by CLI name so `--provider` is
the only thing that changes when a second one is added.
"""

from __future__ import annotations

from faceanchor.search.web_reverse.base import (
    MATCH_TYPES,
    DiscoveryOutcome,
    MalformedResponseError,
    ProviderAuthError,
    ProviderConfigError,
    ProviderHTTPError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ReverseImageError,
    ReverseImageProvider,
    ReverseImageResult,
    dedupe_results,
)
from faceanchor.search.web_reverse.google_vision import GoogleWebDetectionProvider
from faceanchor.search.web_reverse.mock import ReplayProvider
from faceanchor.search.web_reverse.social import (
    DEFAULT_DOMAINS,
    PlatformMatcher,
    SocialFilterReport,
    filter_social,
    parse_domain_overrides,
)

#: CLI name -> provider class. `mock` is a deliberate test double, documented
#: in mock.py; it is never selected implicitly.
PROVIDERS: dict[str, type[ReverseImageProvider]] = {
    GoogleWebDetectionProvider.name: GoogleWebDetectionProvider,
    ReplayProvider.name: ReplayProvider,
}

#: Providers that actually perform a reverse-image search against a live index.
REAL_PROVIDERS = (GoogleWebDetectionProvider.name,)


def available_providers() -> list[str]:
    return sorted(PROVIDERS)


def get_provider(name: str, **kwargs) -> ReverseImageProvider:
    try:
        cls = PROVIDERS[name.strip().lower()]
    except KeyError:
        raise ProviderConfigError(
            f"unknown reverse-image provider {name!r} — available: {', '.join(available_providers())}"
        ) from None
    return cls(**kwargs)


__all__ = [
    "DEFAULT_DOMAINS",
    "MATCH_TYPES",
    "PROVIDERS",
    "REAL_PROVIDERS",
    "DiscoveryOutcome",
    "GoogleWebDetectionProvider",
    "MalformedResponseError",
    "PlatformMatcher",
    "ProviderAuthError",
    "ProviderConfigError",
    "ProviderHTTPError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ReplayProvider",
    "ReverseImageError",
    "ReverseImageProvider",
    "ReverseImageResult",
    "SocialFilterReport",
    "available_providers",
    "dedupe_results",
    "filter_social",
    "get_provider",
    "parse_domain_overrides",
]
