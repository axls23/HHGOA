"""§7.2 platform-mutation detection — falls out of AT Protocol for free.

AT Protocol records are content-addressed: every post carries a `record_cid`
computed by the platform itself. Re-fetching it at verify time and comparing
against what the bundle committed to distinguishes four cases (PRD §6.4/§7.2):

| on-chain digest | re-fetched record_cid | conclusion                          |
|-----------------|------------------------|--------------------------------------|
| fail            | —                       | OUR evidence was altered              |
| ok              | matches                 | record intact, post intact            |
| ok              | diverges                | the PLATFORM's content changed        |
| ok              | fetch 404s              | post deleted after observation        |

Only AT Protocol (Bluesky) carries a `record_cid` — Mastodon/Commons matches
have no equivalent content-addressing, so this check is a no-op (reports
"not applicable") for those platforms rather than a false mutation signal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

import httpx

BSKY_BASE_URL = "https://public.api.bsky.app"


class MutationVerdict(str, Enum):
    OUR_EVIDENCE_ALTERED = "our_evidence_altered"
    RECORD_INTACT = "record_intact"
    PLATFORM_CONTENT_CHANGED = "platform_content_changed"
    POST_DELETED = "post_deleted"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class MutationCheckResult:
    verdict: MutationVerdict
    detail: str


def _parse_at_uri(uri: str) -> tuple[str, str, str]:
    # at://did:plc:xxx/app.bsky.feed.post/yyy
    without_scheme = uri.removeprefix("at://")
    did, collection, rkey = without_scheme.split("/", 2)
    return did, collection, rkey


async def check_bsky_record(client: httpx.AsyncClient, bundle: dict) -> MutationCheckResult:
    match = bundle["match"]
    uri = match["uri"]
    expected_cid = match.get("record_cid") or ""
    expected_text_sha256 = match.get("text_sha256") or ""

    did, collection, rkey = _parse_at_uri(uri)
    try:
        resp = await client.get(
            f"{BSKY_BASE_URL}/xrpc/com.atproto.repo.getRecord",
            params={"repo": did, "collection": collection, "rkey": rkey},
            timeout=5.0,
        )
    except httpx.HTTPError as e:
        return MutationCheckResult(MutationVerdict.POST_DELETED, f"fetch failed: {e}")

    if resp.status_code == 400 or resp.status_code == 404:
        return MutationCheckResult(MutationVerdict.POST_DELETED, "record not found — post deleted after observation")
    resp.raise_for_status()

    data = resp.json()
    current_cid = data.get("cid", "")
    current_text = (data.get("value") or {}).get("text", "")
    current_text_sha256 = hashlib.sha256(current_text.encode()).hexdigest()

    if current_cid == expected_cid and current_text_sha256 == expected_text_sha256:
        return MutationCheckResult(MutationVerdict.RECORD_INTACT, f"record_cid matches ({current_cid})")

    return MutationCheckResult(
        MutationVerdict.PLATFORM_CONTENT_CHANGED,
        f"record_cid diverged: expected {expected_cid}, got {current_cid}",
    )


async def check_platform_mutation(client: httpx.AsyncClient, bundle: dict, digest_verified: bool) -> MutationCheckResult:
    if not digest_verified:
        return MutationCheckResult(
            MutationVerdict.OUR_EVIDENCE_ALTERED, "on-chain digest mismatch — the bundle itself was tampered with"
        )

    platform = bundle["match"]["platform"]
    if platform != "bsky":
        return MutationCheckResult(
            MutationVerdict.NOT_APPLICABLE, f"platform {platform!r} has no content-addressed record to compare"
        )

    # A Bluesky match found by reverse-image search is located by its public
    # https:// permalink, not by the at:// URI the AppView APIs address. There
    # is no record to re-fetch without resolving handle -> DID first, so this
    # reports "not applicable" rather than parsing the https URL into garbage
    # `repo`/`rkey` params and reading the resulting 400 as "post deleted" —
    # a false mutation signal is worse than an absent one.
    if not bundle["match"].get("uri", "").startswith("at://"):
        return MutationCheckResult(
            MutationVerdict.NOT_APPLICABLE,
            "bsky match is located by a web permalink, not an at:// record URI — nothing content-addressed to re-fetch",
        )

    return await check_bsky_record(client, bundle)
