"""Evidence bundle schema (PRD §6.3) — the JCS-canonicalised, keccak256-digested
JSON that gets anchored on-chain.

Explicitly excluded, everywhere in this module and downstream: raw embedding
vectors, image bytes, display names, handles-as-identity. Only hashes of
those things are ever included. A hash is not reversible; a 128-d template on
an immutable ledger is a permanent biometric record you cannot delete.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from ulid import ULID

from faceanchor.evidence.jcs import digest as jcs_digest
from faceanchor.search.models import Candidate, ScoredMatch

BUNDLE_VERSION = 1


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ProbeInfo:
    image_sha256: str
    image_phash: str
    embed_model: str
    embed_sha256: str
    consent_digest: str


def build_bundle(
    *,
    pipeline_commit: str,
    probe: ProbeInfo,
    match: ScoredMatch,
    match_text_sha256: str,
    match_image_url_sha256: str,
    match_author_did: str | None = None,
    run_id: str | None = None,
    observed_at: str | None = None,
) -> dict:
    candidate: Candidate = match.candidate
    return {
        "v": BUNDLE_VERSION,
        "run": {
            "pipeline_commit": pipeline_commit,
            "run_id": run_id or str(ULID()),
            "observed_at": observed_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "probe": {
            "image_sha256": probe.image_sha256,
            "image_phash": probe.image_phash,
            "embed_model": probe.embed_model,
            "embed_sha256": probe.embed_sha256,
            "consent_digest": probe.consent_digest,
        },
        "match": {
            "platform": candidate.platform,
            "uri": candidate.post_uri,
            "record_cid": candidate.record_cid or "",
            "author_did": match_author_did or candidate.author or "",
            "image_url_sha256": match_image_url_sha256,
            "image_sha256": match.image_bytes_sha256,
            "image_phash": "",  # filled by caller if the phash short-circuit accepted the match
            "text_sha256": match_text_sha256,
        },
        "score": {
            "metric": match.metric,
            "value": match.score,
            "threshold": 0.363 if match.metric == "cosine" else 8,
            "path": "embed" if match.metric == "cosine" else "phash",
        },
    }


def bundle_digest(bundle: dict) -> bytes:
    return jcs_digest(bundle)
