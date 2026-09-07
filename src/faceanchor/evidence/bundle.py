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
    # Which detector aligned the crop. Not cosmetic: the two backends align
    # slightly differently, so the same face embeds to a cosine of ~0.95
    # across them. Two bundles naming the same embed_model but different
    # detectors are not directly comparable, and a verifier should be able to
    # see that rather than infer it.
    detector: str = "yunet"


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
    discovery: dict | None = None,
    match_extra: dict | None = None,
) -> dict:
    candidate: Candidate = match.candidate
    run_block = {
        "pipeline_commit": pipeline_commit,
        "run_id": run_id or str(ULID()),
        "observed_at": observed_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if discovery is not None:
        # How the match was *found* is part of what a verifier is checking. A
        # by-face hit over a named corpus and a hit seeded by a text query
        # describing the subject are different claims, and the bundle should
        # not let them look alike. The text query itself is only ever hashed —
        # it describes a person, so it does not belong in an anchored record.
        run_block["discovery"] = discovery
    match_block = {
        "platform": candidate.platform,
        "uri": candidate.post_uri,
        "record_cid": candidate.record_cid or "",
        "author_did": match_author_did or candidate.author or "",
        "image_url_sha256": match_image_url_sha256,
        "image_sha256": match.image_bytes_sha256,
        "image_phash": "",  # filled by caller if the phash short-circuit accepted the match
        "text_sha256": match_text_sha256,
    }
    if match_extra:
        # Provenance that only some discovery modes have — today, the
        # reverse-image provider and the match class it assigned. Merged in
        # rather than always-present so a bundle built the old way stays
        # byte-identical, and therefore keeps the same digest.
        match_block.update(match_extra)

    return {
        "v": BUNDLE_VERSION,
        "run": run_block,
        "probe": {
            "image_sha256": probe.image_sha256,
            "image_phash": probe.image_phash,
            "embed_model": probe.embed_model,
            "detector": probe.detector,
            "embed_sha256": probe.embed_sha256,
            "consent_digest": probe.consent_digest,
        },
        "match": match_block,
        "score": {
            "metric": match.metric,
            "value": match.score,
            "threshold": 0.363 if match.metric == "cosine" else 8,
            "path": "embed" if match.metric == "cosine" else "phash",
        },
    }


def bundle_digest(bundle: dict) -> bytes:
    return jcs_digest(bundle)
