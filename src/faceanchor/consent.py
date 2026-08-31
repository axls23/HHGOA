"""Subject policy hard rules (PRD §3): a run without consent does not happen.

No persistent embedding store, no raw biometric on-chain — those are enforced
by construction elsewhere (embeddings never leave process memory except as
sha256 digests in the bundle). This module enforces the one rule that needs
an explicit gate: `--subject-consent` is required, and its digest is what
gets anchored (§6.3's `consent_digest`), not the artifact itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class MissingConsentError(Exception):
    pass


def load_consent_digest(consent_path: str | Path | None) -> str:
    if consent_path is None:
        raise MissingConsentError(
            "--subject-consent is required (PRD §3: no run without an explicit consent artifact)."
        )
    path = Path(consent_path)
    if not path.exists():
        raise MissingConsentError(f"consent artifact not found: {path}")

    raw = path.read_bytes()
    # Canonicalize the parsed JSON before hashing so re-serialization (e.g. a
    # verifier re-reading the artifact) produces the same digest regardless
    # of the file's original byte-for-byte whitespace/key order.
    from faceanchor.evidence.jcs import canonicalize

    parsed = json.loads(raw)
    return hashlib.sha256(canonicalize(parsed)).hexdigest()
