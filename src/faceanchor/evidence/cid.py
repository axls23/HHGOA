"""Local CIDv1 computation — raw leaves, sha2-256, no IPFS daemon (PRD §6.3).

The digest is meaningful without retrievability; `ipfs add` via Kubo is an
optional add-on for actual pinning, not required for verification.
"""

from __future__ import annotations

from multiformats import CID, multihash


def compute_cid(data: bytes) -> str:
    digest = multihash.digest(data, "sha2-256")
    return str(CID("base32", 1, "raw", digest))
