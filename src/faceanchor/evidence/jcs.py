"""RFC 8785 JSON Canonicalization Scheme + keccak256 digest (PRD §6.3).

Canonicalisation is what makes re-verification deterministic across machines
— never hash `json.dumps` output; key ordering and number formatting are not
guaranteed stable across Python versions/platforms the way JCS is.
"""

from __future__ import annotations

import rfc8785
from web3 import Web3


def canonicalize(obj: dict) -> bytes:
    return rfc8785.dumps(obj)


def digest(obj: dict) -> bytes:
    """keccak256(JCS(obj)) — the 32-byte value anchored on-chain."""
    return Web3.keccak(canonicalize(obj))
