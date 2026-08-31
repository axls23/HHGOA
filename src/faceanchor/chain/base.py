from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorReceipt:
    tx_hash: str
    block_timestamp: int | None = None


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    timestamp: int


class ChainAdapter(ABC):
    """Common interface every chain backend implements.

    Swapping targets (Anvil / Amoy / Solana devnet) is a config change against
    this interface, per PRD §6.1's testnet-impermanence mitigation.
    """

    @abstractmethod
    async def anchor(self, digest: bytes, cid: str) -> AnchorReceipt:
        """Submit `digest` (32 bytes) + `cid` for anchoring. Returns once the tx is sent."""

    @abstractmethod
    async def verify(self, digest: bytes) -> VerifyResult:
        """Read-only check: has `digest` been anchored, and when."""

    @abstractmethod
    async def wait_for_inclusion(self, receipt: AnchorReceipt) -> None:
        """Block until `receipt`'s tx is included. Called after streaming tx_hash to the client (PRD §4)."""
