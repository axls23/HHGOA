"""Block-explorer URLs — the part of verification that needs none of this code.

The point of anchoring is that a stranger can check the claim without trusting
us. That only holds if they can reach the chain by a route we do not control:
a public explorer, where `anchoredAt(digest)` is a Read Contract call anyone
can make in a browser. These helpers exist to hand them that link.

Not every chain offers the same route, and the differences are the reason this
is a dataclass rather than an f-string: EVM tx hashes are 0x-prefixed hex and
Solana signatures are base58; devnet needs a `?cluster=` selector; and SPL
Memo has no contract state to read, so on Solana there is a transaction to
show a stranger but nothing for them to query.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Explorer:
    name: str
    base_url: str
    hex_tx: bool = True
    query_suffix: str = ""
    readable_contract: bool = True

    def tx(self, tx_hash: str) -> str:
        return f"{self.base_url}/tx/{_hex(tx_hash) if self.hex_tx else tx_hash}{self.query_suffix}"

    def address(self, address: str) -> str:
        return f"{self.base_url}/address/{address}{self.query_suffix}"

    def read_contract(self, address: str) -> str | None:
        """Deep link to the tab where a third party checks a digest themselves.

        `None` where the chain has no such tab — better to say so than to hand
        someone a link that cannot answer the question they went there to ask.
        """
        if not self.readable_contract:
            return None
        return f"{self.base_url}/address/{address}#readContract"


def _hex(value: str) -> str:
    return value if value.startswith("0x") else f"0x{value}"


# Anvil is deliberately absent: a local devnet has no public footprint, and
# pretending otherwise would be the exact false-assurance this module exists
# to prevent. Solana devnet is present but not contract-readable — the memo is
# publicly visible in the tx, and that is the whole of what SPL Memo records.
EXPLORERS: dict[str, Explorer] = {
    "amoy": Explorer(name="PolygonScan (Amoy)", base_url="https://amoy.polygonscan.com"),
    "solana": Explorer(
        name="Solana Explorer (devnet)",
        base_url="https://explorer.solana.com",
        hex_tx=False,
        query_suffix="?cluster=devnet",
        readable_contract=False,
    ),
}


def for_chain(chain: str) -> Explorer | None:
    return EXPLORERS.get(chain)
