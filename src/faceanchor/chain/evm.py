from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from web3 import AsyncWeb3, Web3
from web3.middleware import ExtraDataToPOAMiddleware

from faceanchor.chain.base import AnchorReceipt, ChainAdapter, VerifyResult

# Populated by `forge build`; contracts/ is a sibling of src/, both live at repo root.
_CONTRACTS_OUT = Path(__file__).resolve().parents[3] / "contracts" / "out" / "EvidenceAnchor.sol" / "EvidenceAnchor.json"


@lru_cache(maxsize=1)
def _abi() -> list[dict]:
    if not _CONTRACTS_OUT.exists():
        raise FileNotFoundError(
            f"{_CONTRACTS_OUT} not found — run `forge build` in contracts/ first."
        )
    return json.loads(_CONTRACTS_OUT.read_text())["abi"]


class EvmAdapter(ChainAdapter):
    """Generic EVM adapter (Anvil, Polygon Amoy, or any evmChainId-compatible RPC).

    Nonce is cached in-process after the first fetch to avoid a round trip per
    tx (PRD §5.3 / §9 latency budget).
    """

    def __init__(self, rpc_url: str, private_key: str, contract_address: str, poa: bool = False):
        self._w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        if poa:
            self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self._account = self._w3.eth.account.from_key(private_key)
        # Deploy scripts/broadcast logs commonly emit lowercase addresses;
        # web3.py requires EIP-55 checksummed input.
        self._contract = self._w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=_abi())
        self._cached_nonce: int | None = None

    async def _next_nonce(self) -> int:
        if self._cached_nonce is None:
            self._cached_nonce = await self._w3.eth.get_transaction_count(self._account.address)
        else:
            self._cached_nonce += 1
        return self._cached_nonce

    async def _send(self, contract_fn) -> AnchorReceipt:
        nonce = await self._next_nonce()
        tx = await contract_fn.build_transaction(
            {
                "from": self._account.address,
                "nonce": nonce,
                "chainId": await self._w3.eth.chain_id,
            }
        )
        signed = self._account.sign_transaction(tx)
        tx_hash = await self._w3.eth.send_raw_transaction(signed.raw_transaction)
        return AnchorReceipt(tx_hash=tx_hash.hex())

    async def anchor(self, digest: bytes, cid: str) -> AnchorReceipt:
        return await self._send(self._contract.functions.anchor(digest, cid))

    async def verify(self, digest: bytes) -> VerifyResult:
        ok, ts = await self._contract.functions.verify(digest).call()
        return VerifyResult(ok=ok, timestamp=ts)

    async def wait_for_inclusion(self, receipt: AnchorReceipt) -> None:
        await self._w3.eth.wait_for_transaction_receipt(receipt.tx_hash)

    async def revoke(self, consent_digest: bytes) -> AnchorReceipt:
        return await self._send(self._contract.functions.revoke(consent_digest))

    async def consent_status(self, consent_digest: bytes) -> VerifyResult:
        revoked, ts = await self._contract.functions.consentStatus(consent_digest).call()
        return VerifyResult(ok=revoked, timestamp=ts)
