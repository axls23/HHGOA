from __future__ import annotations

import json
import subprocess
import tempfile
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

    async def anchored_events(self, from_block: int = 0, to_block: str | int = "latest") -> list[dict]:
        """Every `Anchored` and `Revoked` event this contract has emitted.

        This is the public footprint: what the contract has recorded, readable
        by anyone with the address and an RPC, with no dependency on our
        evidence files. Public RPCs commonly cap `eth_getLogs` ranges, hence
        the explicit `from_block`.
        """
        events = []
        for name in ("Anchored", "Revoked"):
            event = getattr(self._contract.events, name)
            for log in await event.get_logs(from_block=from_block, to_block=to_block):
                args = dict(log["args"])
                events.append(
                    {
                        "event": name,
                        "block": log["blockNumber"],
                        "tx_hash": log["transactionHash"].hex(),
                        "digest": (args.get("digest") or args.get("consentDigest")).hex(),
                        "attester": args.get("attester") or args.get("revoker"),
                        "timestamp": args.get("timestamp"),
                        "cid": args.get("cid", ""),
                    }
                )
        return sorted(events, key=lambda e: (e["block"], e["event"]))

    async def anchor_with_proof(self, digest: bytes, cid: str, proof: dict, public_signals: list[str]) -> AnchorReceipt:
        # snarkjs's proof.json is (pi_a, pi_b, pi_c, affine-with-a-trailing-"1")
        # in a curve-library-internal layout, and pi_b's inner coordinate order
        # is swapped relative to what Solidity's pairing precompile expects.
        # `snarkjs generatecall` is the one tool that gets that swap right —
        # shell out to it rather than re-deriving the reordering by hand.
        from faceanchor.zk.prove import _snarkjs_bin

        with tempfile.TemporaryDirectory(prefix="faceanchor_zk_calldata_") as tmp:
            proof_path = Path(tmp) / "proof.json"
            public_path = Path(tmp) / "public.json"
            proof_path.write_text(json.dumps(proof))
            public_path.write_text(json.dumps(public_signals))

            result = subprocess.run(
                [_snarkjs_bin(), "generatecall", public_path.name, proof_path.name],
                cwd=tmp,
                check=True,
                capture_output=True,
                text=True,
            )
        a, b, c, pub_signals = json.loads(f"[{result.stdout.strip()}]")
        a = [int(x, 16) for x in a]
        b = [[int(x, 16) for x in row] for row in b]
        c = [int(x, 16) for x in c]
        pub_signals = [int(x, 16) for x in pub_signals]

        return await self._send(self._contract.functions.anchorWithProof(digest, cid, a, b, c, pub_signals))
