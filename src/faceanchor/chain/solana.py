"""Solana devnet — SPL Memo, no custom program needed at all (PRD §6.1).

The tradeoff for that simplicity: SPL Memo has no on-chain state, so unlike
EvidenceAnchor's `anchoredAt` mapping there is no O(1) eth_call-equivalent
lookup by digest alone. `verify()` here scans the anchoring wallet's own
recent transaction history for a memo matching the digest — wallet-scoped,
not a global index. That's a real limitation of the memo-only design (worth
calling out in the README next to the "storage vs event" tradeoff PRD §6.2
already documents for the EVM side), not a corner cut in this adapter.
"""

from __future__ import annotations

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import Transaction

from faceanchor.chain.base import AnchorReceipt, ChainAdapter, VerifyResult

DEVNET_RPC_URL = "https://api.devnet.solana.com"
MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
VERIFY_SCAN_LIMIT = 100


def _memo_payload(digest: bytes, cid: str) -> bytes:
    return f"faceanchor:{digest.hex()}:{cid}".encode()


class SolanaAdapter(ChainAdapter):
    def __init__(self, keypair: Keypair | None = None, rpc_url: str = DEVNET_RPC_URL):
        self._client = AsyncClient(rpc_url)
        self._keypair = keypair or Keypair()

    async def airdrop(self, lamports: int = 1_000_000_000) -> None:
        """Devnet-only convenience — free SOL, no faucet account needed (unlike Amoy)."""
        resp = await self._client.request_airdrop(self._keypair.pubkey(), lamports)
        await self._client.confirm_transaction(resp.value, commitment=Confirmed)

    async def anchor(self, digest: bytes, cid: str) -> AnchorReceipt:
        memo_ix = Instruction(
            program_id=MEMO_PROGRAM_ID,
            accounts=[AccountMeta(pubkey=self._keypair.pubkey(), is_signer=True, is_writable=False)],
            data=_memo_payload(digest, cid),
        )
        blockhash_resp = await self._client.get_latest_blockhash()
        recent_blockhash: Hash = blockhash_resp.value.blockhash
        message = Message.new_with_blockhash([memo_ix], self._keypair.pubkey(), recent_blockhash)
        tx = Transaction([self._keypair], message, recent_blockhash)

        resp = await self._client.send_transaction(tx)
        return AnchorReceipt(tx_hash=str(resp.value))

    async def wait_for_inclusion(self, receipt: AnchorReceipt) -> None:
        await self._client.confirm_transaction(Signature.from_string(receipt.tx_hash), commitment=Confirmed)

    async def verify(self, digest: bytes) -> VerifyResult:
        needle = digest.hex()
        sigs_resp = await self._client.get_signatures_for_address(self._keypair.pubkey(), limit=VERIFY_SCAN_LIMIT)
        for sig_info in sigs_resp.value:
            tx_resp = await self._client.get_transaction(sig_info.signature, max_supported_transaction_version=0)
            if tx_resp.value is None:
                continue
            log_messages = tx_resp.value.transaction.meta.log_messages or []
            if any(needle in log for log in log_messages):
                block_time = tx_resp.value.block_time or 0
                return VerifyResult(ok=True, timestamp=block_time)
        return VerifyResult(ok=False, timestamp=0)

    async def close(self) -> None:
        await self._client.close()
