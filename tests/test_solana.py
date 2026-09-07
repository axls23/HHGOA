import pytest

from faceanchor.chain.solana import _memo_payload


def test_memo_payload_encodes_digest_and_cid():
    digest = bytes.fromhex("aa" * 32)
    payload = _memo_payload(digest, "bafyreitest")
    assert payload == b"faceanchor:" + b"aa" * 32 + b":bafyreitest"


def test_memo_payload_is_deterministic():
    digest = bytes.fromhex("bb" * 32)
    assert _memo_payload(digest, "cid-a") == _memo_payload(digest, "cid-a")
    assert _memo_payload(digest, "cid-a") != _memo_payload(digest, "cid-b")


# --- public-footprint helpers (development stays local: anvil must never
# --- claim a public record it does not have) ---


def test_anvil_has_no_explorer():
    """The safety property: a local devnet must not be presentable as a public
    footprint. Absence here is what makes the CLI say 'none' instead of
    fabricating a link."""
    from faceanchor.chain import explorer

    assert explorer.for_chain("anvil") is None
    assert explorer.for_chain("unknown-chain") is None


def test_amoy_explorer_urls():
    from faceanchor.chain import explorer

    site = explorer.for_chain("amoy")
    assert site is not None
    assert site.tx("abc123") == "https://amoy.polygonscan.com/tx/0xabc123"
    assert site.tx("0xabc123") == "https://amoy.polygonscan.com/tx/0xabc123"
    assert site.address("0xdeadbeef") == "https://amoy.polygonscan.com/address/0xdeadbeef"
    assert site.read_contract("0xdeadbeef").endswith("#readContract")


def test_solana_explorer_is_public_but_devnet_scoped():
    """Devnet is the opposite case to anvil: there *is* a public record, so
    refusing to link it would be its own false signal. But a base58 signature
    is not 0x-hex, and SPL Memo has no contract to read."""
    from faceanchor.chain import explorer

    site = explorer.for_chain("solana")
    assert site is not None
    assert site.tx("5xAbC") == "https://explorer.solana.com/tx/5xAbC?cluster=devnet"
    assert site.address("MemoPubkey") == "https://explorer.solana.com/address/MemoPubkey?cluster=devnet"
    assert site.read_contract("MemoPubkey") is None


# --- CLI wiring: the capability matrix in the README has to be reachable ---


def _write_keypair(tmp_path):
    import json

    from solders.keypair import Keypair

    path = tmp_path / "id.json"
    keypair = Keypair()
    path.write_text(json.dumps(list(bytes(keypair))))
    return path, keypair


def test_resolve_chain_builds_a_solana_adapter(tmp_path, monkeypatch):
    from faceanchor.chain.solana import SolanaAdapter
    from faceanchor.cli import _resolve_chain

    path, keypair = _write_keypair(tmp_path)
    monkeypatch.setenv("SOLANA_KEYPAIR", str(path))
    monkeypatch.delenv("SOLANA_RPC_URL", raising=False)

    adapter = _resolve_chain("solana", contract_address=None)  # SPL Memo needs no program
    assert isinstance(adapter, SolanaAdapter)
    assert adapter._keypair.pubkey() == keypair.pubkey()


def test_resolve_chain_solana_requires_a_keypair(tmp_path, monkeypatch):
    import typer

    from faceanchor.cli import _resolve_chain

    monkeypatch.delenv("SOLANA_KEYPAIR", raising=False)
    with pytest.raises(typer.BadParameter, match="SOLANA_KEYPAIR"):
        _resolve_chain("solana", contract_address=None)

    monkeypatch.setenv("SOLANA_KEYPAIR", str(tmp_path / "nope.json"))
    with pytest.raises(typer.BadParameter, match="could not load"):
        _resolve_chain("solana", contract_address=None)


def test_chain_capability_matrix_matches_the_adapters():
    """README's per-operation table, asserted. The EVM-only rows are EVM-only
    because SPL Memo keeps no state and runs no program — so the base class's
    NotImplementedError is the correct behaviour, not a gap to fill."""
    from faceanchor.chain.anvil import AnvilAdapter
    from faceanchor.chain.base import ChainAdapter
    from faceanchor.chain.evm import EvmAdapter
    from faceanchor.chain.solana import SolanaAdapter

    for op in ("revoke", "consent_status", "anchor_with_proof"):
        assert getattr(SolanaAdapter, op) is getattr(ChainAdapter, op), f"solana should not implement {op}"
        assert getattr(EvmAdapter, op) is not getattr(ChainAdapter, op), f"evm must implement {op}"
        assert getattr(AnvilAdapter, op) is not getattr(ChainAdapter, op), f"anvil must implement {op}"

    for op in ("anchor", "verify", "wait_for_inclusion"):
        assert getattr(SolanaAdapter, op) is not getattr(ChainAdapter, op), f"solana must implement {op}"


def test_zk_is_refused_before_any_work_on_a_chain_that_cannot_verify(tmp_path, monkeypatch):
    """`--zk` means the chain checks the proof (§7.1 L2). On a chain with no
    verifier that is not a degraded run, it is a different claim — so it is
    refused up front, before the probe is even decoded, rather than after a
    full search and ~2.7s of proving."""
    from typer.testing import CliRunner

    from faceanchor.cli import app

    path, _ = _write_keypair(tmp_path)
    monkeypatch.setenv("SOLANA_KEYPAIR", str(path))

    probe = tmp_path / "probe.jpg"
    probe.write_bytes(b"not a real jpeg: decoding this would raise, and we never get that far")

    result = CliRunner().invoke(
        app,
        ["run", "--probe", str(probe), "--subject-consent", "consent/self.example.json",
         "--query", "anything", "--chain", "solana", "--zk"],
    )
    assert result.exit_code == 1
    assert "--zk" in result.output + (result.stderr if result.stderr_bytes is not None else "")
