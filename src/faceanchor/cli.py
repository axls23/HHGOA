from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import typer

from faceanchor.chain.anvil import AnvilAdapter
from faceanchor.chain.base import ChainAdapter
from faceanchor.chain.evm import EvmAdapter
from faceanchor.consent import MissingConsentError, load_consent_digest
from faceanchor.evidence.bundle import ProbeInfo, build_bundle, bundle_digest
from faceanchor.evidence.cid import compute_cid
from faceanchor.evidence.jcs import canonicalize
from faceanchor.search import fanout, score
from faceanchor.search.models import ScoredMatch
from faceanchor.vision import detect, phash, quality
from faceanchor.vision.decode import decode_jpeg_bytes

app = typer.Typer(help="face-anchor: probe -> discovery -> on-chain verification")

EVIDENCE_DIR = Path("evidence")


def _resolve_chain(chain: str, contract_address: str | None) -> ChainAdapter:
    if chain == "anvil":
        if not contract_address:
            raise typer.BadParameter("--contract-address is required for --chain anvil (deploy first: see contracts/script/Deploy.s.sol)")
        return AnvilAdapter(contract_address=contract_address)
    if chain == "amoy":
        import os

        rpc_url = os.environ.get("AMOY_RPC_URL")
        private_key = os.environ.get("AMOY_PRIVATE_KEY")
        if not (rpc_url and private_key and contract_address):
            raise typer.BadParameter(
                "--chain amoy requires AMOY_RPC_URL and AMOY_PRIVATE_KEY env vars plus --contract-address"
            )
        return EvmAdapter(rpc_url=rpc_url, private_key=private_key, contract_address=contract_address, poa=True)
    raise typer.BadParameter(f"unsupported chain: {chain} (supported: anvil, amoy)")


@app.command()
def run(
    probe: Path = typer.Option(..., "--probe", exists=True, help="Path to the probe image"),
    subject_consent: Path | None = typer.Option(None, "--subject-consent", help="Required consent artifact (PRD §3)"),
    query: str = typer.Option(..., "--query", help="Discovery search query (author handle, name, or keywords)"),
    chain: str = typer.Option("anvil", "--chain", help="anvil | amoy"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
    face_index: int | None = typer.Option(None, "--face-index", help="Disambiguate when multiple faces are detected"),
):
    """Stage 1 (vision) -> Stage 2 (discovery) -> Stage 3 (anchor)."""
    try:
        consent_digest = load_consent_digest(subject_consent)
    except MissingConsentError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    probe_bytes = probe.read_bytes()
    probe_image = decode_jpeg_bytes(probe_bytes)
    faces = detect.detect_faces(probe_image)
    face, err = quality.check_single_face(faces, face_index)
    if err:
        typer.echo(f"ERROR: {err}", err=True)
        raise typer.Exit(1)

    quality_result = quality.check_quality(probe_image, face)
    if not quality_result.ok:
        typer.echo(f"ERROR: quality gate failed: {quality_result.reason}", err=True)
        raise typer.Exit(1)

    probe_ctx = score.build_probe_context(probe_image, face)
    typer.echo(f"Stage 1: face detected (score={face.score:.3f}), quality gate passed")

    async def _discover_and_anchor():
        async with fanout.make_client() as client:
            candidates = await fanout.arm_a_fanout(client, query)
            typer.echo(f"Stage 2: {len(candidates)} candidates from Arm A ({query!r})")
            fetched = await fanout.fetch_all_candidates(client, candidates)
            matches = score.score_candidates(probe_ctx, fetched)

        if not matches:
            typer.echo("No match found above threshold.", err=True)
            raise typer.Exit(1)

        best: ScoredMatch = matches[0]
        typer.echo(f"Stage 2: best match {best.metric}={best.score:.4f} platform={best.candidate.platform} uri={best.candidate.post_uri}")

        probe_info = ProbeInfo(
            image_sha256=hashlib.sha256(probe_bytes).hexdigest(),
            image_phash=probe_ctx.phash.hex(),
            embed_model="sface_2021dec",
            embed_sha256=hashlib.sha256(probe_ctx.embedding.tobytes()).hexdigest(),
            consent_digest=consent_digest,
        )
        bundle = build_bundle(
            pipeline_commit="dev",
            probe=probe_info,
            match=best,
            match_text_sha256=hashlib.sha256((best.candidate.text or "").encode()).hexdigest(),
            match_image_url_sha256=hashlib.sha256(best.candidate.image_url.encode()).hexdigest(),
        )
        canonical = canonicalize(bundle)
        digest = bundle_digest(bundle)
        cid = compute_cid(canonical)

        adapter = _resolve_chain(chain, contract_address)
        receipt = await adapter.anchor(digest, cid)
        typer.echo(f"Stage 3: tx submitted {receipt.tx_hash}")
        await adapter.wait_for_inclusion(receipt)
        typer.echo("Stage 3: included on-chain")

        EVIDENCE_DIR.mkdir(exist_ok=True)
        out_path = EVIDENCE_DIR / f"{bundle['run']['run_id']}.json"
        out_path.write_text(json.dumps(bundle, indent=2))
        (EVIDENCE_DIR / "latest.json").write_text(json.dumps(bundle, indent=2))
        typer.echo(f"Evidence bundle written: {out_path}")
        typer.echo(f"digest={digest.hex()} cid={cid}")

    asyncio.run(_discover_and_anchor())


@app.command()
def verify(
    bundle: Path = typer.Option(..., "--bundle", exists=True),
    chain: str = typer.Option("anvil", "--chain"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
):
    """Re-derive the digest from the bundle and check it against on-chain state."""
    data = json.loads(bundle.read_text())
    digest = bundle_digest(data)

    adapter = _resolve_chain(chain, contract_address)
    result = asyncio.run(adapter.verify(digest))

    typer.echo(f"digest={digest.hex()}")
    if result.ok:
        typer.echo(f"OK — anchored at timestamp {result.timestamp}")
    else:
        typer.echo("FAIL — digest not found on-chain (bundle was altered, or never anchored)")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
