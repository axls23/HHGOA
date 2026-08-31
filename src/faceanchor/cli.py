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
from faceanchor.evidence.platform_check import check_platform_mutation
from faceanchor.search import fanout, score
from faceanchor.search.models import ScoredMatch
from faceanchor.vision import align, detect, embed, phash, quality
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
    seed_query: str | None = typer.Option(None, "--seed-query", help="Explicit Arm B (DuckDuckGo) seed; otherwise derived from Arm A's best hit"),
    chain: str = typer.Option("anvil", "--chain", help="anvil | amoy"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
    face_index: int | None = typer.Option(None, "--face-index", help="Disambiguate when multiple faces are detected"),
    zk: bool = typer.Option(False, "--zk", help="Attach a Groth16 match proof (PRD §7.1 L1) — bundle keeps Poseidon commitments only, never the raw embeddings"),
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
            candidates = await fanout.full_fanout(client, query, seed_query=seed_query)
            typer.echo(f"Stage 2: {len(candidates)} candidates ({query!r}, seed_query={seed_query!r})")
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
        if zk:
            from faceanchor.search.score import COSINE_ACCEPT_THRESHOLD
            from faceanchor.zk.prove import generate_proof, verify_proof
            from faceanchor.zk.witness import build_circuit_input

            raw_bytes = next(b for c, b in fetched if c.image_url == best.candidate.image_url)
            cand_image = decode_jpeg_bytes(raw_bytes)
            cand_faces = detect.detect_faces(cand_image)
            cand_face = max(cand_faces, key=lambda f: f.score)
            cand_crop = align.align_and_crop(cand_image, cand_face)
            cand_embedding = embed.embed_one(cand_crop)

            circuit_input = build_circuit_input(probe_ctx.embedding, cand_embedding, COSINE_ACCEPT_THRESHOLD)
            proof = generate_proof(circuit_input)
            if not verify_proof(proof):
                typer.echo("ERROR: generated ZK proof failed self-verification", err=True)
                raise typer.Exit(1)
            bundle["zk"] = {"proof": proof.proof, "public_signals": proof.public_signals}
            typer.echo("Stage 3: ZK match proof attached (192-byte Groth16 proof; no raw embedding in bundle)")

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
    check_platform: bool = typer.Option(False, "--check-platform", help="Re-fetch the matched post and detect platform-side mutation (PRD §7.2)"),
):
    """Re-derive the digest from the bundle and check it against on-chain state."""
    data = json.loads(bundle.read_text())
    digest = bundle_digest(data)
    consent_digest_hex = data.get("probe", {}).get("consent_digest")

    async def _verify():
        adapter = _resolve_chain(chain, contract_address)
        result = await adapter.verify(digest)

        typer.echo(f"digest={digest.hex()}")
        if result.ok:
            typer.echo(f"OK — anchored at timestamp {result.timestamp}")
        else:
            typer.echo("FAIL — digest not found on-chain (bundle was altered, or never anchored)")

        if consent_digest_hex:
            try:
                consent_result = await adapter.consent_status(bytes.fromhex(consent_digest_hex))
                if consent_result.ok:
                    typer.echo(f"⚠ CONSENT REVOKED @ timestamp {consent_result.timestamp}")
            except NotImplementedError:
                pass  # chain doesn't support revocation (e.g. Solana) — nothing to report

        if check_platform:
            async with fanout.make_client() as client:
                mutation_result = await check_platform_mutation(client, data, digest_verified=result.ok)
            typer.echo(f"platform check: {mutation_result.verdict.value} — {mutation_result.detail}")

        return result.ok

    ok = asyncio.run(_verify())
    if not ok:
        raise typer.Exit(1)


@app.command()
def revoke(
    consent: Path = typer.Option(..., "--consent", exists=True),
    chain: str = typer.Option("anvil", "--chain"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
):
    """Revoke a previously-anchored consent artifact (PRD §3/§7.3). Irreversible."""
    consent_digest_hex = load_consent_digest(consent)
    adapter = _resolve_chain(chain, contract_address)

    async def _revoke():
        receipt = await adapter.revoke(bytes.fromhex(consent_digest_hex))
        typer.echo(f"revoke tx submitted: {receipt.tx_hash}")
        await adapter.wait_for_inclusion(receipt)
        typer.echo(f"consent_digest={consent_digest_hex} REVOKED")

    asyncio.run(_revoke())


@app.command(name="forge-score")
def forge_score(
    bundle: Path = typer.Option(..., "--bundle", exists=True),
    score: float = typer.Option(..., "--score", help="A claimed score to forge into the bundle's public signals"),
    chain: str = typer.Option("anvil", "--chain"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
):
    """Demo command (PRD §12): attempts to anchor a bundle whose score was
    edited to `--score` while reusing an existing (now-mismatched) ZK proof.
    Expected outcome is an on-chain revert — this proves the chain enforces
    the proof rather than trusting the claimed score."""
    data = json.loads(bundle.read_text())
    zk_data = data.get("zk")
    if not zk_data:
        typer.echo("ERROR: bundle has no zk proof — run `faceanchor run --zk` first", err=True)
        raise typer.Exit(1)

    from faceanchor.zk.witness import COSINE_TO_DOT_SCALE

    forged = json.loads(json.dumps(data))  # deep copy
    forged["score"]["value"] = score
    forged_public_signals = list(zk_data["public_signals"])
    forged_public_signals[2] = str(int(round(score * COSINE_TO_DOT_SCALE)))

    digest = bundle_digest(forged)
    cid = compute_cid(canonicalize(forged))
    adapter = _resolve_chain(chain, contract_address)

    async def _forge():
        try:
            receipt = await adapter.anchor_with_proof(digest, cid, zk_data["proof"], forged_public_signals)
            await adapter.wait_for_inclusion(receipt)
            typer.echo(f"UNEXPECTED: chain accepted the forged score (tx {receipt.tx_hash})")
            raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception as e:
            typer.echo(f"chain REJECTS: {e}")

    asyncio.run(_forge())


if __name__ == "__main__":
    app()
