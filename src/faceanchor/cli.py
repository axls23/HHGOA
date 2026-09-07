from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import typer

from faceanchor.chain import explorer
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
from faceanchor.search.safety import SafetyReport
from faceanchor.vision import align, detect, embed, phash, quality
from faceanchor.vision.decode import decode_jpeg_bytes

app = typer.Typer(help="face-anchor: probe -> discovery -> on-chain verification")

EVIDENCE_DIR = Path("evidence")

# Distinct exit codes, because "the search found nothing" and "the search
# broke" are opposite facts about the subject and a caller must be able to
# tell them apart without parsing prose (PRD §5.2; see web_reverse/base.py).
EXIT_NO_RESULTS = 1
EXIT_PROVIDER_ERROR = 2

DISCOVERY_MODES = ("text", "by-face", "web-reverse")


# Query params that carry a credential rather than an identifier. Meta's CDNs
# sign every media URL (`oh`, `oe`, `_nc_ohc`…), and a signed URL is a bearer
# token with an expiry, not a name — writing one to a log file persists an
# access grant. Mastodon/Bluesky URLs carry none of these, so redaction costs
# nothing today and is a precondition for ever pointing this at Meta.
_SIGNED_PARAMS = frozenset(
    {
        "oh", "oe", "_nc_ohc", "_nc_sid", "_nc_gid", "_nc_ht", "ccb", "efg", "stp",
        "sig", "signature", "token", "access_token", "sp", "st",
        "x-amz-signature", "x-amz-credential", "x-amz-security-token", "expires",
    }
)


def redact_url(url: str) -> str:
    """Strips credential-bearing query params, keeping the URL identifying.

    The log's job is to say *which* image was looked at, which the path alone
    does. The signature only says *that we were allowed to*, and that is the
    part with a lifetime and a blast radius.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    if not parts.query:
        return url
    kept, redacted = [], False
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in _SIGNED_PARAMS:
            redacted = True
            continue
        kept.append((key, value))
    if not redacted:
        return url
    query = urlencode(kept)
    if query:
        query += "&"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query + "redacted=signature", parts.fragment))


def _write_candidate_log(
    *,
    path: Path,
    probe_path: Path,
    probe_sha256: str,
    probe_phash: str,
    discovery: dict,
    source_reports: list,
    fetch_trace: list[dict],
    score_trace: list[dict],
) -> int:
    """Development log: every URL the run looked at, and what became of it.

    JSONL, one header line then one line per candidate URL, each carrying the
    probe's own sha256/phash so a log can never be read against the wrong
    image. Fetch outcome and cascade verdict are merged per URL, so a single
    line answers "was it fetched, did it decode, did it have a face, what did
    it score".

    Deliberately *not* part of the evidence bundle: it names third-party image
    URLs the pipeline merely looked at and did not match, which is exactly the
    kind of collateral that should not be anchored to an immutable ledger
    (PRD §6.3). It is a debugging artifact and lives under evidence/, which is
    gitignored.
    """
    by_url: dict[str, dict] = {}
    for record in fetch_trace:
        by_url[record["image_url"]] = dict(record)
    for record in score_trace:
        by_url.setdefault(record["image_url"], dict(record)).update(record)
    for record in by_url.values():
        record["image_url"] = redact_url(record["image_url"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "faceanchor.candidate_log",
                    "probe": {"path": str(probe_path), "sha256": probe_sha256, "phash": probe_phash},
                    "discovery": discovery,
                    "sources": [
                        {"name": r.name, "count": r.count, "elapsed_ms": round(r.elapsed_ms, 1), "error": r.error}
                        for r in source_reports
                    ],
                    "candidates_seen": len(by_url),
                }
            )
            + "\n"
        )
        for record in by_url.values():
            record.setdefault("verdict", "not_scored")
            fh.write(json.dumps(record) + "\n")
    return len(by_url)


def _describe_gate(quality_result) -> str:
    """What Stage 1 measured, in the order it decided.

    The pose and learned-quality numbers are printed rather than summarised
    because "quality gate passed" alone gives a reader no way to tell a probe
    that scraped through from one that sailed through — and because a missing
    optional model has to be visible, not inferred from a number that stopped
    appearing.
    """
    parts = [f"face {quality_result.bbox_min_side:.0f}px"]
    pose_result = quality_result.head_pose
    if pose_result is not None:
        parts.append(f"pose yaw {pose_result.yaw:+.0f} pitch {pose_result.pitch:+.0f} roll {pose_result.roll:+.0f}")
    else:
        parts.append(f"pose unsolved, asymmetry {quality_result.landmark_asymmetry:.2f}")
    if quality_result.fiqa is not None:
        parts.append(f"quality {quality_result.fiqa:.2f}")
    else:
        parts.append("quality model not installed (models/fetch.sh) — heuristics only")
    return ", ".join(parts)


def _echo_explorer(chain: str, contract_address: str | None, tx_hash: str | None = None) -> None:
    """Point at the public record, or say plainly that there isn't one."""
    site = explorer.for_chain(chain)
    if site is None:
        typer.echo(
            f"Public footprint: none — {chain} is a local devnet. It exists on this machine only "
            "and dies with the node; nobody can check it on the web. Use --chain amoy for a public anchor."
        )
        return
    if tx_hash:
        typer.echo(f"Public footprint: {site.tx(tx_hash)}")
    if contract_address:
        link = site.read_contract(contract_address)
        if link:
            typer.echo(f"Anyone can re-check it here: {link} -> verify(bytes32)")
    elif site.readable_contract is False:
        typer.echo("The memo is in the transaction itself — there is no contract state to query on this chain.")


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
        # Loud on purpose. Everything else in this repo defaults to a throwaway
        # local chain; this is the one branch that writes a permanent public
        # record, and an anchor cannot be un-anchored (only revoked, §7.3).
        typer.echo(
            "WARNING: --chain amoy writes a PERMANENT PUBLIC record to Polygon Amoy. "
            "The digest cannot be removed afterwards — only flagged revoked. Use --chain anvil for development.",
            err=True,
        )
        return EvmAdapter(rpc_url=rpc_url, private_key=private_key, contract_address=contract_address, poa=True)
    if chain == "solana":
        import os

        from solders.keypair import Keypair

        from faceanchor.chain.solana import DEVNET_RPC_URL, SolanaAdapter

        keypair_path = os.environ.get("SOLANA_KEYPAIR")
        if not keypair_path:
            raise typer.BadParameter(
                "--chain solana requires SOLANA_KEYPAIR (path to a solana-cli keypair JSON — "
                "`solana-keygen new -o ~/.config/solana/id.json`). SOLANA_RPC_URL is optional "
                f"(default {DEVNET_RPC_URL}). --contract-address is not used: SPL Memo needs no program."
            )
        try:
            keypair = Keypair.from_bytes(bytes(json.loads(Path(keypair_path).read_text())))
        except (OSError, ValueError, TypeError) as e:
            raise typer.BadParameter(f"could not load SOLANA_KEYPAIR from {keypair_path}: {type(e).__name__}: {e}")
        # Public but not permanent, and the difference matters to whoever is
        # relying on the anchor — so it is said out loud, like amoy's warning,
        # rather than left for them to discover after a cluster reset.
        typer.echo(
            "NOTE: --chain solana targets Solana devnet. A record there is public and anyone can check it, "
            "but the cluster is periodically reset — it is not a permanent anchor. Use --chain amoy for that.",
            err=True,
        )
        return SolanaAdapter(keypair=keypair, rpc_url=os.environ.get("SOLANA_RPC_URL") or DEVNET_RPC_URL)
    raise typer.BadParameter(f"unsupported chain: {chain} (supported: anvil, amoy, solana)")


async def _close_adapter(adapter: ChainAdapter) -> None:
    """Solana's adapter owns an httpx client; the EVM ones do not. Closing is
    the adapter's own business, so ask rather than assume one exists."""
    closer = getattr(adapter, "close", None)
    if closer is not None:
        await closer()


def _resolve_discovery_mode(discovery: str | None, query: str | None, by_face: bool) -> str:
    """Which of the three discovery arms this invocation asked for.

    They are genuinely different claims, so exactly one runs per invocation
    and the bundle records which (`run.discovery.mode`):

        text_seeded   --query      words about the subject point the search
        by_face       --by-face    the probe's face queries a corpus you named
        web_reverse   --discovery web-reverse
                                   the probe image itself goes to a
                                   reverse-image provider indexing the open web

    `--query` and `--by-face` keep their original meaning and their original
    mutual exclusivity; `--discovery` is the additive way to reach the third
    arm, and also names the first two for callers that prefer one flag.
    """
    if discovery is None:
        if by_face == bool(query):
            raise typer.BadParameter(
                "pass exactly one of --query (text-seeded discovery), --by-face "
                "(face-embedding search over --corpus), or --discovery web-reverse "
                "(genuine image-based reverse-image search)"
            )
        return "by_face" if by_face else "text_seeded"

    normalized = discovery.strip().lower().replace("_", "-")
    if normalized not in DISCOVERY_MODES:
        raise typer.BadParameter(f"unknown --discovery {discovery!r} — supported: {', '.join(DISCOVERY_MODES)}")

    if normalized == "web-reverse":
        if query or by_face:
            raise typer.BadParameter(
                "--discovery web-reverse is its own discovery arm — drop --query/--by-face. "
                "It submits the probe image to a reverse-image provider; --by-face searches a "
                "named corpus with the face embedding, and the two are not the same operation."
            )
        return "web_reverse"
    if normalized == "text":
        if not query:
            raise typer.BadParameter("--discovery text needs --query")
        return "text_seeded"
    if not by_face and not query:
        by_face = True
    if query:
        raise typer.BadParameter("--discovery by-face is incompatible with --query")
    return "by_face"


def _confirm_external_upload(provider_id: str, accepted: bool) -> None:
    """The probe image leaves this machine. Say so, and get a yes first.

    Every other arm of this pipeline sends *hashes and words* outward; this
    one sends the photograph of a person's face to a third party whose
    retention and logging nobody here controls. The consent artifact covers
    using the subject's face in this pipeline — it does not cover handing it
    to Google, so that is a second, explicit gate rather than a footnote.
    """
    import sys

    typer.echo(
        f"NOTICE: reverse-image search uploads the PROBE IMAGE ITSELF to an external provider ({provider_id}).\n"
        "        This is a trust boundary no other discovery mode crosses: the photograph leaves this\n"
        "        machine, and the provider's retention, logging and further use are outside this repo's\n"
        "        control. The consent artifact covers the subject; it does not cover this transfer."
    )
    if accepted:
        typer.echo("        --accept-external-upload given — proceeding.")
        return
    if not sys.stdin.isatty():
        typer.echo(
            "ERROR: refusing to upload the probe without an explicit acknowledgement. "
            "Re-run with --accept-external-upload.",
            err=True,
        )
        raise typer.Exit(1)
    if not typer.confirm("        Send the probe image to this provider?", default=False):
        typer.echo("Aborted — nothing was uploaded, searched or anchored.", err=True)
        raise typer.Exit(1)


@app.command()
def run(
    probe: Path = typer.Option(..., "--probe", exists=True, help="Path to the probe image"),
    subject_consent: Path | None = typer.Option(None, "--subject-consent", help="Required consent artifact (PRD §3)"),
    query: str | None = typer.Option(None, "--query", help="Text-seeded discovery: author handle, name, or keywords. Mutually exclusive with --by-face"),
    by_face: bool = typer.Option(False, "--by-face", help="Face-embedding search: ingest a corpus you name and query it with the probe's face alone, no text about the subject. NOT reverse-image search — see --discovery web-reverse for that"),
    discovery: str | None = typer.Option(None, "--discovery", help="Discovery arm: text (same as --query) | by-face (same as --by-face) | web-reverse (genuine image-based reverse-image search — submits the probe image to an external provider)"),
    provider: str = typer.Option("google", "--provider", help="Reverse-image provider for --discovery web-reverse: google (Cloud Vision Web Detection) | mock (explicit replay double, needs FACEANCHOR_MOCK_REVERSE_RESPONSE)"),
    social_domain: list[str] = typer.Option([], "--social-domain", help="Extra platform:domain mapping for classifying reverse-image results, repeatable, e.g. mastodon:example.social. Also read from FACEANCHOR_SOCIAL_DOMAINS"),
    accept_external_upload: bool = typer.Option(False, "--accept-external-upload", help="Acknowledge that --discovery web-reverse sends the probe image to a third-party provider. Required when stdin is not a TTY"),
    allow_sensitive: bool = typer.Option(False, "--allow-sensitive", help="Do NOT filter out posts the poster flagged sensitive (Mastodon CW) or labelled explicit (Bluesky). Off by default: public timelines carry explicit content, and an unfiltered run downloads it, logs its URLs, and inlines it into any --contact-sheet"),
    corpus: list[str] = typer.Option([], "--corpus", help="Corpus to search with --by-face, repeatable: mastodon:tag/<tag>[@instance] | mastodon:account/<handle> | bsky:feed/<alias|at-uri> | bsky:author/<handle>"),
    corpus_pages: int = typer.Option(3, "--corpus-pages", help="Pages to page back per corpus (~40-50 images/page)"),
    corpus_max: int = typer.Option(600, "--corpus-max", help="Hard cap on ingested images per run"),
    corpus_db: Path | None = typer.Option(None, "--corpus-db", help="Replay a local corpus database built by `corpus-build` instead of fetching live timelines. Offline, deterministic, and nothing explicit is re-downloaded — the safety filter already ran at build time"),
    seed_query: str | None = typer.Option(None, "--seed-query", help="Explicit Arm B (DuckDuckGo) seed; otherwise derived from Arm A's best hit"),
    chain: str = typer.Option("anvil", "--chain", help="anvil | amoy | solana"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
    face_index: int | None = typer.Option(None, "--face-index", help="Disambiguate when multiple faces are detected"),
    zk: bool = typer.Option(False, "--zk", help="Attach a Groth16 match proof (PRD §7.1 L1) — bundle keeps Poseidon commitments only, never the raw embeddings"),
    log_candidates: Path | None = typer.Option(None, "--log-candidates", help="Development: write every searched URL and its verdict to this JSONL file, keyed to the probe. Never anchored, never part of the bundle."),
    contact_sheet: Path | None = typer.Option(None, "--contact-sheet", help="Development: render probe + every fetched candidate as a self-contained HTML page, images inlined, ordered by score. Writes third-party faces to disk; never anchored."),
):
    """Stage 1 (vision) -> Stage 2 (discovery) -> Stage 3 (anchor)."""
    try:
        mode = _resolve_discovery_mode(discovery, query, by_face)
    except typer.BadParameter as e:
        typer.echo(f"ERROR: {e.message}", err=True)
        raise typer.Exit(1)
    by_face = mode == "by_face"

    if by_face and not corpus and corpus_db is None:
        typer.echo(
            "ERROR: --by-face needs at least one --corpus to search (e.g. --corpus mastodon:tag/selfie "
            "--corpus bsky:feed/whats-hot), or --corpus-db <dir> to replay a locally built corpus",
            err=True,
        )
        raise typer.Exit(1)
    if corpus_db is not None and not by_face:
        typer.echo("ERROR: --corpus-db only applies to --by-face discovery", err=True)
        raise typer.Exit(1)

    reverse_provider = None
    matcher = None
    if mode == "web_reverse":
        from faceanchor.search import web_reverse

        try:
            reverse_provider = web_reverse.get_provider(provider)
            matcher = web_reverse.PlatformMatcher.build(social_domain)
        except (web_reverse.ProviderConfigError, ValueError) as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(1)
        if reverse_provider.name not in web_reverse.REAL_PROVIDERS:
            typer.echo(
                f"WARNING: --provider {reverse_provider.name} is a test double. Results are replayed from a file, "
                f"not obtained from a live reverse-image index. The evidence bundle will record the provider as "
                f"{reverse_provider.provider_id!r} so the record says so permanently.",
                err=True,
            )

    specs = []
    if by_face and corpus:
        from faceanchor.search import corpus as corpus_mod

        try:
            specs = [corpus_mod.parse_spec(raw) for raw in corpus]
        except corpus_mod.CorpusSpecError as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(1)

    try:
        consent_digest = load_consent_digest(subject_consent)
    except MissingConsentError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    # Resolved here, before Stage 1 rather than at anchor time: a missing
    # --contract-address or an unusable key is a fact about the invocation, and
    # learning it after a full search (and, with --zk, ~2.7s of proving) is a
    # waste of the run. It stays after the consent gate — no consent, no chain.
    adapter = _resolve_chain(chain, contract_address)
    if zk and type(adapter).anchor_with_proof is ChainAdapter.anchor_with_proof:
        typer.echo(
            f"ERROR: --zk anchors through the on-chain verifier (PRD §7.1 L2), which {chain} "
            "does not implement — SPL Memo runs no program to verify a proof with. "
            "Use --chain anvil|amoy, or drop --zk.",
            err=True,
        )
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
    typer.echo(f"Stage 1: face detected (score={face.score:.3f}), quality gate passed — {_describe_gate(quality_result)}")

    # After Stage 1, before anything leaves the machine: a probe that fails
    # the quality gate is rejected without ever being uploaded.
    if mode == "web_reverse":
        _confirm_external_upload(reverse_provider.provider_id, accept_external_upload)

    async def _discover_and_anchor():
        async with fanout.make_client() as client:
            if mode == "web_reverse":
                import time as _time

                from faceanchor.search import web_reverse
                from faceanchor.search.fanout import SourceReport

                typer.echo(
                    f"Stage 2: reverse-image search — submitting the probe image to {reverse_provider.provider_id}"
                )
                started = _time.monotonic()
                try:
                    results = await reverse_provider.search(probe, client=client)
                except web_reverse.ReverseImageError as e:
                    # PROVIDER_ERROR, never NO_RESULTS: the provider did not
                    # answer, so this run learned nothing about the subject.
                    # Exiting 1 here would let a broken key read as "this face
                    # is nowhere on the web", which is the worst lie this tool
                    # could tell.
                    typer.echo(f"ERROR: PROVIDER_ERROR — {type(e).__name__}: {e}", err=True)
                    raise typer.Exit(EXIT_PROVIDER_ERROR)
                elapsed_ms = (_time.monotonic() - started) * 1000

                social = web_reverse.filter_social(results, matcher)
                candidates = social.candidates
                degraded = False
                # The provider hands back bare URLs with no post metadata, so
                # the per-post sensitivity flags the other two arms read do
                # not exist here. `filter_social` drops adult instances by
                # host instead; see README "Filtering explicit content".
                safety = SafetyReport()
                reports = [
                    SourceReport(reverse_provider.provider_id, len(results), elapsed_ms),
                    SourceReport(f"{reverse_provider.provider_id}:social", len(candidates), 0.0),
                ]
                typer.echo(f"Stage 2: reverse-image results — {len(results)} web images/pages in {elapsed_ms:.0f}ms")
                typer.echo(
                    f"Stage 2: social candidates — {len(candidates)} ({social.summary()}); "
                    f"skipped {social.skip_summary()}"
                )
                if not results:
                    # The provider answered, and the answer was nothing. That
                    # is a finding, and it is reported as one.
                    typer.echo(
                        "Stage 2: NO_RESULTS — the provider found no matching images or pages for this photo.",
                        err=True,
                    )
                    raise typer.Exit(EXIT_NO_RESULTS)
                if not candidates:
                    typer.echo(
                        f"Stage 2: NO_RESULTS — {len(results)} web results, none of them on a known social "
                        "platform. Add domains with --social-domain platform:domain if the platform is one "
                        "this pipeline can handle.",
                        err=True,
                    )
                    raise typer.Exit(EXIT_NO_RESULTS)
                discovery_block = {
                    "mode": "web_reverse",
                    "provider": reverse_provider.provider_id,
                    "provider_results": len(results),
                    "social_candidates": len(candidates),
                    "platforms": sorted(social.by_platform),
                }
            elif by_face and corpus_db is not None:
                from faceanchor.search.corpus_db import CorpusDb, CorpusDbError
                from faceanchor.search.fanout import SourceReport

                try:
                    db = CorpusDb(corpus_db).load()
                except CorpusDbError as e:
                    typer.echo(f"ERROR: {e}", err=True)
                    raise typer.Exit(1)

                prefetched = db.as_fetched()
                candidates = [c for c, _ in prefetched]
                reports = [SourceReport(f"corpus-db[{corpus_db}]", len(candidates), 0.0)]
                degraded = False
                safety = SafetyReport()
                typer.echo(
                    f"Stage 2: face-embedding search over a local corpus database — "
                    f"{len(candidates)} images from {corpus_db}"
                )
                typer.echo(
                    f"Stage 2: offline replay — nothing was fetched; the safety filter ran at build time "
                    f"(manifest {db.manifest_sha256()[:16]}…)"
                )
                discovery_block = {
                    "mode": "by_face",
                    "source": "corpus_db",
                    "corpus": sorted(db.header.get("corpora", [])),
                    "corpus_images": len(candidates),
                    # Pins the bundle to the exact corpus it was scored
                    # against — the replay equivalent of naming the timelines.
                    "corpus_db_manifest_sha256": db.manifest_sha256(),
                }
                if db.header.get("filtered_sensitive"):
                    discovery_block["filtered_sensitive"] = db.header["filtered_sensitive"].get("total", 0)
            elif by_face:
                from faceanchor.search import corpus as corpus_mod

                candidates, reports, safety = await corpus_mod.ingest(
                    client, specs, pages=corpus_pages, max_images=corpus_max, allow_sensitive=allow_sensitive
                )
                degraded = any(r.error for r in reports)
                summary = ", ".join(r.describe() for r in reports)
                typer.echo(
                    f"Stage 2: face-embedding search over a named corpus — {len(candidates)} images "
                    f"ingested from {len(specs)} corpora"
                )
                typer.echo(f"Stage 2: per-corpus — {summary}")
                discovery_block = {
                    "mode": "by_face",
                    "corpus": sorted(str(s) for s in specs),
                    "corpus_images": len(candidates),
                }
            else:
                result = await fanout.full_fanout(
                    client, query, seed_query=seed_query, allow_sensitive=allow_sensitive
                )
                candidates, reports, degraded = result.candidates, result.reports, result.degraded
                safety = result.safety
                typer.echo(f"Stage 2: {len(candidates)} candidates ({query!r}, seed_query={seed_query!r})")
                typer.echo(f"Stage 2: per-source — {result.summary()}")
                discovery_block = {
                    "mode": "text_seeded",
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                }

            # Never silent: a filter that quietly removes candidates is
            # indistinguishable from a search that found nothing.
            if safety.total:
                typer.echo(f"Stage 2: explicit-content filter removed {safety.total} — {safety.describe()}")
                discovery_block["filtered_sensitive"] = safety.total
            elif allow_sensitive:
                typer.echo(
                    "Stage 2: WARNING — --allow-sensitive is set; explicit posts are NOT filtered out",
                    err=True,
                )
            if allow_sensitive:
                discovery_block["allow_sensitive"] = True
            if degraded:
                typer.echo("Stage 2: WARNING — a source failed; recall below is bounded by the sources that answered", err=True)
            want_trace = log_candidates or contact_sheet
            fetch_trace: list[dict] | None = [] if want_trace else None
            score_trace: list[dict] | None = [] if want_trace else None
            if corpus_db is not None:
                # Already on disk, already sha256-verified by the loader —
                # re-fetching them over the network would defeat the point.
                fetched = prefetched
                if fetch_trace is not None:
                    fetch_trace.extend(
                        {
                            "platform": c.platform, "image_url": c.image_url, "post_uri": c.post_uri,
                            "author": c.author, "variant": c.extra.get("variant", "original"),
                            "fetch": "corpus_db", "status": None, "bytes": len(b),
                        }
                        for c, b in prefetched
                    )
                typer.echo(f"Stage 2: {len(fetched)} images read from the corpus database, verifying by face (SFace)")
            else:
                fetched = await fanout.fetch_all_candidates(client, candidates, trace=fetch_trace)
                typer.echo(f"Stage 2: {len(fetched)} images fetched, verifying by face (SFace)")
            matches = score.score_candidates(probe_ctx, fetched, trace=score_trace)

        if log_candidates is not None:
            written = _write_candidate_log(
                path=log_candidates,
                probe_path=probe,
                probe_sha256=hashlib.sha256(probe_bytes).hexdigest(),
                probe_phash=probe_ctx.phash.hex(),
                discovery=discovery_block,
                source_reports=reports,
                fetch_trace=fetch_trace or [],
                score_trace=score_trace or [],
            )
            typer.echo(f"Stage 2: candidate log written: {log_candidates} ({written} URLs)")

        if contact_sheet is not None:
            from faceanchor.review import write_contact_sheet
            from faceanchor.search.score import COSINE_ACCEPT_THRESHOLD

            shown = write_contact_sheet(
                path=contact_sheet,
                probe_bytes=probe_bytes,
                probe_label=probe.name,
                fetched=fetched,
                score_trace=score_trace or [],
                threshold=COSINE_ACCEPT_THRESHOLD,
                discovery=discovery_block,
            )
            typer.echo(f"Stage 2: contact sheet written: {contact_sheet} ({shown} images inlined) — open it in a browser")

        if not matches:
            # A reverse-image provider saying "this image is on that page" is
            # not the same claim as "that page shows this person", so a
            # provider hit that no face verifies against is a non-match, and
            # this is where it is refused. Nothing is anchored.
            typer.echo("No match found above threshold.", err=True)
            raise typer.Exit(EXIT_NO_RESULTS)

        best: ScoredMatch = matches[0]
        typer.echo(f"Stage 2: face verification — {len(matches)} candidate(s) above threshold")
        typer.echo(f"Stage 2: best match {best.metric}={best.score:.4f} platform={best.candidate.platform} uri={best.candidate.post_uri}")

        probe_info = ProbeInfo(
            image_sha256=hashlib.sha256(probe_bytes).hexdigest(),
            image_phash=probe_ctx.phash.hex(),
            embed_model="sface_2021dec",
            detector=detect.backend(),
            embed_sha256=hashlib.sha256(probe_ctx.embedding.tobytes()).hexdigest(),
            consent_digest=consent_digest,
        )
        match_extra = None
        if mode == "web_reverse":
            # The page URL is already `match.uri`, in the clear, exactly as it
            # is for every other arm. The image URL is additionally kept in the
            # clear here (redacted of any signature params) because a
            # reverse-image finding is only independently checkable if a
            # verifier can open the same image the provider pointed at — the
            # sha256 alone proves integrity but names nothing.
            match_extra = {
                "reverse_image": {
                    "provider": reverse_provider.provider_id,
                    "match_type": best.candidate.extra.get("match_type", "unknown"),
                    "page_url": redact_url(best.candidate.extra.get("page_url") or best.candidate.post_uri),
                    "image_url": redact_url(best.candidate.image_url),
                }
            }

        bundle = build_bundle(
            pipeline_commit="dev",
            probe=probe_info,
            match=best,
            match_text_sha256=hashlib.sha256((best.candidate.text or "").encode()).hexdigest(),
            match_image_url_sha256=hashlib.sha256(best.candidate.image_url.encode()).hexdigest(),
            discovery=discovery_block,
            match_extra=match_extra,
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

        if zk:
            # The whole claim of the ZK rung is that the *chain* decides whether
            # the score is real. Anchoring a proven bundle through plain
            # `anchor()` would write the same digest without anyone ever
            # checking the proof, which is the assertion §7.1 exists to remove.
            try:
                receipt = await adapter.anchor_with_proof(
                    digest, cid, bundle["zk"]["proof"], bundle["zk"]["public_signals"]
                )
            except Exception as e:
                typer.echo(
                    f"ERROR: the chain rejected the proof ({type(e).__name__}: {e}). "
                    "If this EvidenceAnchor was deployed with a zero-address verifier, "
                    "anchorWithProof always reverts — redeploy with contracts/script/Deploy.s.sol, "
                    "which wires one.",
                    err=True,
                )
                raise typer.Exit(1)
            typer.echo(f"Stage 3: tx submitted {receipt.tx_hash} (anchorWithProof — the chain verified the proof)")
        else:
            receipt = await adapter.anchor(digest, cid)
            typer.echo(f"Stage 3: tx submitted {receipt.tx_hash}")
        await adapter.wait_for_inclusion(receipt)
        typer.echo("Stage 3: included on-chain")
        _echo_explorer(chain, contract_address, receipt.tx_hash)

        EVIDENCE_DIR.mkdir(exist_ok=True)
        out_path = EVIDENCE_DIR / f"{bundle['run']['run_id']}.json"
        out_path.write_text(json.dumps(bundle, indent=2))
        (EVIDENCE_DIR / "latest.json").write_text(json.dumps(bundle, indent=2))
        typer.echo(f"Evidence bundle written: {out_path}")
        typer.echo(f"digest={digest.hex()} cid={cid}")

    async def _run_and_close():
        try:
            await _discover_and_anchor()
        finally:
            await _close_adapter(adapter)

    asyncio.run(_run_and_close())


@app.command(name="corpus-build")
def corpus_build(
    db: Path = typer.Option(..., "--db", help="Directory to write the corpus database into"),
    corpus: list[str] = typer.Option([], "--corpus", help="Corpus to ingest, repeatable: mastodon:tag/<tag>[@instance] | mastodon:account/<handle> | bsky:feed/<alias|at-uri> | bsky:author/<handle>"),
    from_dir: Path | None = typer.Option(None, "--from-dir", help="Import local image files instead of fetching — for a corpus you already hold and are entitled to use"),
    pages: int = typer.Option(3, "--pages", help="Pages to page back per corpus (~40-50 images/page)"),
    max_images: int = typer.Option(600, "--max-images", help="Hard cap on images fetched per corpus run"),
    append: bool = typer.Option(False, "--append", help="Merge into an existing database instead of starting fresh"),
    allow_sensitive: bool = typer.Option(False, "--allow-sensitive", help="Do NOT filter posts flagged sensitive/explicit. Off by default; see `search/safety.py`"),
):
    """Compile a local, content-addressed image corpus for repeatable testing.

    Fetch a bounded corpus once and keep the bytes, so every later run scores
    the same images. Live timelines move under you — a candidate that scored
    0.46 yesterday is gone today — which makes them useless for answering
    "did my change alter the outcome".

    Stores image bytes, pHash, source provenance and face *geometry*.
    Deliberately stores NO face embeddings (PRD §3: no persistent embedding
    store); those are recomputed in memory at query time.

    The safety filter runs here, at build time, so explicit posts are excluded
    once rather than re-downloaded on every run.
    """
    if not corpus and from_dir is None:
        typer.echo("ERROR: pass at least one --corpus to fetch, or --from-dir to import local images", err=True)
        raise typer.Exit(1)
    if corpus and from_dir is not None:
        typer.echo("ERROR: --corpus and --from-dir are alternative sources; pass one", err=True)
        raise typer.Exit(1)

    from faceanchor.search.corpus_db import BuildStats, CorpusDb

    store = CorpusDb(db)
    if append and store.exists():
        store.load()
        typer.echo(f"Loaded {len(store.entries)} existing images from {db}")
    stats = BuildStats()
    before = len(store.entries)
    safety_json: dict | None = None
    corpora_named: list[str] = []

    if from_dir is not None:
        corpora_named = [f"local:{from_dir}"]
        typer.echo(f"Importing images from {from_dir}")
        from faceanchor.search.models import Candidate

        files = sorted(
            f for f in Path(from_dir).rglob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        for path in files:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() in store.entries:
                stats.already_present += 1
                continue
            # Absolute, so the provenance URI is meaningful from anywhere —
            # a relative path is not expressible as a file: URI at all.
            uri = path.resolve().as_uri()
            store.add(
                Candidate(platform="local", image_url=uri, post_uri=uri, extra={"variant": "original"}),
                raw,
                corpus=f"local:{from_dir}",
                stats=stats,
            )
        typer.echo(f"Scanned {len(files)} files")
    else:
        from faceanchor.search import corpus as corpus_mod

        try:
            specs = [corpus_mod.parse_spec(raw) for raw in corpus]
        except corpus_mod.CorpusSpecError as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(1)
        corpora_named = [str(s) for s in specs]

        async def _fetch():
            async with fanout.make_client() as client:
                candidates, reports, safety = await corpus_mod.ingest(
                    client, specs, pages=pages, max_images=max_images, allow_sensitive=allow_sensitive
                )
                typer.echo(f"Ingested {len(candidates)} candidate URLs — {', '.join(r.describe() for r in reports)}")
                if safety.total:
                    typer.echo(f"Explicit-content filter removed {safety.total} — {safety.describe()}")
                elif allow_sensitive:
                    typer.echo("WARNING: --allow-sensitive is set; explicit posts were NOT filtered out", err=True)
                fetched = await fanout.fetch_all_candidates(client, candidates)
                typer.echo(f"Fetched {len(fetched)} images")
                return fetched, safety

        fetched, safety = asyncio.run(_fetch())
        safety_json = {
            "total": safety.total, "sensitive": safety.sensitive,
            "labelled": safety.labelled, "by_label": safety.by_label,
            "allow_sensitive": allow_sensitive,
        }
        for candidate, raw in fetched:
            store.add(candidate, raw, corpus=candidate.extra.get("corpus", ""), stats=stats)

    if append and store.header.get("corpora"):
        corpora_named = sorted(set(corpora_named) | set(store.header["corpora"]))
    store.save(corpora=corpora_named, safety=safety_json, source="local_dir" if from_dir else "live")

    summary = store.stats()
    typer.echo(f"\nCorpus database written: {db}")
    typer.echo(f"  {stats.describe()}")
    typer.echo(f"  {summary['images']} images ({before} -> {summary['images']}), {summary['faces']} faces, "
               f"{summary['multi_face_images']} multi-face, {summary['bytes'] / 1e6:.1f} MB")
    typer.echo(f"  by platform: {summary['by_platform'] or 'none'}")
    typer.echo(f"  manifest sha256: {store.manifest_sha256()}")
    typer.echo(
        "\nThis is a directory of strangers' faces on your disk. It holds no embeddings "
        "(PRD §3), is never anchored, and is yours to delete."
    )


@app.command(name="corpus-footprint")
def corpus_footprint(
    probe: Path = typer.Option(..., "--probe", exists=True, help="Path to the probe image"),
    db: Path = typer.Option(..., "--db", help="Corpus database directory built by `corpus-build`"),
    subject_consent: Path | None = typer.Option(None, "--subject-consent", help="Required consent artifact (PRD §3)"),
    face_index: int | None = typer.Option(None, "--face-index", help="Disambiguate when multiple faces are detected"),
    top: int = typer.Option(10, "--top", help="How many ranked appearances to print"),
    report: Path | None = typer.Option(None, "--report", help="Write the full footprint as JSON"),
    contact_sheet: Path | None = typer.Option(None, "--contact-sheet", help="Render probe + every scored image, inlined and score-ordered. Writes third-party faces to disk"),
    limit: int | None = typer.Option(None, "--limit", help="Score only the first N images (development)"),
):
    """Where does this face appear across a fixed corpus — every hit, ranked.

    `run --by-face` asks "is there a match above threshold" and stops at the
    best one, which is the right shape for anchoring and the wrong shape for
    understanding a corpus. This scores every image and reports the whole
    distribution, so the question "is 0.42 a real match" can be answered by
    looking at what the runner-up scored.

    Anchors nothing, persists no embedding, and needs no network.
    """
    try:
        load_consent_digest(subject_consent)
    except MissingConsentError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    from faceanchor.search import footprint as footprint_mod
    from faceanchor.search.corpus_db import CorpusDb, CorpusDbError

    try:
        store = CorpusDb(db).load()
    except CorpusDbError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    if not store.entries:
        typer.echo(f"ERROR: corpus database at {db} is empty", err=True)
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
    typer.echo(f"Probe: {probe.name} sha256={hashlib.sha256(probe_bytes).hexdigest()[:16]}… "
               f"face score={face.score:.3f}, quality gate passed — {_describe_gate(quality_result)}")
    typer.echo(f"Corpus: {len(store.entries)} images, manifest {store.manifest_sha256()[:16]}…\n")

    result = footprint_mod.scan(probe_ctx, store, limit=limit)
    dist = result.distribution()

    typer.echo(f"Footprint — {len(result.matches)} appearance(s) at or above the "
               f"{score.COSINE_ACCEPT_THRESHOLD} accept threshold, out of {result.corpus_images} images")
    if dist.get("scored"):
        typer.echo(f"  scores: max {dist['max']} · runner-up {dist['runner_up']} · margin {dist['margin']} "
                   f"· mean {dist['mean']} · p95 {dist['p95']}")
        if dist["margin"] is not None and dist["margin"] < 0.05 and dist["max"] >= score.COSINE_ACCEPT_THRESHOLD:
            typer.echo("  NOTE: the best hit barely stands clear of the runner-up — look at the images "
                       "before believing it (--contact-sheet)")
    if result.skipped:
        typer.echo(f"  skipped: {', '.join(f'{k} {v}' for k, v in result.skipped.items())}")

    typer.echo(f"\n{'score':>7}  {'metric':<7} {'faces':>5}  {'platform':<9} post")
    for appearance in result.top(top):
        mark = "*" if appearance.above_threshold else " "
        typer.echo(f"{appearance.score:7.4f}{mark} {appearance.metric:<7} {appearance.faces_in_image:>5}  "
                   f"{appearance.platform:<9} {appearance.post_uri[:60]}")
    if len(result.appearances) > top:
        typer.echo(f"  … {len(result.appearances) - top} more (use --top N or --report)")
    typer.echo("\n  * = at or above threshold. A score above a threshold is not a verified match "
               "until a human has looked at it.")

    if report is not None:
        footprint_mod.write_report(result, report)
        typer.echo(f"\nFootprint report written: {report}")

    if contact_sheet is not None:
        from faceanchor.review import write_contact_sheet

        by_sha = {a.sha256: a for a in result.appearances}
        fetched = [
            (c, b) for c, b in store.as_fetched() if c.extra.get("local_sha256") in by_sha
        ]
        trace = [
            {
                "image_url": c.extra.get("local_sha256") and by_sha[c.extra["local_sha256"]].image_url or c.image_url,
                "verdict": "match_cosine" if by_sha[c.extra["local_sha256"]].above_threshold else "below_threshold",
                "cosine": by_sha[c.extra["local_sha256"]].score,
            }
            for c, _ in fetched
        ]
        shown = write_contact_sheet(
            path=contact_sheet, probe_bytes=probe_bytes, probe_label=probe.name,
            fetched=fetched, score_trace=trace, threshold=score.COSINE_ACCEPT_THRESHOLD,
            discovery={"mode": "corpus_footprint", "corpus_db": str(db)},
        )
        typer.echo(f"Contact sheet written: {contact_sheet} ({shown} images inlined) — open it in a browser")


@app.command()
def verify(
    bundle: Path = typer.Option(..., "--bundle", exists=True),
    chain: str = typer.Option("anvil", "--chain", help="anvil | amoy | solana"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
    check_platform: bool = typer.Option(False, "--check-platform", help="Re-fetch the matched post and detect platform-side mutation (PRD §7.2)"),
):
    """Re-derive the digest from the bundle and check it against on-chain state."""
    data = json.loads(bundle.read_text())
    digest = bundle_digest(data)
    consent_digest_hex = data.get("probe", {}).get("consent_digest")

    adapter = _resolve_chain(chain, contract_address)

    async def _verify():
        result = await adapter.verify(digest)

        typer.echo(f"digest={digest.hex()}")
        if result.ok:
            typer.echo(f"OK — anchored at timestamp {result.timestamp}")
        else:
            typer.echo("FAIL — digest not found on-chain (bundle was altered, or never anchored)")

        _echo_explorer(chain, contract_address)

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

    async def _verify_and_close():
        try:
            return await _verify()
        finally:
            await _close_adapter(adapter)

    ok = asyncio.run(_verify_and_close())
    if not ok:
        raise typer.Exit(1)


@app.command()
def footprint(
    chain: str = typer.Option("anvil", "--chain", help="anvil | amoy (solana has no event log to enumerate)"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
    from_block: int = typer.Option(0, "--from-block", help="Public RPCs cap eth_getLogs ranges; narrow this if the node complains"),
    bundle: Path | None = typer.Option(None, "--bundle", help="Also mark which row is this bundle's digest"),
):
    """Everything this contract has recorded on-chain, and where to check it on the web.

    Nothing here reads the evidence files — it is the chain's own account of
    what was anchored, which is the only part a stranger has to believe.
    """
    adapter = _resolve_chain(chain, contract_address)
    mine = bundle_digest(json.loads(bundle.read_text())).hex() if bundle else None

    async def _list():
        try:
            if not hasattr(adapter, "anchored_events"):
                typer.echo(
                    f"ERROR: {chain} has no event log to enumerate — SPL Memo emits no events and "
                    "keeps no state, so there is no chain-side list of what was anchored.",
                    err=True,
                )
                raise typer.Exit(1)
            try:
                return await adapter.anchored_events(from_block=from_block)
            except typer.Exit:
                raise
            except Exception as e:
                typer.echo(f"ERROR: could not read logs ({type(e).__name__}: {e}). Try a narrower --from-block.", err=True)
                raise typer.Exit(1)
        finally:
            await _close_adapter(adapter)

    events = asyncio.run(_list())
    site = explorer.for_chain(chain)

    typer.echo(f"contract {contract_address} on {chain} — {len(events)} record(s) from block {from_block}")
    for e in events:
        marker = " <- this bundle" if mine and e["digest"] == mine else ""
        typer.echo(f"  [{e['block']:>9}] {e['event']:<8} {e['digest']} ts={e['timestamp']}{marker}")
        if e["cid"]:
            typer.echo(f"              cid={e['cid']}")
        if site:
            typer.echo(f"              {site.tx(e['tx_hash'])}")
    if mine and not any(e["digest"] == mine for e in events):
        typer.echo(f"\nthis bundle's digest {mine} is NOT among them")

    typer.echo("")
    _echo_explorer(chain, contract_address)


@app.command()
def revoke(
    consent: Path = typer.Option(..., "--consent", exists=True),
    chain: str = typer.Option("anvil", "--chain", help="anvil | amoy (revocation is EVM-only)"),
    contract_address: str | None = typer.Option(None, "--contract-address"),
):
    """Revoke a previously-anchored consent artifact (PRD §3/§7.3). Irreversible."""
    consent_digest_hex = load_consent_digest(consent)
    adapter = _resolve_chain(chain, contract_address)

    async def _revoke():
        try:
            receipt = await adapter.revoke(bytes.fromhex(consent_digest_hex))
        except NotImplementedError:
            # PRD §7.3 needs somewhere on-chain to record the veto. SPL Memo
            # has no state to flip, so `verify` would have nothing to read back
            # — a revocation that cannot be read is not a revocation.
            typer.echo(
                f"ERROR: {chain} cannot record a revocation — SPL Memo keeps no on-chain state, so "
                "nothing would flip and `verify` would never report it. Revocation is EVM-only "
                "(--chain anvil|amoy).",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"revoke tx submitted: {receipt.tx_hash}")
        await adapter.wait_for_inclusion(receipt)
        typer.echo(f"consent_digest={consent_digest_hex} REVOKED")

    async def _revoke_and_close():
        try:
            await _revoke()
        finally:
            await _close_adapter(adapter)

    asyncio.run(_revoke_and_close())


@app.command(name="forge-score")
def forge_score(
    bundle: Path = typer.Option(..., "--bundle", exists=True),
    score: float = typer.Option(..., "--score", help="A claimed score to forge into the bundle's public signals"),
    chain: str = typer.Option("anvil", "--chain", help="anvil | amoy (the ZK path is EVM-only)"),
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

    async def _forge_and_close():
        try:
            await _forge()
        finally:
            await _close_adapter(adapter)

    asyncio.run(_forge_and_close())


if __name__ == "__main__":
    app()
