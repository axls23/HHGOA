# Architecture

Three stages, see PRD §4 for the full diagram and §9 for the latency budget.
The numbers below are stated here rather than delegated to the PRD: they are
the decisions, and a reader should not have to open another document to learn
what "the cascade" actually does.

- **Stage 1 (`src/faceanchor/vision/`)**: `decode.py` → detect
  (`detect.py`: bbox + 5-pt landmarks in one pass, from YuNet or, with
  `FACEANCHOR_DETECTOR=yolo`, YOLOv8-face via `yolo_face.py` — both emit the
  same 15-value row, so everything downstream is indifferent to which ran)
  → **quality gate**
  (`quality.py`, using `pose.py` and `fiqa.py`) → `align.py`
  (`FaceRecognizerSF.alignCrop`) → SFace embed (`embed.py`), with `phash.py`
  computed over the whole probe image for the cascade's first rung.

  The quality gate is architectural, not cosmetic: it is what makes the whole
  Stage-2 budget conditional. It runs *before anything leaves the machine* —
  including before the `--discovery web-reverse` upload acknowledgement
  (`cli.py`), so a probe too poor to search is never sent to a third party.
  Cheapest question first: a single dominant face (else `--face-index`), bbox
  ≥ 80 px, Laplacian variance ≥ 60, head pose within ±35° yaw / ±35° pitch /
  ±45° roll, then learned quality ≥ 0.35.

  The last two are models, and they replaced heuristics that were measuring
  the wrong thing:

  - `pose.py` solves PnP against a canonical 3D face using the same five
    landmarks, so yaw, pitch and roll come out in degrees. The ratio it
    replaced compared the nose's distance to each eye — a yaw proxy blind to
    pitch (chin to the ceiling stays symmetric) and to roll (a tilted head is
    symmetric about its own axis), both of which wreck a 5-point alignment.
    The model points are calibrated so the population medians over the
    reference corpus land at ~0°, since an absolute threshold in degrees is
    meaningless otherwise. Net effect: it rejects 14% of that corpus where the
    ratio rejected 31%, while catching the failures the ratio could not see.
  - `fiqa.py` (eDifFIQA-T, MIT) scores how usable a face is for recognition,
    trained against that outcome. Laplacian variance answers "does this image
    contain high-frequency detail", which rises with a busy background and
    with JPEG blocking: a 6× nearest-neighbour pixelation of a face reads
    *twelve times above* the blur threshold while being unrecognisable. The
    Laplacian check stays, demoted to a coarse screen for an obviously broken
    frame.

  The detector is pluggable for the same reason the chain adapters are, and
  with the same discipline: one interface, and the bundle records which
  implementation ran (`probe.detector`). It matters more here than it looks,
  because the two backends align slightly differently — the same face embeds
  to a cosine of 0.93-0.98 across them — so a run must not mix them between
  the probe and the candidates. It cannot: `score.py` re-detects candidates
  through the same `detect_faces()`. The YOLO weights are also the one
  non-permissive thing in the model cache (GPL/AGPL against everything else's
  Apache-2.0/MIT), and unlike the GPL Groth16Verifier a detector cannot be
  scoped to an opt-in path — hence YuNet stays the default.

  Both quality models are optional by construction. Absent weights degrade to the
  heuristics and every caller says so out loud (`cli.py::_describe_gate`),
  rather than a gate quietly becoming weaker than it claims to be.

  Two things live at the *capture* boundary rather than in the gate, because
  they are only answerable of a live camera (`scripts/capture_probe.py`):

  - **Liveness** (`liveness.py`, MiniFASNet ensemble, Apache-2.0). A printed
    photograph held to the lens passes every quality check there is, and
    anchors a bundle whose every downstream guarantee holds while attesting to
    something that never happened. Tamper-evidence downstream of a spoofed
    capture is the wrong shape of assurance, so the check runs on the frame
    about to be written and refuses it. It is a single-frame RGB classifier —
    beaten by a good mask, an unseen attack, or a frame injected below the
    camera API — and it does not run on file probes, where it would only be
    guessing.
  - **Best-of-N frame selection.** Gate-passing frames go into a short rolling
    buffer and the shutter keeps the best, ranked on learned quality and
    exposure. Pose is deliberately excluded from that ranking: blurring moves
    the landmarks it is solved from, and moves them toward the mean, so a
    degraded frame reads as more frontal. Ranking on pose picks the blurriest
    frame in the burst — measured, then removed.

  Single-probe embed runs CPU (raw ONNX Runtime, thread-tuned
  `SessionOptions`, not `cv2.FaceRecognizerSF.feature()` — see `embed.py`'s
  docstring for the measured ~38ms vs ~17ms p50 difference). Candidate
  batch embed runs CUDA EP against a batch-dynamic export of the same
  weights (`models.py::ensure_sface_dynbatch_onnx` — the upstream OpenCV
  Zoo export hardcodes batch=1), falling back to `CPUExecutionProvider`
  when CUDA EP fails to initialize, so a GPU-less machine degrades in speed
  rather than in behaviour.

- **Stage 2 (`src/faceanchor/search/`)**: three discovery arms, exactly one
  per run, all three converging on the same cascade.
  - *text-seeded* (`--query`): `fanout.py` runs Arm A (Bluesky + Mastodon +
    Commons, concurrent, per-source timeout) then Arm B (`duckduckgo.py`,
    seeded from Arm A's best hit).
  - *face-embedding over a named corpus* (`--by-face`): `corpus.py` ingests
    the hashtag timelines / feeds / accounts you listed. Not reverse-image
    search — nothing is submitted anywhere, the probe's embedding is the query.
  - *genuine reverse-image* (`--discovery web-reverse`):
    `web_reverse/google_vision.py` submits the probe image to Google Cloud
    Vision Web Detection (`images:annotate`, `WEB_DETECTION`);
    `web_reverse/social.py` filters the open-web results down to Bluesky and
    Mastodon posts and turns them into ordinary `Candidate`s. This is the one
    arm that crosses a third-party trust boundary, so `cli.py` gates it behind
    an explicit acknowledgement separate from the consent artifact.
    `web_reverse/base.py` holds the provider ABC, the normalized result type
    and the error hierarchy that keeps PROVIDER_ERROR from ever being
    reported as NO_RESULTS.

  Any of the three can be replaced by a **replay**: `corpus_db.py` compiles a
  bounded corpus once into a content-addressed local store (`manifest.jsonl`
  + `images/<xx>/<sha256>`), and `run --by-face --corpus-db <dir>` scores
  those bytes with no network at all. The store keeps provenance and face
  geometry and deliberately keeps *no embeddings* (PRD §3), so it is a test
  bench rather than a facial-recognition database. `footprint.py` scans a
  probe against the whole store and reports the score distribution — the
  question `run` cannot answer, because it stops at the best match.

  **`safety.py` sits between every arm and the cascade**, and it runs at
  *extraction* rather than after download: a post the poster flagged (Mastodon
  `sensitive`) or that carries an explicit AT Protocol label is dropped
  before it costs a request, bytes, or a decode, so nothing explicit is
  fetched, logged in `--log-candidates`, or inlined into a `--contact-sheet`.
  Corpus specs that are adult corpora by definition (`ADULT_TAGS`,
  `ADULT_INSTANCES`) are refused outright rather than filtered, because
  ingesting one means downloading hundreds of strangers' explicit images to
  search for a face. It is a metadata filter, not a classifier — it trusts
  the poster and their instance — and it is on by default with the count
  surfaced per source (`filtered_sensitive`), never silently.

  Whichever arm ran, `score.py` then runs the cascade from PRD §5.2,
  cheapest filter first: pHash Hamming ≤ 8 accepts immediately (it removes
  ~70% of embedding calls on real corpora); survivors are re-detected and
  dropped if they carry no face; the rest are batch-embedded (batch 32, CUDA
  EP); cosine ≥ 0.363 accepts, and ≥ 0.50 early-exits the whole fanout. The
  reverse-image provider's opinion is never mixed into that score: it
  proposes candidates, SFace decides matches.

  The network budget lives in `fanout.py`: HTTP/2 with a shared keep-alive
  pool capped at 64 connections, 64-way candidate fetch concurrency, a 512 KB
  range cap per candidate image, and a 3.0 s per-arm timeout (Arm B carries
  its own 1.5 s inner deadline) so one slow source cannot stall the stage.

  **Degrading cleanly is right; degrading silently is not.** Both halves of
  Stage 2 encode the same principle: `fanout.SourceReport` distinguishes
  "blocked" (Bluesky's 403) from "answered honestly with nothing" (Mastodon's
  auth-gated search: HTTP 200, empty list) from "no hits", and reports each
  per source; `web_reverse/base.py`'s error hierarchy keeps a broken provider
  from ever reading as "the face is nowhere". Distinct CLI exit codes carry
  the same distinction to a caller who is not reading prose.

- **Stage 3 (`src/faceanchor/evidence/` + `src/faceanchor/chain/`)**:
  `bundle.py` builds the PRD §6.3 schema, `jcs.py` canonicalizes + hashes
  it (RFC 8785, keccak256), `cid.py` computes a local CIDv1. `chain/base.py`
  is the adapter interface; `anvil.py`/`evm.py` implement it for EVM chains,
  `solana.py` for SPL Memo, and `explorer.py` turns a receipt into a link on
  a public block explorer — the route to verification that runs none of this
  code. It has no Anvil entry on purpose: a local devnet has no public
  footprint, and pretending otherwise is the false assurance the module
  exists to prevent.

  What the bundle **excludes is the design**, not an omission: no embedding
  vectors, no image bytes, no display names, no handles-as-identity — only
  hashes of those things. A hash is not reversible; a 128-d template on an
  immutable ledger is a permanent biometric record nobody can delete. The
  bundle does record *how* the match was found (`run.discovery`), because a
  by-face hit over a named corpus, a text-seeded hit, and a corpus replay are
  three different claims and a verifier should not have to guess which one
  they are checking; a replay additionally commits to the corpus manifest's
  own digest.

  `evidence/quantize.py` is the float→field boundary for the ZK rung: SFace
  embeddings are L2-normalized, so `round(x · 2¹²)` lands in [-4096, 4096]
  (measured cosine drift < 0.001, well inside the 0.363 decision margin), and
  `shift_unsigned` moves that into the unsigned range circom signals require.

  Chain capability is deliberately uneven, and the CLI says so rather than
  pretending otherwise: `revoke`/`consentStatus` (§7.3) and `anchorWithProof`
  (§7.1 L2) are EVM-only — SPL Memo keeps no on-chain state and runs no
  program — so `chain/base.py` raises `NotImplementedError` for them and each
  command turns that into an explanation. Solana's `verify()` is likewise a
  wallet-scoped scan of recent signatures, not the O(1) `anchoredAt` lookup
  the EVM path keeps precisely so `verify` needs no indexer (PRD §6.2).

`cli.py` is the only place these three stages meet in a single function —
intentionally, so the actual pipeline order is readable in one file rather
than spread across a service layer that doesn't exist yet (no FastAPI/SSE
layer is built — PRD §5.3 specs one, but the CLI is the judge-facing surface
per the task's "no website" constraint, so it wasn't built out this pass).
`search/score.py` and `search/corpus.py` also call into `fanout` for fetching;
what is not duplicated anywhere is the *wiring* of vision → discovery →
evidence → chain.

`src/faceanchor/zk/` (witness.py, prove.py) shells out to `node` +
`snarkjs` rather than reimplementing Groth16 — see PRD §7.1's own
reasoning for picking Circom/snarkjs over Noir/Halo2 specifically because
`zkey export solidityverifier` is one command. `run --zk` anchors through
`EvidenceAnchor.anchorWithProof`, so the chain runs the verifier before it
will record the digest; anchoring a proven bundle through plain `anchor()`
would write the same digest with nobody having checked the proof, which is
the assertion §7.1 exists to remove.

## Cross-cutting

Things that are not any one stage's, and that the stage-by-stage reading
above will not show you:

- **The consent gate runs first.** `consent.py::load_consent_digest` is
  checked in `run` and `corpus-footprint` before the probe is even decoded,
  so a run without `--subject-consent` does not happen (PRD §3). Only the
  artifact's JCS-canonicalized SHA-256 digest is used — the artifact never
  leaves the machine, and it is the digest that is anchored and that
  `revoke` later flips.
- **Everything is asyncio, one client per run.** Each CLI command wraps its
  work in a single coroutine under `asyncio.run`; `fanout.make_client()`
  yields one shared `httpx.AsyncClient` for the whole of Stage 2. Adapters
  that own a transport (`solana.py`) are closed by the command that made
  them; the ones that don't (`evm.py`) are not asked to.
- **`forge build` is a runtime dependency of the Python package.**
  `chain/evm.py` loads the EvidenceAnchor ABI from
  `contracts/out/EvidenceAnchor.sol/EvidenceAnchor.json` at first use — the
  Solidity build output is not vendored, so `make setup` builds it.
- **The EVM nonce is cached in-process** after the first fetch, to save a
  round trip per tx (§9). That assumes one writer per process: a future
  service layer anchoring concurrently from the same key would need to drop
  the cache or serialize on it.
- **Model weights are provisioned, not committed.** `models/fetch.sh` fetches
  YuNet/SFace (required) plus the MiniFASNet pair and eDifFIQA-T (optional, and
  a failed download warns rather than aborting);
  `models.py::ensure_sface_dynbatch_onnx` derives the batch-dynamic export on
  first use; `vision/_cuda_libs.py` dlopens cuDNN/cuBLAS out of site-packages
  so no `LD_LIBRARY_PATH` export is needed. Every weight in the cache is
  Apache-2.0 or MIT — which is why the capture stage uses eDifFIQA and
  MiniFASNet rather than the better-known CR-FIQA (CC BY-NC) or the InsightFace
  zoo (non-commercial), the same rule that keeps buffalo_l out of the default
  path.
- **`review.py` is a local data-egress surface, and is treated as one.**
  `--contact-sheet` renders the probe and every fetched candidate as inlined
  thumbnails so a 0.46 can be judged by eye — which means writing other
  people's faces to disk. So it is opt-in, lands under gitignored
  `evidence/`, is never anchored and never part of the bundle (§6.3 excludes
  image bytes by construction), and is yours to delete. `--log-candidates`
  is the same posture without the images, and redacts signed CDN URLs
  because a signed URL is a bearer token, not a name.
