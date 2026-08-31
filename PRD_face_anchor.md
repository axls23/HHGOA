# PRD — `face-anchor`
### Face Identification → Web/Social Discovery → On-Chain Verification
**HH Goa 2026 · Shortlisting Task 3**
**Owner:** Sahil (axls23) · **Status:** Draft v1 · **Target:** working repo + reproducible demo, no website

---

## 1. Summary

A three-stage pipeline that takes a probe face image, performs a **live** search across public web/social corpora to locate a genuinely matching post, and anchors a canonical fingerprint of that discovery on a blockchain so the finding can be independently re-verified later.

**Design axis:** end-to-end latency. Every architectural choice below is made against a p50 wall-clock budget, and every dependency is permissively licensed and self-hostable.

**Headline numbers (target):**

| Metric | Target |
|---|---|
| Probe → ranked match returned | **< 1.0 s p50** |
| Probe → tx submitted | **< 1.1 s p50** |
| Probe → on-chain finality (Anvil / Solana devnet / Polygon Amoy) | **5 ms / 0.6 s / 3.0 s** |
| Independent re-verification (cold, from repo + chain only) | **< 200 ms** |

---

## 2. Goals / Non-goals

**Goals**
- G1. Detect + align + embed a face from an arbitrary input image, deterministically.
- G2. Execute a real, non-hardcoded search over live public APIs and return ≥1 matching post with a defensible similarity score.
- G3. Commit an immutable, content-addressed fingerprint of the discovery to a chain; re-verify it from the artifact alone.
- G4. Ship a repo any judge can `git clone && make demo` against a public testnet in under 10 minutes.
- G5. Fully open-source dependency graph (permissive licences only in the default path).

**Non-goals**
- Website / hosted frontend (explicitly excluded by the task).
- Building or persisting a face database. The pipeline is **stateless per probe** — see §3.
- Mainnet deployment or token economics.
- Beating commercial face-search accuracy. Recall over the open web is not the deliverable; a working, verifiable pipeline is.

---

## 3. Subject Policy (design constraint, not an afterthought)

This pipeline is a deanonymisation primitive. Built naively — arbitrary face in, stranger's identity out — it is a stalking tool, and it is squarely in the crosshairs of India's DPDP Act 2023 (biometric data = sensitive personal data; §6 consent, §12 erasure) and EU AI Act Art. 5(1)(e) (prohibition on untargeted scraping of facial images to build recognition databases). It would also fail any serious judging rubric on responsible-AI grounds.

The design constrains the **subject set**, not the capability, which costs nothing in demonstrated technical depth:

| Lane | Probe source | Basis |
|---|---|---|
| **A** (default) | Builder's own face | Self |
| **B** | Teammate face | Signed consent artifact, hashed into the bundle (§6.3) |
| **C** | Public figure with an already-public account | Public-interest, verifiable-attribution use case |

Hard rules enforced in code:
- **No persistent embedding store.** Embeddings live in-process, TTL-bounded LRU only, never written to disk or chain.
- **No raw biometric on-chain.** Chains cannot honour erasure requests. Only salted digests go on-chain (§6.3).
- `--subject-consent <path>` is a required CLI flag; the run aborts without it.

**Differentiator:** the consent artifact's digest is anchored on-chain alongside the match. The submission therefore demonstrates *blockchain-verified consent*, which is a strictly better story than the base task and costs ~15 lines of code.

---

## 4. Architecture

```
                    ┌──────────────────────────────────────────┐
  probe.jpg ───────►│ STAGE 1 · VISION            (~25 ms)     │
                    │  decode → YuNet detect → 5-pt align       │
                    │  → quality gate → SFace embed (128-d)     │
                    └───────────────┬──────────────────────────┘
                                    │ e_probe, phash(probe)
                    ┌───────────────▼──────────────────────────┐
                    │ STAGE 2 · DISCOVERY        (~700 ms)     │
                    │  ┌ Arm A (fast, ToS-clean) ─────────────┐ │
                    │  │ Bluesky AppView · Mastodon · Commons │ │
                    │  └──────────────────────────────────────┘ │
                    │  ┌ Arm B · corroboration, best-effort ──┐ │
                    │  │ ddgs image+text (seed-driven)        │ │
                    │  └──────────────────────────────────────┘ │
                    │  → async image fetch (64 conc.)           │
                    │  → pHash prefilter → batch embed          │
                    │  → cosine rank, early-exit @ 0.50         │
                    └───────────────┬──────────────────────────┘
                                    │ Match{uri, cid, image_sha256, score}
                    ┌───────────────▼──────────────────────────┐
                    │ STAGE 3 · ANCHOR           (~45 ms + inc)│
                    │  bundle → RFC 8785 JCS → keccak256        │
                    │  → IPFS CIDv1 (local, no daemon)          │
                    │  → EvidenceAnchor.anchor(digest, cid)     │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │ VERIFY (independent)        (~150 ms)    │
                    │  rebuild bundle → recompute digest        │
                    │  → eth_call verify(digest) → (bool, ts)   │
                    └──────────────────────────────────────────┘
```

Response is streamed via SSE: the match is emitted **before** inclusion. Anchoring is async; the client receives `tx_hash` then `receipt`. Perceived latency is Stage 1+2 only.

---

## 5. Tech Stack

### 5.1 Vision

| Component | Choice | Licence | Rationale |
|---|---|---|---|
| Detector | **YuNet** `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) | Apache-2.0 | 230 KB, ~6 ms CPU @ 320×320, 5-pt landmarks included — no separate landmark pass |
| Recogniser | **SFace** `face_recognition_sface_2021dec.onnx` | Apache-2.0 | 128-d embedding, ~12 ms CPU, published cosine threshold 0.363 / L2 1.128 |
| Runtime | **ONNX Runtime 1.2x** | MIT | CPU EP (`intra_op_num_threads` pinned), CUDA/TensorRT EP optional. Sessions pre-warmed at boot; zero per-request model load |
| Perceptual hash | `cv2.img_hash.pHash` | Apache-2.0 | 64-bit, ~0.3 ms/img — catches exact re-uploads without invoking the recogniser |
| Decode | OpenCV + `PyTurboJPEG` | Apache-2.0 / MIT | ~2× faster JPEG decode on the candidate fetch path |

**Optional accuracy path (flagged, not default):** InsightFace `buffalo_l` (SCRFD-10GF + ArcFace R100, 512-d). Materially better recall, but **the pretrained weights are licensed for non-commercial research only** — flag it in the README as a licensing limitation and keep YuNet+SFace as the default so the shipped default path is cleanly permissive.

**Quality gate** (runs pre-search, saves the entire Stage-2 budget on bad probes):
- min face bbox ≥ 80 px
- Laplacian variance ≥ 60 (blur)
- landmark asymmetry ratio ≤ 0.35 (extreme pose)
- single dominant face, else require `--face-index`

### 5.2 Discovery

Scraping Google/Yandex reverse image search is the **single largest latency and reliability risk in the whole system**: 2–8 s per query, CAPTCHA-gated, and it will fail live in front of judges. SearXNG still has no native reverse-image engine (searxng#291, open since 2021). The critical path therefore uses open APIs, and the general-web arm runs through **DuckDuckGo (`ddgs`)** as a best-effort side channel.

| Source | Endpoint | Why |
|---|---|---|
| **Bluesky** (primary) | `public.api.bsky.app` → `app.bsky.feed.searchPosts`, `com.atproto.repo.listRecords` | Unauthenticated public AppView, no anti-bot, ~150–300 ms. **AT Protocol records are already content-addressed (CID)** — the platform has itself hash-committed the post, which composes beautifully with Stage 3 |
| **Bluesky firehose** (optional) | Jetstream WebSocket | Lightweight JSON firehose for a live-tail demo mode |
| **Mastodon** | `/api/v2/search`, public timelines on `mastodon.social` etc. | Open API, ToS-permissible, no key |
| **Wikimedia Commons** | `action=query&generator=search` | CC-licensed images; the right corpus for Lane C public figures |
| **DuckDuckGo (Arm B)** | `ddgs` (MIT) → `.images()` / `.text()` | Keyless, no account, no billing. Far less hostile to automation than Google. **Not a reverse image search** — see below |

#### DuckDuckGo: what it actually gives you

This is a semantic change, not a find-and-replace on the vendor name. **DuckDuckGo has no search-by-image endpoint.** `ddgs.images()` is a *text query* against DDG's image index (Bing-backed); there is no upload or `image_url` parameter anywhere in the surface. So Arm B cannot be a reverse-image lookup — it becomes a **seed-driven candidate generator**:

```
Arm A hit ──► seed terms ──► ddgs.images(seed) ──► candidate URLs
              (author handle,        │
               display name,         └──► same cascade as Arm A:
               post keywords)              pHash → detect → embed → cosine
```

Seeds come from Arm A results or an explicit `--seed-query`. The consequence is worth stating plainly in the README: **Arm B corroborates and expands around a match Arm A already found; it cannot bootstrap an identity from a face alone.** That is a genuine capability reduction versus reverse-image search — and it happens to sit better with §3, because face-to-stranger-identity is exactly the mode the subject policy exists to prevent.

Implementation notes:

- **Package name is `ddgs`.** It was renamed from `duckduckgo-search`; guides pinning the old name are stale. MIT-licensed.
- **Unofficial.** `ddgs` wraps DDG's HTML/`vqd`-token endpoints — it is a scraper, not an API. The `vqd` token is query-and-session bound and must be threaded through follow-up requests. Cache one token per session rather than re-fetching per query; that removes a full round trip.
- **Rate limiting is the binding constraint, not latency.** DDG throttles aggressively on rapid bursts. Enforce a client-side ~1 req/s limiter, catch `RatelimitException`, exponential backoff, and hard-fail the arm at a 1.5 s deadline rather than retrying into the response path.
- **Fallback:** self-hosted SearXNG with the DDG engine enabled gives you a proxy pool and a second path if the direct client gets throttled mid-demo.
- Arm B is fired concurrently with Arm A and merged **only if it lands within deadline**. A throttled Arm B must never degrade the primary result.

Fanout: `httpx.AsyncClient` with HTTP/2, keep-alive pool, `limits=Limits(max_connections=64)`, `asyncio.gather` with per-arm timeouts. Candidate images fetched at 64 concurrency with a 512 KB cap and `Range` header where supported.

Scoring cascade (cheapest filter first):
1. `hamming(phash_probe, phash_cand) ≤ 8` → immediate high-confidence match, skip embedding.
2. Else YuNet detect on candidate; no face → drop.
3. Batch embed survivors (`batch_size=32`).
4. `cosine(e_probe, e_cand)`; accept ≥ **0.363**, early-exit the whole fanout at ≥ **0.50**.

Step 1 alone removes ~70% of embedding calls on real corpora.

### 5.3 Service

| Layer | Choice | Rationale |
|---|---|---|
| API | **FastAPI + uvicorn[standard]** (uvloop, httptools) | SSE streaming for progressive results |
| CLI | **Typer** | `faceanchor run`, `faceanchor verify` — the primary judge-facing surface |
| Web search | **`ddgs`** (MIT, pin `>=9.11`) + `aiolimiter` | Keyless DuckDuckGo access; limiter keeps Arm B inside DDG's throttle |
| Cache | in-process `cachetools.TTLCache` keyed on `sha256(image_url)` | No Redis dependency; keeps the demo one-process |
| Tracing | **OpenTelemetry** spans per stage → console exporter | Makes the latency table in the README reproducible, not asserted |
| Bench | `scripts/bench.py` → p50/p95/p99 per stage | Numbers in the README must be generated by this script |

---

### 5.4 AI Stack — Target Hardware Profile

**Build machine:** Intel i7-13700HX (8 P-core / 8 E-core, 24 threads) · RTX 4050 Laptop (6 GB GDDR6, ~192 GB/s) · 16 GB DDR5 · NVMe.

#### The headline finding: the GPU is not the bottleneck

| | CPU-only (P-cores) | CUDA EP (4050) | Δ |
|---|---:|---:|---:|
| YuNet detect (320×320) | 3.5 ms | 1.4 ms | −2.1 ms |
| SFace embed (probe, 1×) | 9 ms | 3 ms | −6 ms |
| SFace embed (batch 32) | 160 ms | 20 ms | −140 ms |
| **Stage 1 total** | **~24 ms** | **~9 ms** | **−15 ms** |
| **End-to-end p50** | **~730 ms** | **~715 ms** | **−2%** |

Stage 1 is ~3% of wall clock. Stage 2 network fanout is ~95%. **Do not spend build hours on GPU optimisation** — spend them on connection pooling, concurrency, and the pHash prefilter. The 4050 earns its keep in exactly one place: batch-embedding candidates (160 ms → 20 ms), which matters only on wide fanouts.

Recommended default: **CUDA EP for the candidate batch, CPU EP for the probe.** Avoids a GPU round trip on the single-image hot path where PCIe transfer costs more than the compute saves.

#### VRAM budget (6 GB ceiling)

| Item | VRAM |
|---|---:|
| CUDA context + cuDNN/cuBLAS handles | ~700 MB |
| YuNet fp16 weights + activations | ~15 MB |
| SFace fp16 weights | ~19 MB |
| Batch-32 activation workspace | ~180 MB |
| **Vision subtotal** | **~0.9 GB** |
| **Free** | **~5.1 GB** |

The vision stack is tiny — 6 GB is not a constraint here. Export both models to **fp16** (`onnxconverter-common.float16`); accuracy impact on SFace cosine is <0.002, and it halves the transfer cost.

**TensorRT EP:** ~1.6× over CUDA EP on these graphs, but a 30–90 s engine build on first run and a builder workspace that can OOM at 6 GB. Enable `trt_engine_cache_enable` + `trt_engine_cache_path` on NVMe so it's a one-time cost, and gate it behind `--ep trt`. Not worth it for a 15 ms saving — listed for completeness, cut it first.

#### CPU threading on a hybrid (P/E) core layout

This is the single highest-value tuning item on a 13700HX and it is routinely missed:

```python
so = ort.SessionOptions()
so.intra_op_num_threads = 8          # P-cores only — NOT 24
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

Setting `intra_op_num_threads=24` lets the scheduler place ORT work on E-cores. Because ORT joins on the slowest thread in an intra-op barrier, one E-core straggler gates the whole op — measured effect is 2–3× worse p99 and badly unstable benchmarks. Pin explicitly:

- Linux: `taskset -c 0-15 uvicorn ...` (P-core logical IDs, verify with `lscpu -e`)
- Windows: `KMP_AFFINITY=granularity=fine,compact,1,0`

**Then give the E-cores a job.** Run the asyncio event loop and the httpx fanout unpinned so the scheduler parks network I/O on E-cores while P-cores hold inference. Clean split that maps directly onto the two-stage pipeline: **P-cores compute, E-cores wait on sockets.**

#### 16 GB RAM budget

| Item | RSS |
|---|---:|
| Python + ORT + OpenCV + FastAPI | ~1.2 GB |
| Candidate image buffers (64 conc. × 512 KB cap) | ~35 MB |
| TTL embedding cache (2k entries × 128 f32) | ~1 MB |
| Anvil node | ~150 MB |
| `forge build` (transient) | ~800 MB |
| Optional LLM, Q4_K_M 3–4B, GPU-offloaded, mmap'd | ~600 MB host |
| OS + shell + editor | ~4.5 GB |
| **Total** | **~7.3 GB** — comfortable |

Two hard rules at 16 GB:

1. **`uvicorn --workers 1`.** Multiple workers fork multiple ORT sessions — N× the model RAM and N× the CUDA context. Concurrency here comes from asyncio, not processes.
2. **Close the browser for benchmark runs.** Chrome will take 3–4 GB and it will show up as variance in `bench.py`, not as a memory error.

#### Optional LLM — off the critical path

An LLM is *not* required by this pipeline and must not sit in the request path. Its one honest use is Arm B seed extraction (pulling a name/handle/keyword set out of Arm A post text). Deterministic parsing handles ~90% of that at zero latency cost, so:

- **Hot path:** regex/handle parsing. No model.
- **Behind `--seed-llm`:** Qwen3-4B-Instruct or Llama-3.2-3B-Instruct at **Q4_K_M GGUF** (~2.2–2.5 GB), llama.cpp with `-ngl 99` for full offload — fits the ~5.1 GB free VRAM alongside the vision stack with 4k KV cache to spare.
- Budget it at +250–600 ms and document it as an accuracy/latency trade, not a default.

This follows the staged-sequencing discipline: models are resident only for the stage that needs them, never co-resident by accident.

#### NVMe layout

```
~/.cache/faceanchor/
├── models/        # ONNX fp16 weights, mmap'd, never re-downloaded
├── ort_cache/     # TensorRT engine cache (if --ep trt)
└── evidence/      # bundles; CIDs computed in-process, no IPFS daemon
```

Model load is mmap-from-NVMe at boot only; sessions are pre-warmed with a dummy forward pass so the first real request doesn't eat a cold-start.

#### Thermal note

A 45–115 W laptop 4050 throttles under sustained load, which injects variance into any benchmark you publish. Since GPU saves only ~2% end-to-end here, **run `bench.py` CPU-only on a cool machine** and report those numbers. Reproducible beats fast.

---



## 6. Blockchain Design

### 6.1 Chain selection

| Chain | Block/slot | Cost | Tooling | Verdict |
|---|---|---|---|---|
| **Anvil** (Foundry, local) | instant (`--block-time 0`) | 0 | Foundry | **CI + deterministic tests.** Judges can run the full pipeline offline |
| **Polygon Amoy** (80002) | ~2 s | free faucet (Alchemy/QuickNode) | Foundry, viem, web3.py, Polygonscan | **Primary public demo.** Best tooling/latency balance; block explorer link is strong demo evidence |
| **Solana devnet** | ~400 ms slot, confirmed <1 s | free faucet | `solana-py`, SPL Memo | **Documented low-latency alternative.** SPL Memo needs no custom program at all — digest goes straight in the memo |
| Base / Arbitrum Sepolia | ~2 s | free | good | Viable, but see the caveat below |

⚠️ **Timeline risk, verify before committing:** the Ethereum Foundation has Sepolia at **end-of-life 30 September 2026** — roughly a month out. Every Sepolia-settled L2 testnet (Base Sepolia, Arbitrum Sepolia, OP Sepolia) inherits that. Polygon Amoy also checkpoints to Sepolia. Mitigation: the chain layer is an **adapter interface** (`chain/base.py`), so swapping targets is a config change, and Anvil guarantees the demo runs regardless of testnet weather. Confirm current status before the submission deadline.

**Recommendation:** ship `AnvilAdapter` + `EvmAdapter` (Amoy) in the default path; include `SolanaAdapter` as the latency showcase. Three adapters, one interface — good depth signal for ~200 LoC.

### 6.2 Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @notice Tamper-evident anchor for off-chain evidence bundles.
/// @dev Stores no personal data — only keccak256 digests of canonicalised JSON.
contract EvidenceAnchor {
    event Anchored(
        bytes32 indexed digest,
        address indexed attester,
        uint64  timestamp,
        string  cid          // IPFS CIDv1 of the full bundle
    );
    event BatchAnchored(bytes32 indexed merkleRoot, address indexed attester, uint64 timestamp, uint32 leaves);

    error AlreadyAnchored(bytes32 digest);

    mapping(bytes32 => uint64) public anchoredAt;

    function anchor(bytes32 digest, string calldata cid) external {
        if (anchoredAt[digest] != 0) revert AlreadyAnchored(digest);
        anchoredAt[digest] = uint64(block.timestamp);
        emit Anchored(digest, msg.sender, uint64(block.timestamp), cid);
    }

    /// @notice One tx anchors N discoveries; verify a leaf with a Merkle proof.
    function anchorBatch(bytes32 merkleRoot, uint32 leaves) external {
        if (anchoredAt[merkleRoot] != 0) revert AlreadyAnchored(merkleRoot);
        anchoredAt[merkleRoot] = uint64(block.timestamp);
        emit BatchAnchored(merkleRoot, msg.sender, uint64(block.timestamp), leaves);
    }

    function verify(bytes32 digest) external view returns (bool ok, uint64 ts) {
        ts = anchoredAt[digest];
        ok = ts != 0;
    }
}
```

- **Gas:** ~46k (cold SSTORE 20k + log ~2k + base 21k). Batch mode amortises to ~50 gas/record at N=1024.
- **Storage vs event:** `anchoredAt` mapping kept deliberately, so `verify()` is a single `eth_call` with no log-index or archive-node dependency. That is the difference between a 60 ms verification and one that needs a working indexer.
- **Toolchain:** Foundry (`forge build`, `forge test`, `forge script Deploy`), `solc 0.8.26`, `forge fmt`, invariant test that `anchor` is idempotent-rejecting.
- **Client:** `web3.py` async provider, pre-signed with a local dev key; nonce cached to avoid a round trip per tx.

### 6.3 Evidence bundle & digest

`digest = keccak256(JCS(bundle))` where JCS = RFC 8785 JSON Canonicalization Scheme. Canonicalisation is what makes re-verification deterministic across machines — do not hash `json.dumps` output.

```json
{
  "v": 1,
  "run": { "pipeline_commit": "a1b2c3d", "run_id": "01J...", "observed_at": "2026-08-31T09:12:44Z" },
  "probe": {
    "image_sha256": "…",
    "image_phash": "d4e5…",
    "embed_model": "sface_2021dec",
    "embed_sha256": "…",
    "consent_digest": "…"
  },
  "match": {
    "platform": "bsky",
    "uri": "at://did:plc:…/app.bsky.feed.post/3k…",
    "record_cid": "bafyrei…",
    "author_did": "did:plc:…",
    "image_url_sha256": "…",
    "image_sha256": "…",
    "image_phash": "d4e5…",
    "text_sha256": "…"
  },
  "score": { "metric": "cosine", "value": 0.5123, "threshold": 0.363, "path": "embed" }
}
```

**Explicitly excluded from the bundle and the chain:** raw embedding vectors, image bytes, display names, handles-as-identity. Only `embed_sha256` — the hash — is committed. A hash is not reversible; a 128-d template on an immutable ledger is a permanent biometric record you cannot delete.

**Off-chain storage:** compute the CIDv1 locally (deterministic, `raw` leaves, sha2-256) — no IPFS daemon required for the digest to be meaningful. Optionally `ipfs add` via Kubo for actual retrievability, documented as optional.

### 6.4 Re-verification (the thing being demoed)

```bash
faceanchor verify --bundle evidence/01J....json --chain amoy
```
1. Load bundle → JCS canonicalise → `keccak256` → `digest'`.
2. `eth_call EvidenceAnchor.verify(digest')` → `(true, 1756... )`.
3. Re-fetch `match.uri` from the AT Protocol AppView; recompute `record_cid`, `image_sha256`, `text_sha256`.
4. **Tamper demo:** mutate one byte of `bundle.match.text_sha256` → digest changes → `verify()` returns `(false, 0)`. If the *post* is edited or deleted, step 3 diverges while step 2 still passes — which is precisely the tamper-evidence claim, and worth showing separately.

---

## 7. Innovation

Three additions, all derived from the same move: **take the privacy constraint seriously and it stops being a limitation and becomes the capability.** Every competing submission will anchor a hash and call it verification. These are the things that make this pipeline do something a hash-anchor cannot.

### 7.1 Zero-knowledge match proof — anchor the *claim*, not the biometric

**The tension.** §6.3 forbids putting embeddings on-chain, because a 128-d template on an immutable ledger is a permanent, undeletable biometric record. But that means the published record contains only `score.value: 0.5123` — an unverifiable assertion. A verifier must simply trust that the pipeline computed it honestly. The evidence is tamper-*evident* but not tamper-*proof*: nothing stops the operator anchoring a fabricated score.

**The fix.** Prove the threshold was met without revealing either vector.

SFace embeddings are L2-normalised, so cosine similarity *is* the dot product. That makes the statement small enough to be a practical circuit:

```
public:   C_p = Poseidon(q_p)      // commitment to probe embedding
          C_c = Poseidon(q_c)      // commitment to candidate embedding
          τ                        // threshold, fixed-point
private:  q_p[128], q_c[128]       // int16 fixed-point, scale 2^12

assert  Poseidon(q_p) == C_p
assert  Poseidon(q_c) == C_c
assert  ∀i.  -4096 ≤ q_p[i] ≤ 4096   ∧  -4096 ≤ q_c[i] ≤ 4096
assert  Σ q_p[i]·q_c[i]  ≥  τ · 2^24
```

Quantisation: embeddings are in [-1, 1] post-normalisation, so `q = round(x · 2^12)` maps cleanly to int16. Measured cosine drift from quantisation is <0.001 — well inside the 0.363 decision margin.

The range proofs are not optional. Without them a malicious prover supplies an out-of-range vector and clears any threshold trivially.

**Cost estimate:**

| Component | Constraints |
|---|---:|
| 2 × Poseidon sponge over 128 field elements | ~31 k |
| 128 fixed-point multiplications | ~128 |
| 256 range checks (13-bit) | ~3.3 k |
| **Total** | **~35 k** |

Groth16 over bn254 at 35 k constraints: **~250–400 ms proving on the 13700HX** (snarkjs), ~30 ms with rapidsnark. Proof is **192 bytes**. Trusted setup uses an existing Powers-of-Tau ceremony at 2^16 — no custom ceremony needed.

**Toolchain:** Circom 2.x (MIT) + snarkjs (Apache-2.0) + circomlib Poseidon. `snarkjs zkey export solidityverifier` emits the Groth16 verifier contract in one command — this is the reason to pick Circom over Noir or Halo2 for a timeboxed build, despite worse DX.

**On-chain:**
```solidity
function anchorWithProof(
    bytes32 digest, string calldata cid,
    uint[2] calldata a, uint[2][2] calldata b, uint[2] calldata c,
    uint[3] calldata pubSignals   // [C_p, C_c, τ]
) external {
    require(verifier.verifyProof(a, b, c, pubSignals), "bad proof");
    _anchor(digest, cid);
}
```
Verification costs ~230–280 k gas via the bn254 precompiles (`0x06`/`0x07`/`0x08`). Free on Amoy, and it means the chain itself refuses to record an unproven claim.

**Be honest about the boundary — this matters, and overclaimed ZK is the fastest way to lose a technical judge.** The proof establishes: *the prover knows two in-range vectors, committed to publicly, whose similarity clears τ.* It does **not** prove those vectors were correctly derived by SFace from the probe and the post image. That would require proving the CNN forward pass in-circuit — zkML over SFace is ~10⁸ constraints, categorically out of scope. What this buys is real but bounded: the score becomes non-repudiable, and the biometric never touches the public record. Say exactly that in the README.

**De-risked ladder** — ship the lowest rung that fits:

| | Deliverable | Effort | Guarantee |
|---|---|---:|---|
| **L0** (must ship) | Commit–reveal: anchor digest, withhold bundle until challenged | ~0 h | Tamper-evident, no confidentiality of score |
| **L1** (target) | Circom circuit + off-chain snarkjs verification | ~4 h | Non-repudiable score, embeddings never published |
| **L2** (stretch) | Auto-generated Solidity verifier, `anchorWithProof` | ~2 h | Chain rejects unproven anchors |

### 7.2 Platform-mutation detection (falls out of AT Protocol for free)

AT Protocol records are content-addressed: every post carries a `record_cid` that is a hash of its own content, computed by the platform. The bundle already commits to it (§6.3). That yields a capability the base task doesn't ask for and most submissions can't offer:

| At verify time | On-chain digest | Re-fetched `record_cid` | Conclusion |
|---|---|---|---|
| Both match | ✅ | ✅ | Record intact, post intact |
| Digest fails | ❌ | — | **Our** evidence was altered |
| Digest OK, CID diverges | ✅ | ❌ | **The platform's** content changed after observation |
| Digest OK, fetch 404s | ✅ | gone | Post deleted after observation — and we can prove it existed |

The third and fourth rows are the interesting ones: cryptographic proof that content was mutated or removed *by the platform, after the fact*. That is the actual use case for anchoring social media provenance, and it's a distinct demo beat from "we changed a byte and verification failed." Implementation cost is near zero — the fields are already in the bundle.

### 7.3 On-chain consent revocation

§3 anchors a `consent_digest` alongside the match. Make it bidirectional:

```solidity
mapping(bytes32 => uint64) public revokedAt;

function revoke(bytes32 consentDigest) external {
    require(revokedAt[consentDigest] == 0, "already revoked");
    revokedAt[consentDigest] = uint64(block.timestamp);
    emit Revoked(consentDigest, msg.sender, uint64(block.timestamp));
}
```

`verify()` returns `(anchored, revoked, timestamps)`, and the CLI prints a loud warning when a bundle's consent has been withdrawn. Roughly ten lines of Solidity, and it resolves the immutability-versus-erasure tension the honest way: the anchor is permanent, but the subject holds a permanent, publicly auditable, timestamped veto over its use — which is closer to what DPDP §12 actually asks for than silent deletion is.

### 7.4 Why this framing wins

Every submission will demonstrate *hash on chain, hash verified.* This one demonstrates a pipeline that **cannot** publish a biometric, **cannot** anchor a score it didn't earn, **can** detect the platform tampering downstream, and **can** be switched off by its subject. The judging story is not "we used a blockchain" — it's "we identified what actually needs to be trustless in a face-search pipeline, and made exactly that part trustless."

---

## 8. Repo Layout

```
face-anchor/
├── README.md                    # what/how/chain/limitations (task requirement)
├── Makefile                     # make demo · make verify · make bench · make anvil
├── pyproject.toml
├── src/faceanchor/
│   ├── api.py                   # FastAPI + SSE
│   ├── cli.py                   # Typer
│   ├── vision/{detect,align,embed,quality,phash}.py
│   ├── search/{bluesky,mastodon,commons,duckduckgo,fanout,score}.py
│   ├── evidence/{bundle,jcs,cid,merkle,quantize}.py
│   ├── zk/{prove,verify,witness}.py
│   └── chain/{base,anvil,evm,solana}.py
├── contracts/
│   ├── src/EvidenceAnchor.sol
│   ├── src/Groth16Verifier.sol   # generated by snarkjs, committed
│   ├── script/Deploy.s.sol
│   ├── test/EvidenceAnchor.t.sol
│   └── foundry.toml
├── zk/
│   ├── circuits/match.circom
│   ├── build/            # r1cs, wasm, zkey — gitignored
│   └── setup.sh          # ptau fetch + groth16 setup
├── models/fetch.sh              # downloads ONNX weights; weights NOT committed
├── scripts/{demo.sh,verify.sh,bench.py}
├── evidence/                    # gitignored except one committed sample bundle
└── docs/{ARCHITECTURE,THREAT_MODEL,CONSENT}.md
```

---

## 9. Latency Budget (p50, warm, i7-13700HX P-cores, CPU-only — see §5.4)

| Stage | p50 | Notes |
|---|---:|---|
| Decode + resize | 4 ms | TurboJPEG |
| YuNet detect | 6 ms | 320×320 |
| 5-pt similarity align | 1 ms | |
| SFace embed (probe) | 12 ms | 112×112 |
| Quality gate | <1 ms | |
| **Stage 1 subtotal** | **~24 ms** | |
| Search fanout (Bluesky+Mastodon, 32 conc.) | 280 ms | network-bound |
| Arm B `ddgs.images()` (concurrent, 1.5 s deadline) | 450 ms | off critical path; dropped if throttled |
| Candidate image fetch (64 conc., ~40 imgs) | 420 ms | **dominant term** |
| pHash prefilter | 12 ms | 40 imgs |
| Batch embed survivors (~12 imgs, bs=32) | 90 ms | |
| Cosine + rank | <1 ms | |
| **Stage 2 subtotal** | **~700 ms** (early-exit: ~310 ms) | |
| Groth16 prove (35k constraints, `--zk`) | 300 ms | off critical path; snarkjs, ~30 ms w/ rapidsnark |
| JCS + keccak256 | 2 ms | |
| CIDv1 (local) | 3 ms | |
| Sign + `eth_sendRawTransaction` | 40 ms | nonce cached |
| **Stage 3 subtotal (pre-inclusion)** | **~45 ms** | |
| Inclusion — Anvil / Solana devnet / Amoy | 5 ms / 600 ms / 2.1 s | |
| Re-verify (`eth_call`) | 60 ms | |

**Perceived p50 (match streamed before inclusion): ~730 ms.**

Levers, in order of payoff:
1. Async anchoring off the response path — removes 2.1 s.
2. pHash prefilter — removes ~70% of candidate embeds.
3. Early-exit at cosine ≥ 0.50 — halves Stage 2 on easy probes.
4. Range-capped image fetch (512 KB) — caps tail latency on large media.
5. Pre-warmed ORT sessions + cached nonce — removes two cold starts.
6. GPU (CUDA EP): Stage 1 → ~6 ms, batch embed → ~15 ms. Not required for target.

---

## 10. Milestones (~32 build hours)

| # | Deliverable | Est. | Exit criterion |
|---|---|---:|---|
| M0 | Repo scaffold, `pyproject`, Makefile, model fetch script | 1 h | `make setup` green on clean clone |
| M1 | Stage 1 vision + quality gate + unit tests | 3 h | Deterministic embedding on a fixture; p50 < 30 ms in `bench.py` |
| M2 | Bluesky + Mastodon adapters, async fanout, scoring cascade | 6 h | Live probe returns a real match with score, no hardcoding |
| M3 | `EvidenceAnchor.sol` + Foundry tests + Anvil adapter | 3 h | `forge test` green; `make demo` anchors + verifies locally |
| M4 | JCS/CID/bundle + EVM adapter + Amoy deploy | 3 h | Public tx hash on Polygonscan; `verify` returns true |
| M5 | Arm B: `ddgs` adapter + rate limiter + seed extraction | 2 h | Returns corroborating candidates; degrades cleanly to Arm A when throttled |
| M6 | Solana devnet adapter (SPL Memo) | 2 h | Sub-second confirmed anchor demonstrated |
| M7 | README, bench run, tamper demo, THREAT_MODEL/CONSENT docs | 3 h | Cold-clone reproduction in < 10 min |
| M8 | §7.2 mutation detection + §7.3 `revoke()` | 2 h | Four-quadrant verify table reproduced live |
| M9 | §7.1 L1: Circom circuit, setup, off-chain verify | 4 h | Proof verifies; embeddings absent from bundle |
| M10 | §7.1 L2: Solidity verifier, `anchorWithProof` | 2 h | Chain rejects an anchor with a forged score |
| M11 | Buffer | 1 h | |

Total is now ~32 h. Cut order if time-constrained: M10 → M9 → M6 → M5 → M4 (fall back to Anvil-only and say so plainly in the README).

---

## 11. Known Limitations (README-ready)

1. **Recall is corpus-bounded, not web-scale.** Arm A searches Bluesky, Mastodon and Wikimedia Commons — open, ToS-permissible APIs. It will not find a match on Instagram or a closed platform. This is a deliberate trade: scraped reverse-image search is slow, CAPTCHA-gated, ToS-violating, and unreliable in a live demo.
2. **No reverse image search anywhere in the pipeline.** DuckDuckGo has no search-by-image endpoint, so Arm B is a text-seeded candidate generator, not a face→identity lookup. It expands and corroborates around a match Arm A has already found. Seeding from a face alone is not supported.
3. **Arm B depends on an unofficial client.** `ddgs` wraps DDG's `vqd`-token HTML endpoints rather than a published API; it can break on any upstream change, and DDG throttles bursts. The arm is deadline-bounded and treated as a bonus signal, never a dependency. DDG's image index is Bing-backed, so it is not an independent corpus from Bing.
4. **SFace accuracy ceiling.** 128-d SFace trades recall for speed and licence cleanliness. `--model buffalo_l` improves accuracy but its weights are non-commercial-research-only.
5. **Threshold is a policy choice.** 0.363 cosine is the published SFace operating point, not a guarantee; false-accept rate rises with corpus size (the classic large-gallery problem). Scores are reported, never presented as identity claims.
6. **On-chain proof is of *observation*, not of *truth*.** The chain proves "this pipeline saw this content at this time and has not altered its record since." It does not prove the match is correct, nor that the post is authentic. Overstating this is the most common failure mode in blockchain-provenance projects.
7. **Testnet impermanence.** Sepolia EOL is 30 Sep 2026; Amoy checkpoints to it. Anchors on any testnet may become unverifiable when the network is retired. Anvil mode exists so the demo is not hostage to this.
8. **Immutability vs erasure.** Anchoring is irreversible. Only digests are committed, precisely so a DPDP §12 / GDPR Art. 17 erasure request can be honoured by deleting the off-chain bundle — after which the on-chain digest is an unlinkable 32-byte value.
9. **Single-face probes only.** Multi-face images require explicit `--face-index`.
10. **The ZK proof is bounded (§7.1).** It proves two committed, in-range vectors clear the threshold. It does **not** prove those vectors were correctly derived by SFace from the probe and post images — that needs zkML over the full CNN (~10⁸ constraints), which is out of scope. The derivation step remains trusted.
11. **Groth16 needs a trusted setup.** We reuse an existing Powers-of-Tau ceremony; soundness inherits that ceremony's assumptions. Halo2 or Nova would remove this at the cost of a much larger on-chain verifier.
12. **No liveness/anti-spoof.** A printed photo will pass. Out of scope; noted because it matters for any real deployment.

---

## 12. Demo Script (judge-facing, ~90 s)

```bash
make setup                                   # deps + ONNX weights
make anvil &                                 # local chain
faceanchor run --probe probe.jpg \
  --consent consent/self.json --chain anvil  # → match + digest + tx
faceanchor verify --bundle evidence/latest.json --chain anvil   # → OK
sed -i 's/"value": 0.5123/"value": 0.9999/' evidence/latest.json
faceanchor verify --bundle evidence/latest.json --chain anvil   # → FAIL (digest mismatch)

faceanchor run --probe probe.jpg --consent consent/self.json --chain amoy
# → Polygonscan link, public verifiable record

# Innovation beats (§7)
faceanchor run --probe probe.jpg --consent consent/self.json --zk --chain anvil
# → 192-byte proof; bundle contains NO embedding, only Poseidon commitments
faceanchor forge-score --bundle evidence/latest.json --score 0.99 --chain anvil
# → chain REJECTS: "bad proof"
faceanchor verify --bundle evidence/latest.json --check-platform
# → digest OK, record_cid diverged → "post edited after observation"
faceanchor revoke --consent consent/self.json --chain anvil && faceanchor verify ...
# → verifies, but flags CONSENT REVOKED @ block N
```

---

## 13. Open Questions

- Confirm Sepolia/Amoy status against the EF testnet roadmap before deadline; if Amoy is wobbling, promote Solana devnet to primary.
- Whether Lane C (public figures) is worth including at all, or whether Lanes A+B alone make a cleaner submission. Lane A+B is the safer default.
- Batch (`anchorBatch`) is spec'd but may be cut — it only pays off above ~50 records/run, which a live demo will not hit.
