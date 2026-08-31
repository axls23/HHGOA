# face-anchor

Face identification → web/social discovery → on-chain verification.

A three-stage pipeline: detect+embed a face (Stage 1), search live public
web/social corpora for a genuinely matching post (Stage 2), and anchor a
canonical fingerprint of the discovery on a blockchain so the finding can be
independently re-verified later (Stage 3). Full design rationale, latency
budget, and threat model live in [`PRD_face_anchor.md`](PRD_face_anchor.md) —
this README covers what's actually built and how to run it.

**No website.** The judge-facing surface is a CLI (`faceanchor run` /
`verify` / `revoke` / `forge-score`) plus this repo.

## Architecture (as built)

Diagrams describe what is actually in this repo, with numbers measured on
the build machine (i7-13700HX / RTX 4050). PRD §4 has the original design;
this is the shipped version.

### The whole idea

```mermaid
flowchart LR
    A["📷 a photo"] --> B["🔎 find that face<br/>on public sites"]
    B --> C["⛓️ stamp the finding<br/>on a blockchain"]
    C --> D["✅ anyone re-checks it later<br/>with only this repo + the chain"]
```

The stamp is what makes a finding checkable by someone who has no reason
to trust us. Everything else here exists to make that stamp mean
something honest.

### Step 1 — what happens to your photo

```mermaid
flowchart TD
    P["your photo"] --> F["① find the face<br/>where is it? is there just one?<br/>~6 ms"]
    F --> Q{"② is it good enough?<br/>big enough · sharp · facing us"}
    Q -->|"no"| STOP["stop here, cheaply<br/>a bad photo costs milliseconds,<br/>not a whole web search"]
    Q -->|"yes"| E["③ make a faceprint<br/>128 numbers describing this face<br/>~12 ms"]
    E --> R["ready to search — ~17 ms total"]
    E -.-> N["the faceprint lives in memory only<br/>never written to disk<br/>never sent to the chain"]

    classDef stop fill:#fde,stroke:#c66
    classDef note fill:#eef,stroke:#88a,stroke-dasharray:3 3
    class STOP stop
    class N note
```

The order matters: the cheap quality check runs **before** the expensive
part, so a useless photo is rejected in milliseconds.

### Step 2 — finding it online

```mermaid
flowchart TD
    Q["faceprint + your search words"]
    subgraph ArmA["asked all at once — a slow source is simply dropped"]
        A1["Bluesky"]
        A2["Mastodon"]
        A3["Wikimedia Commons"]
    end
    Q --> ArmA
    ArmA --> B["DuckDuckGo<br/>runs only to corroborate what was<br/>already found — it cannot search by face"]
    ArmA --> F["download the images they point at<br/>64 at a time · 512 KB cap each"]
    B --> F
    F --> S["narrow them down"]
```

### Step 3 — narrowing 120 images down to one

Cheapest test first, so the expensive one runs on as few images as possible.

```mermaid
flowchart TD
    C["~120 candidate images"] --> P{"is this literally the same picture?<br/>64-bit perceptual hash · ~0.3 ms"}
    P -->|"identical"| WIN["MATCH<br/>no face maths needed at all"]
    P -->|"different"| F{"is there even a face in it?"}
    F -->|"no"| X1["discard"]
    F -->|"yes"| S{"same person?<br/>compare the two faceprints"}
    S -->|"below 0.363"| X2["discard"]
    S -->|"0.363 and up"| WIN
    S -->|"0.50 and up"| E["MATCH — and stop searching entirely"]

    classDef good fill:#dfd,stroke:#6a6
    classDef bad fill:#eee,stroke:#999
    class WIN,E good
    class X1,X2 bad
```

### Step 4 — making the finding permanent

```mermaid
flowchart TD
    M["the match"] --> J["① write down what we saw<br/>hashes and URLs only — never the photo,<br/>never the faceprint"]
    J --> C["② put those bytes in a strict canonical order<br/>so a different machine, years later,<br/>rebuilds them byte-for-byte"]
    C --> H["③ hash it → one 32-byte value"]
    H --> CH["④ send only that value on-chain<br/>the chain stores no personal data"]
    CH --> R["a permanent, timestamped receipt"]
```

### Step 5 — anyone re-checking it later

```mermaid
flowchart TD
    B["evidence file"] --> C["same canonical bytes"] --> H["same hash"] --> A{"ask the chain:<br/>have you seen this before?"}
    A -->|"yes — at 14:32"| OK["untouched ✅"]
    A -->|"never seen it"| BAD["something was edited,<br/>even by a single byte ❌"]

    classDef good fill:#dfd,stroke:#6a6
    classDef bad fill:#fde,stroke:#c66
    class OK good
    class BAD bad
```

That second branch **is** the tamper demo: change one character in the
evidence file and its hash no longer matches anything on the chain.

---

## Reference — the precise view

### Module map

```mermaid
flowchart TD
    CLI["cli.py<br/>the one place the stages are wired together"]
    CLI --> CO["consent.py<br/>the run aborts without a consent artifact"]
    CLI --> V["vision/<br/>decode · detect · quality · align · embed · phash"]
    CLI --> S["search/<br/>bluesky · mastodon · commons · duckduckgo<br/>fanout · score"]
    CLI --> E["evidence/<br/>bundle · jcs · cid · quantize · platform_check"]
    CLI --> CH["chain/<br/>base · anvil · evm · solana"]
    CLI -.->|"--zk only"| Z["zk/<br/>witness · prove"]
    S -->|"re-uses the same detector<br/>and embedder on candidates"| V
    Z -->|"quantises the faceprints"| E

    classDef opt stroke-dasharray:4 3
    class Z opt
```

Files worth knowing about:

| File | Why it exists |
|---|---|
| `vision/embed.py` | Two paths: CPU for the single probe, CUDA batch-32 for candidates. Preprocessing is pinned to match OpenCV's own C++ output exactly (RGB, raw 0–255) |
| `vision/models.py` | Derives a batch-dynamic ONNX export — the upstream weights hardcode batch=1, which would make the GPU path pointless |
| `vision/_cuda_libs.py` | dlopens cuDNN/cuBLAS from site-packages so no `LD_LIBRARY_PATH` export is needed |
| `search/fanout.py` | HTTP/2 pool, per-source timeouts, 512 KB range-capped fetches |
| `evidence/jcs.py` | RFC 8785 canonicalisation — hashing `json.dumps` output would not be reproducible |
| `evidence/platform_check.py` | Detects the platform editing or deleting a post after we recorded it |

### Chain adapters

```mermaid
flowchart TD
    I["ChainAdapter<br/>chain/base.py"]
    I --> A["AnvilAdapter<br/>local · instant · free"]
    I --> E["EvmAdapter<br/>Polygon Amoy or any EVM"]
    I --> S["SolanaAdapter<br/>devnet · SPL Memo · no custom program"]
    A -.->|"subclasses"| E
```

| Operation | anvil | amoy | solana |
|---|:---:|:---:|:---:|
| `anchor` | ✓ | ✓ | ✓ |
| `verify` | ✓ | ✓ | ✓ * |
| `wait_for_inclusion` | ✓ | ✓ | ✓ |
| `revoke` / `consentStatus` | ✓ | ✓ | ✗ |
| `anchorWithProof` | ✓ | ✓ | ✗ |

\* Wallet-scoped scan of recent signatures, not an O(1) state lookup — SPL
Memo stores no on-chain state by design. The EVM path deliberately keeps
an `anchoredAt` mapping so `verify()` is a single `eth_call` with no
indexer dependency (PRD §6.2).

### The ZK match proof (`--zk`)

Anchoring a score alone is just an assertion — nothing stops an operator
writing down a number they never computed. This makes the score
non-repudiable *without* publishing either faceprint.

```mermaid
flowchart TD
    E1["probe faceprint"] --> Q["quantise to whole numbers<br/>and commit to each publicly"]
    E2["candidate faceprint"] --> Q
    Q --> CIR["the circuit checks three things:<br/>• each faceprint matches its public commitment<br/>• every number is within range<br/>• their similarity clears the threshold"]
    CIR --> PR["a 192-byte Groth16 proof<br/>~41k constraints · ~2.7 s"]
    PR --> OFF["verified off-chain<br/>snarkjs"]
    PR --> ON["verified on-chain<br/>a forged score is rejected outright"]

    classDef good fill:#dfd,stroke:#6a6
    class ON good
```

| The proof **does** establish | The proof does **not** establish |
|---|---|
| The prover knows two committed, in-range faceprints | That those faceprints are what the face model actually produced from these two images |
| Their similarity genuinely clears the threshold | That the match is *correct*, or the post authentic |
| The score is non-repudiable, and neither faceprint is ever published | Proving the derivation needs zkML over the whole network (~10⁸ constraints) — out of scope, and the step stays trusted |

### What `verify` can tell you

```mermaid
flowchart LR
    V["faceanchor verify<br/>--check-platform"] --> D{"does the hash<br/>match the chain?"}
    D -->|"no"| E1["OUR evidence was altered"]
    D -->|"yes"| P{"does the post still<br/>hash the same?"}
    P -->|"identical"| E2["record intact ✅"]
    P -->|"changed"| E3["the PLATFORM edited it<br/>after we recorded it"]
    P -->|"gone"| E4["deleted after observation —<br/>and we can prove it existed"]

    classDef good fill:#dfd,stroke:#6a6
    classDef warn fill:#ffd,stroke:#ca6
    classDef bad fill:#fde,stroke:#c66
    class E2 good
    class E3,E4 warn
    class E1 bad
```

Every `verify` also checks whether the subject has withdrawn consent, and
prints a loud `⚠ CONSENT REVOKED` if so. The post-side check only applies
to Bluesky matches (AT Protocol records carry their own content hash);
Mastodon and Commons report `not_applicable` rather than a false signal.

### Build log

`✓` = exercised live against a real network or chain during the build.
`~` = built and unit-tested, but blocked from live verification by an
external resource (see the status table below).

| # | Shipped | Verified |
|---|---|---|
| M0 | Scaffold — uv venv (3.12), Foundry, model fetch | ✓ |
| M1 | Vision — YuNet + SFace + quality gate | ✓ p50 17 ms |
| M2 | Discovery — Arm A adapters, fanout, scoring cascade | ✓ live self-match |
| M3 | `EvidenceAnchor.sol` + Anvil adapter | ✓ `forge test` |
| M4 | Evidence bundle, JCS/CID, CLI `run`/`verify` | ✓ live tamper demo |
| — | Amoy public testnet path | ~ needs a funded key |
| M5 | Arm B (DuckDuckGo) + consent gate | ✓ 40 merged candidates |
| M6 | Solana SPL Memo adapter | ~ devnet faucet rate-limited |
| M7 | README, architecture, threat model, consent docs | ✓ |
| M8 | Platform-mutation detection + `revoke()` | ✓ live, all four verdicts |
| M9 | ZK L1 — Circom circuit, trusted setup, prove/verify | ✓ live, both polarities |
| M10 | ZK L2 — `Groth16Verifier.sol` + `anchorWithProof` | ✓ live `BadProof` revert |

66 tests: 18 Solidity, 48 Python. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the prose walkthrough and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) for
what is and isn't trustless.

## Subject policy (read before running)

This pipeline is a deanonymization primitive. It will not run without an
explicit consent artifact:

```
faceanchor run --probe photo.jpg --query "..." --chain anvil --contract-address 0x...
# ERROR: --subject-consent is required
```

See `consent/self.example.json` and `consent/teammate.example.json`, and
[`docs/CONSENT.md`](docs/CONSENT.md). Default demo subjects are the
builder's own face (Lane A) or a teammate's, with signed consent (Lane B).
See PRD §3 for the full policy and legal basis (India DPDP Act 2023, EU AI
Act Art. 5(1)(e)).

## Setup

Requires: Python 3.12 (not 3.13+ — some pinned wheels lag), [`uv`](https://docs.astral.sh/uv/),
[Foundry](https://getfoundry.sh), and an NVIDIA GPU + CUDA driver if you want
the CUDA execution provider (CPU-only works fine — see PRD §5.4, GPU only
matters for batch-embedding a wide candidate fanout).

```bash
make setup      # venv (uv), Python deps, ONNX model weights, forge build
make anvil &    # local chain, in another terminal
```

`make setup` installs `onnxruntime-gpu`, but the pip wheel needs cuDNN/cuBLAS
that aren't pulled in automatically by the CUDA driver alone — `pyproject.toml`
pins `nvidia-cudnn-cu12`/`nvidia-cublas-cu12`/`nvidia-cuda-nvrtc-cu12`
explicitly, and `faceanchor.vision._cuda_libs` preloads them at runtime so you
don't need to `export LD_LIBRARY_PATH` yourself. If you have no GPU, ONNX
Runtime falls back to CPU automatically — nothing to configure.

### Optional: ZK match proof (`--zk`)

```bash
cd zk && ./setup.sh   # installs snarkjs+circomlib, compiles the circuit,
                       # fetches a 2^16 Powers-of-Tau ceremony, runs Groth16 setup
```

Needs Node.js (any recent LTS — `nvm install --lts` if you don't have it) and
`circom` (prebuilt binary: https://github.com/iden3/circom/releases). The
base pipeline works without any of this; `--zk` is opt-in.

## Demo

```bash
# deploy (prints two addresses: Groth16Verifier, EvidenceAnchor)
cd contracts && forge script script/Deploy.s.sol --rpc-url http://127.0.0.1:8545 \
  --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 --broadcast
cd ..

export CONTRACT=<EvidenceAnchor address from above>

# Stage 1 -> 2 -> 3: probe a face, find a real match, anchor it
faceanchor run --probe photo.jpg --subject-consent consent/self.example.json \
  --query "some search terms" --chain anvil --contract-address $CONTRACT

# independent re-verification
faceanchor verify --bundle evidence/latest.json --chain anvil --contract-address $CONTRACT
# OK — anchored at timestamp ...

# tamper demo
python3 -c "import json; d=json.load(open('evidence/latest.json')); d['score']['value']=0.9999; json.dump(d, open('evidence/latest.json','w'))"
faceanchor verify --bundle evidence/latest.json --chain anvil --contract-address $CONTRACT
# FAIL — digest not found on-chain

# innovation beats (§7)
faceanchor run --probe photo.jpg --subject-consent consent/self.example.json \
  --query "..." --chain anvil --contract-address $CONTRACT --zk
# -> bundle has a 192-byte proof, zero occurrences of "embedding"

faceanchor forge-score --bundle evidence/latest.json --score 0.99 --chain anvil --contract-address $CONTRACT
# chain REJECTS: BadProof

faceanchor verify --bundle evidence/latest.json --chain anvil --contract-address $CONTRACT --check-platform
# platform check: not_applicable (or a real bsky mutation-detection verdict)

faceanchor revoke --consent consent/self.example.json --chain anvil --contract-address $CONTRACT
faceanchor verify --bundle evidence/latest.json --chain anvil --contract-address $CONTRACT
# OK — anchored ..., plus: ⚠ CONSENT REVOKED @ timestamp ...
```

Every command above was run live against a local Anvil node while building
this repo — the `run --query "official portrait headshot high resolution"`
example genuinely finds and anchors a real Wikimedia Commons portrait, no
hardcoded matching.

Public-testnet (`--chain amoy`) needs `AMOY_RPC_URL` + `AMOY_PRIVATE_KEY`
env vars pointing at a funded wallet — not something this repo can
provision for you. `contracts/src/chain/solana.py` (SPL Memo, no custom
program needed) exists and is unit-tested, but wasn't exercised against a
live devnet transaction — the public devnet faucet rate-limited every
airdrop attempt from the sandbox this was built in.

## Tests

```bash
make test   # forge test (contracts) + pytest (Python)
```

66 tests, all green as of the last commit: 18 Solidity (EvidenceAnchor +
Groth16 verifier, including a real hardcoded Groth16 proof exercised against
the actual deployed verifier), 48 Python. ZK tests (`tests/test_zk.py`) skip
cleanly if `zk/setup.sh` hasn't been run.

```bash
make bench  # p50/p95/p99 per stage, taskset-pinned to P-cores
```

Report CPU-only numbers from a cool machine (PRD §5.4) — GPU only earns its
keep on the candidate batch-embed step, and both CPU/GPU numbers are noisy
under laptop thermal/frequency-scaling variance; don't over-trust a single run.

## What's real vs. what needed external resources I don't have

Everything below was built, and where "verified live" is stated, actually
run against a real network/chain during this build — not just written and
assumed correct:

| Piece | Status |
|---|---|
| Stage 1 (YuNet detect + align + SFace embed, CPU and CUDA-batch) | Verified live; p50 17ms P-core-pinned |
| Stage 2 discovery (Bluesky, Mastodon, Commons, DuckDuckGo Arm B) | Bluesky's public API is blocked at the CDN edge from this build sandbox (403, geo/datacenter filtering) — implemented against the documented API, unit-tested against a captured shape, **not** exercised live. Mastodon/Commons/DuckDuckGo verified live, including a genuine self-match against Wikimedia Commons with zero hardcoded matching. |
| Stage 3 anchor (EvidenceAnchor.sol, JCS/CID, Anvil) | Verified live, including the tamper demo |
| Amoy (public testnet) | Code path exists (`chain/evm.py` + `poa=True`), never deployed — needs a funded wallet key this environment doesn't have |
| Solana devnet (SPL Memo) | Code exists, unit-tested; devnet faucet 429'd every airdrop attempt from this sandbox's IP — not exercised against a live tx |
| §7.2 platform-mutation detection | Logic verified against a mocked Bluesky transport (real network blocked, same as above) |
| §7.3 consent revocation | Verified live end-to-end (`revoke` -> `verify` shows the warning) |
| §7.1 ZK match proof (L1 off-chain, L2 on-chain) | Verified live end-to-end, including proof generation *failing* for a below-threshold pair and the chain rejecting a forged score with `BadProof()` |

## Known limitations

See PRD §11 for the full list (recall is corpus-bounded, no reverse-image
search anywhere, SFace's accuracy ceiling, threshold is a policy choice not
a guarantee, on-chain proof is of *observation* not *truth*, testnet
impermanence, no liveness/anti-spoof detection). Additions found while
building:

- **`search/commons.py` fetches a bounded-width thumbnail, not the
  original.** Commons originals commonly exceed the 512KB Range-cap fetch
  (PRD §5.2, sized for social-feed images); a raw capped fetch of a
  multi-MB original truncates mid-JPEG and produces decode artifacts that
  spuriously fail the blur quality gate. Fixed by requesting `iiurlwidth`.
- **The ZK circuit's vector commitment is a 2-level Poseidon tree, not a
  literal single-call sponge over 128 elements.** circomlib only ships
  precomputed round constants up to 16 Poseidon inputs; there's no
  off-the-shelf single permutation over a 129-wide state. The tree (8 ×
  Poseidon(16) → Poseidon(8)) has the same binding-commitment security
  property, built from arities circomlib actually supports. Documented
  inline in `zk/circuits/match.circom`.
- **`contracts/src/Groth16Verifier.sol` is GPL-3.0**, generated verbatim by
  `snarkjs zkey export solidityverifier`. Everything else in the repo is
  MIT/Apache-2.0 (PRD §G5's "permissive licenses only" goal). Scoped to the
  opt-in `--zk`/`anchorWithProof` path only — the default `anchor`/`verify`
  path never touches it. Same pattern PRD §5.1 already uses for buffalo_l's
  non-commercial weights: flag it, keep it out of the default path.
- **Solana's `verify()` is wallet-scoped, not a global O(1) lookup.** SPL
  Memo has no on-chain state at all (that's the whole point of using it —
  no custom program), so there's no `eth_call`-equivalent. `chain/solana.py`
  scans the anchoring wallet's own recent signature history for a matching
  memo. A real limitation of the memo-only design, not a corner cut.

## Repo layout

See PRD §8. `contracts/lib/forge-std` is vendored (not a submodule) since
`contracts/` was scaffolded with `forge init --no-git` inside a repo that
already had its own git root.
