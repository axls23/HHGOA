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
