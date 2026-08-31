# Architecture

Three stages, see PRD §4 for the full diagram and §9 for the latency budget.

- **Stage 1 (`src/faceanchor/vision/`)**: YuNet detect (bbox + 5-pt
  landmarks in one pass) → `FaceRecognizerSF.alignCrop` → SFace embed.
  Single-probe embed runs CPU (raw ONNX Runtime, thread-tuned
  `SessionOptions`, not `cv2.FaceRecognizerSF.feature()` — see `embed.py`'s
  docstring for the measured ~38ms vs ~17ms p50 difference). Candidate
  batch embed runs CUDA EP against a batch-dynamic export of the same
  weights (`models.py::ensure_sface_dynbatch_onnx` — the upstream OpenCV
  Zoo export hardcodes batch=1).
- **Stage 2 (`src/faceanchor/search/`)**: `fanout.py` runs Arm A (Bluesky +
  Mastodon + Commons, concurrent, per-source timeout) then Arm B
  (`duckduckgo.py`, seeded from Arm A's best hit). `score.py` runs the
  pHash → detect-or-drop → batch-cosine cascade from PRD §5.2.
- **Stage 3 (`src/faceanchor/evidence/` + `src/faceanchor/chain/`)**:
  `bundle.py` builds the PRD §6.3 schema, `jcs.py` canonicalizes + hashes
  it (RFC 8785, keccak256), `cid.py` computes a local CIDv1. `chain/base.py`
  is the adapter interface; `anvil.py`/`evm.py` implement it for EVM chains,
  `solana.py` for SPL Memo.

`cli.py` is the only place these three stages get wired together — it's
intentionally the sole caller of `search.fanout`, `evidence.bundle`, and
`chain.*` in the same function, so the actual pipeline order is readable in
one file rather than spread across a service layer that doesn't exist yet
(no FastAPI/SSE layer is built — PRD §5.3 specs one, but the CLI is the
judge-facing surface per the task's "no website" constraint, so it wasn't
built out this pass).

`src/faceanchor/zk/` (witness.py, prove.py) shells out to `node` +
`snarkjs` rather than reimplementing Groth16 — see PRD §7.1's own
reasoning for picking Circom/snarkjs over Noir/Halo2 specifically because
`zkey export solidityverifier` is one command.
