# Threat model

Full framing in PRD §7.4 and §11 (Known Limitations) — this is the short
version: what this system does and doesn't protect against.

## What's trustless

- **Tamper-evidence of the recorded observation.** Once a bundle's digest is
  anchored, any change to the bundle (score, match URI, hashes) changes the
  digest and `verify()` fails. Demonstrated live in the README's tamper demo.
- **Platform-side mutation, for AT Protocol matches.** Since Bluesky records
  are self-content-addressed, re-fetching `record_cid` at verify time and
  comparing against the anchored value distinguishes "our evidence was
  altered" from "the platform's content changed after observation" from
  "the post was deleted" (`evidence/platform_check.py`).
- **The claimed similarity score, when `--zk` is used.** `anchorWithProof`
  makes the chain itself refuse an anchor whose public signals don't
  correspond to a real Groth16 proof — verified live (`forge-score`
  triggers a `BadProof()` revert).
- **Consent state.** `revoke()` is permanent and public; a bundle whose
  consent has been revoked is flagged loudly by `verify`.

## What's explicitly NOT trustless (and why)

- **The derivation step.** The ZK proof establishes that the prover knows
  two in-range, committed vectors whose dot product clears a threshold — it
  does **not** prove those vectors came from running SFace on the actual
  probe/candidate images. Proving that would need zkML over the full CNN
  forward pass (~10^8 constraints, PRD §7.1) — out of scope. An operator
  could in principle feed the circuit fabricated (but well-formed) vectors.
  This is the same trust boundary every "proof of ML inference" system in
  this constraint class has; it's stated here rather than glossed over.
- **Discovery itself.** Nothing prevents an operator from running the
  pipeline against a search query chosen to surface a specific pre-known
  result. The tamper-evidence guarantee is about the *bundle*, not about
  operator intent — "this pipeline saw this content at this time," not
  "this match is correct" (PRD §11, item 6).
- **Off-chain availability.** The evidence bundle itself isn't required to
  be retrievable for `verify()` to work (only its digest is checked
  on-chain), but `--check-platform` and any human audit of *why* a match
  was accepted needs the bundle file. It's gitignored by default
  (deliberately, per PRD §6.3's erasure story) — losing it doesn't break
  chain-level tamper-evidence, but it does break auditability.
- **Sybil/relay attacks on discovery sources.** Arm A/B fetch from public
  APIs over plain HTTPS with no content-provenance guarantee beyond TLS —
  a compromised or malicious intermediary serving fabricated "matches" from
  a spoofed-but-valid-looking source isn't detected. Out of scope for this
  build; would need per-platform signature verification (AT Protocol's own
  record signing, which isn't currently checked here beyond the
  content-addressing `record_cid` comparison).

## Key management

Anvil's well-known dev key #0 is hardcoded in `chain/anvil.py` — by design,
never used against a real network, and Anvil funds it automatically. Amoy
and Solana devnet keys are read from environment variables
(`AMOY_PRIVATE_KEY`), never committed, never logged.
