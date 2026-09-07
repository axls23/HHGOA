# Consent

Full policy rationale in PRD §3. This is the practical how-to.

## Hard rules, enforced in code

- `--subject-consent <path>` is required on every `faceanchor run` and every
  `faceanchor corpus-footprint` — `src/faceanchor/consent.py::load_consent_digest`
  raises and the CLI aborts before touching any embedding logic if it's
  missing. (`faceanchor revoke` takes the same artifact as `--consent`: it
  needs the digest to revoke.) `corpus-build` is the deliberate exception —
  it ingests strangers' public posts and never touches a subject's face, so
  a subject consent artifact would be the wrong thing to ask for; what
  governs it is the explicit-content filter and the corpus specs you name.
- Only the artifact's JCS-canonicalized SHA-256 digest is ever used — the
  artifact contents themselves never leave your machine, never get hashed
  into anything retrievable from the chain, and are gitignored
  (`consent/*.json` except the `*.example.json` templates).
- No embedding is ever written to disk or chain — see `vision/embed.py`
  and `evidence/bundle.py`'s docstrings for where this is enforced.

## Lanes

- **Lane A (default)**: your own face. Use `consent/self.example.json` as a
  template — copy it to `consent/self.json`, fill in your own
  name/email/timestamp, and it's gitignored automatically.
- **Lane B**: a teammate's face, with their signed consent. Use
  `consent/teammate.example.json` as a template. "Signed" here means the
  artifact's content, hashed and anchored — this repo doesn't implement
  cryptographic signature verification of the artifact itself (e.g. a
  detached PGP/ECDSA signature you could verify independently); the
  consent_digest anchored on-chain proves *an* artifact with that exact
  content existed at anchor time, not that a specific named person signed
  it with a verifiable key. Strengthening that (e.g. requiring the
  artifact to include a signature over its own JCS digest, verified before
  the run proceeds) is a reasonable extension not built in this pass.
- **Lane C**: a public figure with an already-public account. PRD §13
  flags this as possibly worth cutting for a cleaner submission — Lane A/B
  alone is the safer default and what the demo script in the README uses.

## Revocation

```bash
faceanchor revoke --consent consent/self.example.json --chain anvil --contract-address $CONTRACT
```

Irreversible in the sense that the *revocation itself* is a permanent,
public, timestamped fact (PRD §7.3) — but from that point on, every
`faceanchor verify` against a bundle referencing that consent digest prints
a loud `⚠ CONSENT REVOKED` warning. The underlying anchor can't be deleted
(it's a hash on an immutable ledger), but its continued validity is now
publicly disputed. This is the DPDP §12 / GDPR Art. 17 resolution PRD §7.3
argues for: the anchor is permanent, but the subject holds a permanent,
auditable veto over its use.
