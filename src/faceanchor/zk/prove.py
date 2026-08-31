"""Groth16 proof generation/verification for the match circuit (PRD §7.1).

Shells out to snarkjs (Apache-2.0) rather than reimplementing Groth16 —
that's the whole reason PRD picks Circom/snarkjs over Noir/Halo2 for a
timeboxed build: `zkey export solidityverifier` (M10) is one command.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from faceanchor.zk.witness import VERIFICATION_KEY_PATH, ZK_DIR, ZKEY_PATH, _require, compute_witness

SNARKJS_LOCAL_BIN = ZK_DIR / "node_modules" / ".bin" / "snarkjs"


@dataclass(frozen=True)
class Proof:
    proof: dict
    public_signals: list[str]  # [probeCommitment, candCommitment, threshold] — circuit output order


def _snarkjs_bin() -> str:
    if SNARKJS_LOCAL_BIN.exists():
        return str(SNARKJS_LOCAL_BIN)
    found = shutil.which("snarkjs")
    if found:
        return found
    raise FileNotFoundError(f"snarkjs not found at {SNARKJS_LOCAL_BIN} or on PATH — run `npm install` in zk/.")


def generate_proof(circuit_input: dict, work_dir: Path | None = None) -> Proof:
    _require(ZKEY_PATH)
    witness_path = compute_witness(circuit_input, out_dir=work_dir)
    work_dir = witness_path.parent
    proof_path = work_dir / "proof.json"
    public_path = work_dir / "public.json"

    subprocess.run(
        [_snarkjs_bin(), "groth16", "prove", str(ZKEY_PATH), str(witness_path), str(proof_path), str(public_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    return Proof(proof=json.loads(proof_path.read_text()), public_signals=json.loads(public_path.read_text()))


def verify_proof(proof: Proof) -> bool:
    _require(VERIFICATION_KEY_PATH)
    with tempfile.TemporaryDirectory(prefix="faceanchor_zk_verify_") as tmp:
        public_path = Path(tmp) / "public.json"
        proof_path = Path(tmp) / "proof.json"
        public_path.write_text(json.dumps(proof.public_signals))
        proof_path.write_text(json.dumps(proof.proof))

        result = subprocess.run(
            [_snarkjs_bin(), "groth16", "verify", str(VERIFICATION_KEY_PATH), str(public_path), str(proof_path)],
            capture_output=True,
            text=True,
        )
    return "OK!" in result.stdout
