"""Shared paths + circuit input construction for the match-proof circuit (PRD §7.1)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from faceanchor.evidence.quantize import quantize, shift_unsigned

ZK_DIR = Path(__file__).resolve().parents[3] / "zk"
BUILD_DIR = ZK_DIR / "build"
WASM_PATH = BUILD_DIR / "match_js" / "match.wasm"
GENERATE_WITNESS_JS = BUILD_DIR / "match_js" / "generate_witness.js"
ZKEY_PATH = BUILD_DIR / "match_final.zkey"
VERIFICATION_KEY_PATH = BUILD_DIR / "verification_key.json"

COSINE_TO_DOT_SCALE = 2**24  # matches the circuit's fixed-point product scale (2^12 * 2^12)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `zk/setup.sh` first.")
    return path


def find_node() -> str:
    node = shutil.which("node")
    if node:
        return node
    nvm_dir = Path.home() / ".nvm" / "versions" / "node"
    if nvm_dir.exists():
        candidates = sorted(nvm_dir.iterdir(), reverse=True)
        for c in candidates:
            binary = c / "bin" / "node"
            if binary.exists():
                return str(binary)
    raise FileNotFoundError("node not found on PATH or under ~/.nvm — install Node.js first.")


def build_circuit_input(probe_embedding, cand_embedding, cosine_threshold: float) -> dict:
    probe_shifted = shift_unsigned(quantize(probe_embedding))
    cand_shifted = shift_unsigned(quantize(cand_embedding))
    threshold_scaled = int(round(cosine_threshold * COSINE_TO_DOT_SCALE))
    return {
        "probeShifted": [str(v) for v in probe_shifted],
        "candShifted": [str(v) for v in cand_shifted],
        "threshold": str(threshold_scaled),
    }


def compute_witness(circuit_input: dict, out_dir: Path | None = None) -> Path:
    """Runs the compiled circuit's witness calculator (WASM, via Node) over
    `circuit_input`. Returns the path to the .wtns file.
    """
    _require(WASM_PATH)
    _require(GENERATE_WITNESS_JS)
    node = find_node()

    work_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="faceanchor_zk_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.json"
    witness_path = work_dir / "witness.wtns"
    input_path.write_text(json.dumps(circuit_input))

    subprocess.run(
        [node, str(GENERATE_WITNESS_JS), str(WASM_PATH), str(input_path), str(witness_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return witness_path
