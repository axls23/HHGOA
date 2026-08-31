import shutil

import numpy as np
import pytest

from faceanchor.zk.witness import BUILD_DIR, ZKEY_PATH

pytestmark = pytest.mark.skipif(
    not ZKEY_PATH.exists() or not shutil.which("node"),
    reason="ZK toolchain not set up — run zk/setup.sh (needs node + circom; --zk is an opt-in flag, not required for the base pipeline)",
)


def test_proof_roundtrip_for_identical_vectors():
    from faceanchor.zk.prove import generate_proof, verify_proof
    from faceanchor.zk.witness import build_circuit_input

    rng = np.random.default_rng(1)
    e = rng.normal(size=128).astype(np.float32)
    e /= np.linalg.norm(e)

    circuit_input = build_circuit_input(e, e, cosine_threshold=0.363)
    proof = generate_proof(circuit_input)

    assert len(proof.public_signals) == 3
    assert proof.public_signals[0] == proof.public_signals[1]  # same vector -> same commitment
    assert verify_proof(proof)


def test_proof_generation_fails_below_threshold():
    from faceanchor.zk.prove import generate_proof
    from faceanchor.zk.witness import build_circuit_input

    rng = np.random.default_rng(2)
    e1 = rng.normal(size=128).astype(np.float32)
    e1 /= np.linalg.norm(e1)
    e2 = rng.normal(size=128).astype(np.float32)
    e2 /= np.linalg.norm(e2)
    assert abs(float(np.dot(e1, e2))) < 0.363  # near-orthogonal random vectors in 128-d

    circuit_input = build_circuit_input(e1, e2, cosine_threshold=0.363)
    with pytest.raises(Exception):
        generate_proof(circuit_input)


def test_different_vectors_produce_different_commitments():
    from faceanchor.zk.prove import generate_proof
    from faceanchor.zk.witness import build_circuit_input

    rng = np.random.default_rng(3)
    base = rng.normal(size=128).astype(np.float32)
    base /= np.linalg.norm(base)

    # a candidate close enough to clear the threshold but not identical
    noise = rng.normal(size=128).astype(np.float32) * 0.05
    cand = base + noise
    cand /= np.linalg.norm(cand)
    cosine = float(np.dot(base, cand))
    assert cosine >= 0.363

    circuit_input = build_circuit_input(base, cand, cosine_threshold=0.363)
    proof = generate_proof(circuit_input)
    assert proof.public_signals[0] != proof.public_signals[1]
