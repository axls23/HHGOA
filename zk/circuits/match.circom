pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";

/// Commits to an n-element vector via a 2-level Poseidon tree: n/groupSize
/// groups hashed with Poseidon(groupSize), then those sub-hashes hashed
/// together with Poseidon(n/groupSize).
///
/// PRD §7.1 describes this as "2 x Poseidon sponge over 128 field elements",
/// but circomlib's Poseidon template only ships round constants up to 16
/// inputs (t=17) — there are no precomputed constants for a genuine
/// single-call sponge over 128 elements. A 2-level tree over 16-groups is
/// the practical substitute: same binding-commitment security property
/// (collision resistance inherited from Poseidon), built entirely from
/// arities circomlib actually supports, at a comparable constraint count.
template VectorCommitment(n, groupSize) {
    signal input in[n];
    signal output root;

    var nGroups = n / groupSize;
    component groupHash[nGroups];
    for (var g = 0; g < nGroups; g++) {
        groupHash[g] = Poseidon(groupSize);
        for (var i = 0; i < groupSize; i++) {
            groupHash[g].inputs[i] <== in[g * groupSize + i];
        }
    }

    component rootHash = Poseidon(nGroups);
    for (var g = 0; g < nGroups; g++) {
        rootHash.inputs[g] <== groupHash[g].out;
    }
    root <== rootHash.out;
}

/// circom signals are unsigned field elements, so a signed int16 in
/// [-bound, bound] is passed in as `shifted = value + bound`, range-checked
/// to [0, 2*bound], and un-shifted back to the true signed value in-circuit.
/// (Field subtraction of two values this small is exact — no wraparound —
/// since bound << the bn254 scalar field size.)
template SignedRangeCheck(bound, bits) {
    signal input shifted;
    signal output signedVal;

    component lte = LessEqThan(bits);
    lte.in[0] <== shifted;
    lte.in[1] <== 2 * bound;
    lte.out === 1;

    signedVal <== shifted - bound;
}

/// Public:  probeCommitment, candCommitment, threshold
/// Private: probeShifted[n], candShifted[n]  (int16 fixed-point, scale 2^12,
///          shifted to unsigned as described above)
///
/// Proves: the prover knows two vectors, each individually committed to and
/// range-bounded, whose dot product clears `threshold`. It does NOT prove
/// those vectors were correctly derived by SFace from any particular image
/// (see PRD §7.1's "be honest about the boundary" — that would need zkML
/// over the full CNN, ~10^8 constraints, out of scope).
template MatchProof(n, groupSize, bound, bits) {
    signal input probeShifted[n];
    signal input candShifted[n];
    signal input threshold;

    signal output probeCommitment;
    signal output candCommitment;

    component probeCommit = VectorCommitment(n, groupSize);
    component candCommit = VectorCommitment(n, groupSize);

    signal probeSigned[n];
    signal candSigned[n];
    component probeRange[n];
    component candRange[n];

    for (var i = 0; i < n; i++) {
        probeCommit.in[i] <== probeShifted[i];
        candCommit.in[i] <== candShifted[i];

        probeRange[i] = SignedRangeCheck(bound, bits);
        probeRange[i].shifted <== probeShifted[i];
        probeSigned[i] <== probeRange[i].signedVal;

        candRange[i] = SignedRangeCheck(bound, bits);
        candRange[i].shifted <== candShifted[i];
        candSigned[i] <== candRange[i].signedVal;
    }

    probeCommitment <== probeCommit.root;
    candCommitment <== candCommit.root;

    signal partialSum[n + 1];
    partialSum[0] <== 0;
    for (var i = 0; i < n; i++) {
        partialSum[i + 1] <== partialSum[i] + probeSigned[i] * candSigned[i];
    }

    component geq = GreaterEqThan(bits * 2 + 8);
    geq.in[0] <== partialSum[n];
    geq.in[1] <== threshold;
    geq.out === 1;
}

// n=128 (embedding dim), groupSize=16 (circomlib's Poseidon arity ceiling),
// bound=4096 (int16 fixed-point scale 2^12), bits=13 (covers [0, 8192]).
component main {public [threshold]} = MatchProof(128, 16, 4096, 13);
