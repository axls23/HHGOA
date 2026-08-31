// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {EvidenceAnchor} from "../src/EvidenceAnchor.sol";
import {Groth16Verifier} from "../src/Groth16Verifier.sol";

/// @dev The proof/public-signal values below are a real Groth16 proof,
/// generated end-to-end via zk/circuits/match.circom + snarkjs against a
/// real (identical-vector, self-similarity=1.0) witness — see M9's commit
/// for how; dumped via `snarkjs generatecall`. Not a placeholder: this test
/// fails if the circuit, the trusted setup, or the verifier export ever
/// stop agreeing with each other.
contract AnchorWithProofTest is Test {
    EvidenceAnchor anchor;
    Groth16Verifier verifier;

    uint[2] validA = [
        0x242540ade44142b4cc09d96067c6f22dbd0bb1791e19366532e96738bbfa7e31,
        0x1b060f7f8333ef68d2c5ad3b3e2bb9d1c3ad3ca9221eb52b40dc1f41ed22dbe6
    ];
    uint[2][2] validB = [
        [
            0x05d093253d778fbd6c23ef21f065b74dc40a1f300610d467096e4ee39f7a7a76,
            0x212508568e8dbf3728b3a46bd307ea59acba4d2b3f83d5fcfb719d367e0eab95
        ],
        [
            0x2f241d4ae994ecfa6d85089d3f52f92958432c78d081bd39fc8636afd839a603,
            0x06645b152e669960b3ed32fc2a72e277397af8e1bfbfaed55938b067b3f66304
        ]
    ];
    uint[2] validC = [
        0x15eba2499023421f00395e7394e856eeb04193cefa4abc77a599993a95a2aba1,
        0x08e6fee06333777d91117dd64c328bf5afac82f30c8571fb257034f3c1cb3ef9
    ];
    uint[3] validPubSignals = [
        0x1faaa7496f4828b29f6f9f9efe2b85896f59f326519c20622d6dfc62a87b3dff,
        0x1faaa7496f4828b29f6f9f9efe2b85896f59f326519c20622d6dfc62a87b3dff,
        0x00000000000000000000000000000000000000000000000000000000005ced91
    ];

    function setUp() public {
        verifier = new Groth16Verifier();
        anchor = new EvidenceAnchor(address(verifier));
    }

    function test_VerifierAcceptsRealProof() public view {
        bool ok = verifier.verifyProof(validA, validB, validC, validPubSignals);
        assertTrue(ok);
    }

    function test_AnchorWithProof_ValidProofAnchors() public {
        bytes32 digest = keccak256("zk-bundle-1");
        anchor.anchorWithProof(digest, "bafyreizk1", validA, validB, validC, validPubSignals);

        (bool ok, uint64 ts) = anchor.verify(digest);
        assertTrue(ok);
        assertEq(ts, uint64(block.timestamp));
    }

    function test_AnchorWithProof_ForgedProofReverts() public {
        // corrupt one field of a real proof — a syntactically valid but
        // cryptographically forged proof, not garbage input.
        uint[2] memory forgedA = validA;
        forgedA[0] = forgedA[0] + 1;

        bytes32 digest = keccak256("zk-bundle-forged");
        vm.expectRevert(EvidenceAnchor.BadProof.selector);
        anchor.anchorWithProof(digest, "bafyreiforged", forgedA, validB, validC, validPubSignals);
    }

    function test_AnchorWithProof_ForgedScoreReverts() public {
        // same proof, but claiming a higher threshold than what was actually
        // proven — pubSignals are part of what verifyProof checks, so this
        // is exactly PRD §7.1's "chain rejects an anchor with a forged score".
        uint[3] memory forgedPubSignals = validPubSignals;
        forgedPubSignals[2] = 0x1fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff;

        bytes32 digest = keccak256("zk-bundle-forged-score");
        vm.expectRevert(EvidenceAnchor.BadProof.selector);
        anchor.anchorWithProof(digest, "bafyreiforgedscore", validA, validB, validC, forgedPubSignals);
    }

    function test_AnchorWithProof_RejectsWhenVerifierUnset() public {
        EvidenceAnchor noVerifierAnchor = new EvidenceAnchor(address(0));
        vm.expectRevert();
        noVerifierAnchor.anchorWithProof(keccak256("x"), "cid", validA, validB, validC, validPubSignals);
    }

    function test_AnchorWithProof_RejectsDoubleAnchor() public {
        bytes32 digest = keccak256("zk-bundle-double");
        anchor.anchorWithProof(digest, "cid-a", validA, validB, validC, validPubSignals);

        vm.expectRevert(abi.encodeWithSelector(EvidenceAnchor.AlreadyAnchored.selector, digest));
        anchor.anchorWithProof(digest, "cid-b", validA, validB, validC, validPubSignals);
    }
}
