// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {EvidenceAnchor} from "../src/EvidenceAnchor.sol";

contract EvidenceAnchorTest is Test {
    EvidenceAnchor anchor;

    function setUp() public {
        anchor = new EvidenceAnchor(address(0)); // ZK path disabled; not exercised by these tests
    }

    function test_AnchorAndVerify() public {
        bytes32 digest = keccak256("bundle-1");
        anchor.anchor(digest, "bafyreitest");

        (bool ok, uint64 ts) = anchor.verify(digest);
        assertTrue(ok);
        assertEq(ts, uint64(block.timestamp));
    }

    function test_VerifyUnknownDigestReturnsFalse() public view {
        (bool ok, uint64 ts) = anchor.verify(keccak256("never-anchored"));
        assertFalse(ok);
        assertEq(ts, 0);
    }

    function test_RevertOnDoubleAnchor() public {
        bytes32 digest = keccak256("bundle-2");
        anchor.anchor(digest, "cid-a");

        vm.expectRevert(abi.encodeWithSelector(EvidenceAnchor.AlreadyAnchored.selector, digest));
        anchor.anchor(digest, "cid-b");
    }

    function testFuzz_AnchorIsIdempotentRejecting(bytes32 digest, string calldata cid) public {
        anchor.anchor(digest, cid);
        vm.expectRevert(abi.encodeWithSelector(EvidenceAnchor.AlreadyAnchored.selector, digest));
        anchor.anchor(digest, cid);
    }

    function test_EmitsAnchoredEvent() public {
        bytes32 digest = keccak256("bundle-3");
        vm.expectEmit(true, true, false, true);
        emit EvidenceAnchor.Anchored(digest, address(this), uint64(block.timestamp), "cid-3");
        anchor.anchor(digest, "cid-3");
    }

    function test_AnchorBatch() public {
        bytes32 root = keccak256("merkle-root-1");
        anchor.anchorBatch(root, 1024);

        (bool ok, uint64 ts) = anchor.verify(root);
        assertTrue(ok);
        assertEq(ts, uint64(block.timestamp));
    }

    function test_RevertOnDoubleAnchorBatch() public {
        bytes32 root = keccak256("merkle-root-2");
        anchor.anchorBatch(root, 10);

        vm.expectRevert(abi.encodeWithSelector(EvidenceAnchor.AlreadyAnchored.selector, root));
        anchor.anchorBatch(root, 10);
    }

    function test_GasSnapshot_Anchor() public {
        bytes32 digest = keccak256("gas-check");
        uint256 gasBefore = gasleft();
        anchor.anchor(digest, "bafyreigas");
        uint256 used = gasBefore - gasleft();
        // PRD target ~46k (cold SSTORE 20k + log ~2k + base 21k); allow headroom for calldata cost.
        assertLt(used, 70_000);
    }

    function test_RevokeAndConsentStatus() public {
        bytes32 consentDigest = keccak256("consent-1");

        (bool revokedBefore, uint64 tsBefore) = anchor.consentStatus(consentDigest);
        assertFalse(revokedBefore);
        assertEq(tsBefore, 0);

        anchor.revoke(consentDigest);

        (bool revokedAfter, uint64 tsAfter) = anchor.consentStatus(consentDigest);
        assertTrue(revokedAfter);
        assertEq(tsAfter, uint64(block.timestamp));
    }

    function test_RevertOnDoubleRevoke() public {
        bytes32 consentDigest = keccak256("consent-2");
        anchor.revoke(consentDigest);

        vm.expectRevert(abi.encodeWithSelector(EvidenceAnchor.AlreadyRevoked.selector, consentDigest));
        anchor.revoke(consentDigest);
    }

    function test_EmitsRevokedEvent() public {
        bytes32 consentDigest = keccak256("consent-3");
        vm.expectEmit(true, true, false, true);
        emit EvidenceAnchor.Revoked(consentDigest, address(this), uint64(block.timestamp));
        anchor.revoke(consentDigest);
    }

    function test_RevokeIsIndependentFromAnchoredAt() public {
        // anchoring a digest and revoking a (different-namespace) consent
        // digest that happens to share the same bytes32 value must not
        // cross-contaminate — they're separate mappings.
        bytes32 sharedValue = keccak256("shared");
        anchor.anchor(sharedValue, "cid-shared");
        anchor.revoke(sharedValue);

        (bool anchoredOk,) = anchor.verify(sharedValue);
        (bool revoked,) = anchor.consentStatus(sharedValue);
        assertTrue(anchoredOk);
        assertTrue(revoked);
    }
}
