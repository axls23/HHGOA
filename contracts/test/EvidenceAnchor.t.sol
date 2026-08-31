// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {EvidenceAnchor} from "../src/EvidenceAnchor.sol";

contract EvidenceAnchorTest is Test {
    EvidenceAnchor anchor;

    function setUp() public {
        anchor = new EvidenceAnchor();
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
}
