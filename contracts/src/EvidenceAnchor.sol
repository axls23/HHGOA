// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @notice Tamper-evident anchor for off-chain evidence bundles.
/// @dev Stores no personal data — only keccak256 digests of canonicalised JSON.
contract EvidenceAnchor {
    event Anchored(
        bytes32 indexed digest,
        address indexed attester,
        uint64  timestamp,
        string  cid          // IPFS CIDv1 of the full bundle
    );
    event BatchAnchored(bytes32 indexed merkleRoot, address indexed attester, uint64 timestamp, uint32 leaves);

    error AlreadyAnchored(bytes32 digest);

    mapping(bytes32 => uint64) public anchoredAt;

    function anchor(bytes32 digest, string calldata cid) external {
        if (anchoredAt[digest] != 0) revert AlreadyAnchored(digest);
        anchoredAt[digest] = uint64(block.timestamp);
        emit Anchored(digest, msg.sender, uint64(block.timestamp), cid);
    }

    /// @notice One tx anchors N discoveries; verify a leaf with a Merkle proof.
    function anchorBatch(bytes32 merkleRoot, uint32 leaves) external {
        if (anchoredAt[merkleRoot] != 0) revert AlreadyAnchored(merkleRoot);
        anchoredAt[merkleRoot] = uint64(block.timestamp);
        emit BatchAnchored(merkleRoot, msg.sender, uint64(block.timestamp), leaves);
    }

    function verify(bytes32 digest) external view returns (bool ok, uint64 ts) {
        ts = anchoredAt[digest];
        ok = ts != 0;
    }
}
