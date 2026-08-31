// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Script, console} from "forge-std/Script.sol";
import {EvidenceAnchor} from "../src/EvidenceAnchor.sol";
import {Groth16Verifier} from "../src/Groth16Verifier.sol";

/// @dev Usage:
///   forge script script/Deploy.s.sol --rpc-url anvil --broadcast
///   forge script script/Deploy.s.sol --rpc-url amoy --broadcast --verify
///
/// Always deploys Groth16Verifier too (cheap — a few hundred k gas one-time)
/// so anchorWithProof works even if --zk isn't used in this particular demo.
contract Deploy is Script {
    function run() external returns (EvidenceAnchor, Groth16Verifier) {
        vm.startBroadcast();
        Groth16Verifier verifier = new Groth16Verifier();
        EvidenceAnchor anchor = new EvidenceAnchor(address(verifier));
        vm.stopBroadcast();

        console.log("Groth16Verifier deployed at:", address(verifier));
        console.log("EvidenceAnchor deployed at:", address(anchor));
        return (anchor, verifier);
    }
}
