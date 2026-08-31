// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Script, console} from "forge-std/Script.sol";
import {EvidenceAnchor} from "../src/EvidenceAnchor.sol";

/// @dev Usage:
///   forge script script/Deploy.s.sol --rpc-url anvil --broadcast
///   forge script script/Deploy.s.sol --rpc-url amoy --broadcast --verify
contract Deploy is Script {
    function run() external returns (EvidenceAnchor) {
        vm.startBroadcast();
        EvidenceAnchor anchor = new EvidenceAnchor();
        vm.stopBroadcast();

        console.log("EvidenceAnchor deployed at:", address(anchor));
        return anchor;
    }
}
