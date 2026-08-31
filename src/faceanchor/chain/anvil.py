from __future__ import annotations

from faceanchor.chain.evm import EvmAdapter

# Foundry's well-known dev key #0 — funded automatically by `anvil` in dev mode.
# Anvil-only; never used against a real network.
ANVIL_DEFAULT_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
ANVIL_DEFAULT_RPC_URL = "http://127.0.0.1:8545"


class AnvilAdapter(EvmAdapter):
    """Local Foundry Anvil chain — instant finality, zero cost, no network dependency.

    Exists so `make demo` and CI both run fully offline (PRD §6.1).
    """

    def __init__(self, contract_address: str, rpc_url: str = ANVIL_DEFAULT_RPC_URL, private_key: str = ANVIL_DEFAULT_PRIVATE_KEY):
        super().__init__(rpc_url=rpc_url, private_key=private_key, contract_address=contract_address, poa=False)
