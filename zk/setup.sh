#!/usr/bin/env bash
# Compiles the match circuit and runs a one-time Groth16 trusted setup
# (PRD §7.1): fetches an existing Powers-of-Tau ceremony at 2^16 rather than
# running a custom ceremony, then contributes local entropy for the
# circuit-specific phase 2 zkey. Everything under build/ is gitignored and
# regenerable via this script.
set -euo pipefail

cd "$(dirname "$0")"

command -v circom >/dev/null 2>&1 || { echo "circom not found — see README for install (prebuilt binary from github.com/iden3/circom/releases)"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "node not found — install via nvm (https://github.com/nvm-sh/nvm) or your package manager"; exit 1; }

if [ ! -d node_modules ]; then
    echo "installing snarkjs + circomlib..."
    npm install
fi

mkdir -p build

echo "compiling circuit..."
circom circuits/match.circom --r1cs --wasm --sym -o build

PTAU=build/powersOfTau28_hez_final_16.ptau
if [ ! -f "$PTAU" ]; then
    echo "fetching Powers-of-Tau (2^16, Hermez ceremony)..."
    curl -fsSL "https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_16.ptau" -o "$PTAU"
fi

echo "groth16 setup..."
npx snarkjs groth16 setup build/match.r1cs "$PTAU" build/match_0000.zkey

echo "contributing entropy (phase 2)..."
echo "faceanchor-setup-$(date +%s)-$RANDOM" | npx snarkjs zkey contribute \
    build/match_0000.zkey build/match_final.zkey --name="faceanchor setup"

echo "exporting verification key..."
npx snarkjs zkey export verificationkey build/match_final.zkey build/verification_key.json

echo "zk setup complete: build/match_final.zkey, build/verification_key.json"
