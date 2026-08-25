#!/usr/bin/env bash
set -euo pipefail

# Adapter shells remain compatibility surfaces, but they are not part of the
# ordinary capability-development loop. Run this when changing CLI/server/WASM
# adapters or before a distribution cutover.
cargo test --workspace
