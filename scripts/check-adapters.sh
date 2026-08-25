#!/usr/bin/env bash
set -euo pipefail

# Adapter shells remain compatibility surfaces, but they are not part of the
# ordinary capability-development loop. CI and distribution checks still
# exercise the complete workspace and feature surface.
cargo test --workspace --all-features
