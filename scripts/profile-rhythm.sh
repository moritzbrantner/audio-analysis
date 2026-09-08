#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
binary="$root/target/release/examples/profile_rhythm"
needs_build=0

if [[ ! -x "$binary" ]]; then
  needs_build=1
elif [[ "$root/Cargo.toml" -nt "$binary" || "$root/Cargo.lock" -nt "$binary" ]]; then
  needs_build=1
elif find \
  "$root/crates/audio/audio-analysis-core" \
  "$root/crates/audio/audio-analysis-fourier" \
  "$root/crates/audio/audio-analysis-rhythm" \
  -type f \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "$binary" -print -quit | grep -q .; then
  needs_build=1
fi

if [[ "$needs_build" -eq 1 ]]; then
  cargo build --quiet --locked --release \
    -p moenarch-audio-analysis-rhythm \
    --example profile_rhythm
fi

exec "$binary"
