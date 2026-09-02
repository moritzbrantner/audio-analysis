#!/usr/bin/env bash
set -euo pipefail

mode="${1:-setup}"
if [[ "$mode" != "setup" && "$mode" != "maintenance" ]]; then
  printf 'usage: %s [setup|maintenance]\n' "$0" >&2
  exit 2
fi

root="$(git rev-parse --show-toplevel)"
config="$root/.repository-environment.toml"

if [[ ! -f "$config" ]]; then
  printf 'missing environment-v1 config: %s\n' "$config" >&2
  exit 2
fi

run_privileged() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

publish_path() {
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    printf '%s\n' "$1" >> "$GITHUB_PATH"
  fi
}

if [[ "$mode" == "setup" ]] && command -v apt-get >/dev/null 2>&1; then
  mapfile -t apt_packages < <(python3 - "$config" <<'PY'
import sys, tomllib
with open(sys.argv[1], 'rb') as handle:
    data = tomllib.load(handle)
for package in data.get('system', {}).get('apt', []):
    print(package)
PY
  )
  if (( ${#apt_packages[@]} )); then
    run_privileged apt-get update
    run_privileged apt-get install -y --no-install-recommends "${apt_packages[@]}"
  fi
fi

rust_toolchain="$(python3 - "$root/rust-toolchain.toml" <<'PY'
import pathlib, sys, tomllib
path = pathlib.Path(sys.argv[1])
if path.is_file():
    print(tomllib.loads(path.read_text()).get('toolchain', {}).get('channel', ''))
PY
)"

if [[ -n "$rust_toolchain" ]]; then
  if ! [[ "$rust_toolchain" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf 'Rust toolchain must use an exact version, got %s\n' "$rust_toolchain" >&2
    exit 2
  fi
  if ! command -v rustup >/dev/null 2>&1; then
    printf '%s\n' 'rustup is required before repository setup; provision it through the trusted agent/base environment rather than an unpinned installer.' >&2
    exit 2
  fi
  if [[ -d "$HOME/.cargo/bin" ]]; then
    publish_path "$HOME/.cargo/bin"
  fi
  rustup toolchain install "$rust_toolchain" --profile minimal
  mapfile -t rust_components < <(python3 - "$root/rust-toolchain.toml" <<'PY'
import pathlib, sys, tomllib
path = pathlib.Path(sys.argv[1])
if path.is_file():
    for component in tomllib.loads(path.read_text()).get('toolchain', {}).get('components', []):
        print(component)
PY
  )
  for component in "${rust_components[@]}"; do
    rustup component add --toolchain "$rust_toolchain" "$component"
  done
fi

bun_version="$(python3 - "$root/package.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if path.is_file():
    value = json.loads(path.read_text()).get('packageManager', '')
    print(value.removeprefix('bun@') if value.startswith('bun@') else '')
PY
)"
if [[ -n "$bun_version" ]]; then
  if ! [[ "$bun_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$ ]]; then
    printf 'Bun packageManager must use an exact version, got %s\n' "$bun_version" >&2
    exit 2
  fi
  if ! command -v bun >/dev/null 2>&1; then
    printf '%s\n' 'bun is required before repository setup; provision the exact packageManager version through the trusted agent/base environment.' >&2
    exit 2
  fi
  observed_bun="$(bun --version)"
  if [[ "$observed_bun" != "$bun_version" ]]; then
    printf 'Bun preflight mismatch: expected %s, got %s\n' "$bun_version" "$observed_bun" >&2
    exit 1
  fi
fi

mapfile -t environment_commands < <(python3 - "$config" "$mode" <<'PY'
import sys, tomllib
with open(sys.argv[1], 'rb') as handle:
    data = tomllib.load(handle)
for command in data.get(sys.argv[2], {}).get('commands', []):
    print(command)
PY
)
for command in "${environment_commands[@]}"; do
  (cd "$root" && bash -lc "$command")
done

if [[ -n "$rust_toolchain" ]]; then
  observed_rust="$(cd "$root" && rustc --version | awk '{print $2}')"
  if [[ "$observed_rust" != "$rust_toolchain" ]]; then
    printf 'Rust preflight mismatch: expected %s, got %s\n' "$rust_toolchain" "$observed_rust" >&2
    exit 1
  fi
fi
