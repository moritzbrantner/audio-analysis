#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

tooling_dir="${CODING_TOOLING_DIR:-$root/../coding-tooling}"
mode="${1:-quick}"
minimum_free_gib="${AGENT_MIN_FREE_GIB:-8}"
target_dir="${CARGO_TARGET_DIR:-$root/target}"
target_parent="$(dirname "$target_dir")"
mkdir -p "$target_parent"

case "$mode" in
  quick) profile="default" ;;
  --with-source) profile="source-development" ;;
  *)
    printf '%s\n' "usage: scripts/check-agent-readiness.sh [--with-source]" >&2
    exit 2
    ;;
esac

run_tooling() {
  if command -v coding-tooling >/dev/null 2>&1; then
    coding-tooling "$@"
    return
  fi
  if [[ -f "$tooling_dir/src/cli.ts" ]]; then
    if ! command -v bun >/dev/null 2>&1; then
      printf '%s\n' "bun is required to run the sibling coding-tooling checkout" >&2
      exit 2
    fi
    bun "$tooling_dir/src/cli.ts" "$@"
    return
  fi
  printf '%s\n' "coding-tooling is required. Install it or set CODING_TOOLING_DIR to its checkout." >&2
  exit 2
}

if ! [[ "$minimum_free_gib" =~ ^[0-9]+$ ]]; then
  printf 'AGENT_MIN_FREE_GIB must be a non-negative integer, got %s\n' "$minimum_free_gib" >&2
  exit 2
fi

free_kib="$(df -Pk "$target_parent" | awk 'NR == 2 { print $4 }')"
required_kib="$((minimum_free_gib * 1024 * 1024))"
if [[ -z "$free_kib" || "$free_kib" -lt "$required_kib" ]]; then
  printf 'insufficient free disk for Cargo target: require %s GiB at %s\n' "$minimum_free_gib" "$target_parent" >&2
  df -Ph "$target_parent" >&2 || true
  exit 1
fi

activated_here=false
cleanup() {
  if [[ "$activated_here" == "true" ]]; then
    run_tooling source-deps deactivate --config "$root/.coding-tooling.source-deps.json" --json >/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$mode" == "--with-source" ]]; then
  source_status="$(run_tooling source-deps status --config "$root/.coding-tooling.source-deps.json" --json)"
  was_active="$(printf '%s' "$source_status" | python3 -c 'import json,sys; print("true" if json.load(sys.stdin).get("data", {}).get("active") else "false")')"
  run_tooling source-deps activate --config "$root/.coding-tooling.source-deps.json" --json >/dev/null
  if [[ "$was_active" != "true" ]]; then
    activated_here=true
  fi
fi

receipt="$(mktemp)"
trap 'rm -f "$receipt"; cleanup' EXIT
if ! run_tooling environment verify --profile "$profile" --json > "$receipt"; then
  cat "$receipt" >&2
  exit 1
fi

fingerprint="$(python3 - "$receipt" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    receipt = json.load(handle)
if receipt.get('status') != 'passed':
    raise SystemExit(f"environment verification did not pass: {receipt.get('status')}")
data = receipt.get('data', {})
expected = data.get('expectedFingerprint')
verified = data.get('verifiedFingerprint')
if not expected or verified != expected:
    raise SystemExit('environment fingerprint was not verified')
print(verified)
PY
)"

cargo metadata --locked --format-version 1 --no-deps >/dev/null

if [[ "$activated_here" == "true" ]]; then
  run_tooling source-deps deactivate --config "$root/.coding-tooling.source-deps.json" --json >/dev/null
  activated_here=false
fi
rm -f "$receipt"
trap - EXIT

free_gib="$((free_kib / 1024 / 1024))"
printf 'agent readiness: passed (profile=%s, fingerprint=%s, free-disk=%sGiB)\n' "$profile" "${fingerprint:0:12}" "$free_gib"
