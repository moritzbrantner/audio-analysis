#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/_site}"
PLAYWRIGHT_VERSION="1.62.1"
PORT="${PAGES_E2E_PORT:-4173}"
BASE_URL="http://127.0.0.1:$PORT"
TOOL_DIR="$(mktemp -d)"
SERVER_LOG="$TOOL_DIR/http-server.log"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TOOL_DIR"
}
trap cleanup EXIT

cd "$ROOT_DIR"

if [[ ! -f "$OUTPUT_DIR/index.html" ]]; then
  bash scripts/build-pages.sh "$OUTPUT_DIR"
fi

cp "$ROOT_DIR/tests/pages/audio-inspector.e2e.mjs" "$TOOL_DIR/audio-inspector.e2e.mjs"
cat > "$TOOL_DIR/package.json" <<EOF
{
  "private": true,
  "type": "module",
  "dependencies": {
    "playwright": "$PLAYWRIGHT_VERSION"
  }
}
EOF

(
  cd "$TOOL_DIR"
  bun install --no-progress
  if [[ "${CI:-}" == "true" ]]; then
    ./node_modules/.bin/playwright install --with-deps chromium
  else
    ./node_modules/.bin/playwright install chromium
  fi
)

python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$OUTPUT_DIR" >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

ready=0
for _ in $(seq 1 50); do
  if python3 - "$BASE_URL" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=0.25) as response:
        raise SystemExit(0 if response.status < 400 else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    ready=1
    break
  fi
  sleep 0.1
done

if [[ "$ready" != "1" ]]; then
  cat "$SERVER_LOG" >&2 || true
  printf '%s\n' "Audio Inspector E2E server did not become ready at $BASE_URL" >&2
  exit 1
fi

PAGES_E2E_BASE_URL="$BASE_URL" bun "$TOOL_DIR/audio-inspector.e2e.mjs"
