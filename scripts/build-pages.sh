#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/_site}"

if ! command -v wasm-pack >/dev/null 2>&1; then
  printf '%s\n' "wasm-pack is required to build the Audio Inspector Pages artifact." >&2
  exit 2
fi

if ! command -v node >/dev/null 2>&1; then
  printf '%s\n' "node is required to validate the Audio Inspector JavaScript." >&2
  exit 2
fi

bash "$ROOT_DIR/packages/audio-analysis-core-wasm/scripts/build-wasm.sh"
bash "$ROOT_DIR/packages/audio-analysis-fourier-wasm/scripts/build-wasm.sh"
bash "$ROOT_DIR/packages/audio-analysis-pitch-wasm/scripts/build-wasm.sh"
bash "$ROOT_DIR/packages/audio-analysis-rhythm-wasm/scripts/build-wasm.sh"

node --check "$ROOT_DIR/site/app.js"
python3 -m json.tool "$ROOT_DIR/site/analysis-capabilities.json" >/dev/null

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/wasm"
cp -R "$ROOT_DIR/site/." "$OUTPUT_DIR/"

write_pages_wasm_adapter() {
  local package="$1"
  local source_root="$ROOT_DIR/packages/${package}-wasm"
  local target_root="$OUTPUT_DIR/wasm/$package"
  local candidate
  local wasm_entries=()

  mkdir -p "$target_root"
  cp -R "$source_root/pkg" "$target_root/pkg"

  for candidate in "$source_root"/pkg/*_wasm.js; do
    [[ -f "$candidate" ]] || continue
    wasm_entries+=("$(basename "$candidate")")
  done

  if [[ ${#wasm_entries[@]} -ne 1 ]]; then
    printf 'expected exactly one generated *_wasm.js entry for %s, found %s\n' \
      "$package" "${#wasm_entries[@]}" >&2
    exit 1
  fi

  cat > "$target_root/index.js" <<EOF
let wasmModulePromise;

export async function init() {
  const wasmEntry = "./pkg/${wasm_entries[0]}";
  wasmModulePromise ??= import(wasmEntry).then(async (module) => {
    if (typeof module.default === "function") {
      await module.default();
    }
    return module;
  });
  return wasmModulePromise;
}

export async function packageSurface() {
  const module = await init();
  return module.packageSurface();
}

export async function runOperation(request) {
  const module = await init();
  return module.runOperation(request);
}
EOF
}

for package in audio-analysis-core audio-analysis-fourier audio-analysis-pitch audio-analysis-rhythm; do
  write_pages_wasm_adapter "$package"
done

touch "$OUTPUT_DIR/.nojekyll"
printf 'Audio Inspector Pages artifact: %s\n' "$OUTPUT_DIR"
