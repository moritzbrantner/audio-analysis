#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git rev-parse --show-toplevel)
scratch_parent=${AUDIO_CONTRACT_SCRATCH_PARENT:-${TMPDIR:-/tmp}}
mkdir -p "$scratch_parent"
scratch_root=$(mktemp -d "$scratch_parent/native-whisperx-audio-contract.XXXXXX")

cleanup() {
  case "$scratch_root" in
    "$scratch_parent"/native-whisperx-audio-contract.*)
      rm -rf -- "$scratch_root"
      ;;
    *)
      echo "refusing to remove unexpected candidate directory: $scratch_root" >&2
      ;;
  esac
}
trap cleanup EXIT

git init --quiet "$scratch_root/native-whisperx"
git -C "$scratch_root/native-whisperx" remote add origin https://github.com/moritzbrantner/native-whisperx.git
git -C "$scratch_root/native-whisperx" fetch --quiet --depth=1 origin 0f42fa16f95a9675bb7562112f388706481f839e
git -C "$scratch_root/native-whisperx" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$scratch_root/native-whisperx" rev-parse HEAD)" = 0f42fa16f95a9675bb7562112f388706481f839e

patch_config="$scratch_root/audio-contract-patches.toml"
cat >"$patch_config" <<PATCH
[patch.crates-io]
"moenarch-audio-analysis-core" = { path = "$repository_root/crates/audio/audio-analysis-core" }
"moenarch-audio-analysis-fourier" = { path = "$repository_root/crates/audio/audio-analysis-fourier" }
"moenarch-audio-analysis-recognition" = { path = "$repository_root/crates/audio/audio-analysis-recognition" }
"moenarch-audio-analysis-io" = { path = "$repository_root/crates/audio/audio-analysis-io" }
"moenarch-audio-analysis-speakers" = { path = "$repository_root/crates/audio/audio-analysis-speakers" }
"moenarch-audio-analysis-transcription" = { path = "$repository_root/crates/audio/audio-analysis-transcription" }
PATCH

manifest="$scratch_root/native-whisperx/Cargo.toml"
for package_version in   moenarch-audio-analysis-core@0.1.1   moenarch-audio-analysis-fourier@0.1.1   moenarch-audio-analysis-recognition@0.1.1   moenarch-audio-analysis-io@0.1.2   moenarch-audio-analysis-speakers@0.1.5   moenarch-audio-analysis-transcription@0.1.16
do
  package=${package_version%@*}
  version=${package_version#*@}
  cargo update --manifest-path "$manifest" --config "$patch_config" -p "$package" --precise "$version"
done

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$repository_root/target"}
cargo check --manifest-path "$manifest" --workspace --locked --config "$patch_config"
if cargo tree --manifest-path "$manifest" -p native-whisperx --locked --config "$patch_config"   | rg 'moenarch-video-analysis-(core|ffmpeg)'
then
  echo "native-whisperx candidate still selects legacy visual contracts" >&2
  exit 1
fi
echo "native-whisperx PR #230 candidate passed against the six-package closure"
