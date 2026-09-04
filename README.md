# audio-analysis

Rust-first audio analysis, recognition, transcription, synthesis, and generation packages extracted from `moritzbrantner/rust-packages`.

For the Rust packages assigned to `audio-analysis`, this repository is the canonical source, test, issue, version, and release authority. Historical copies in `rust-packages` are compatibility/provenance material rather than a competing implementation or release source. Ownership does not itself publish, tag, or remove historical source; those remain explicit destination-local release or migration operations.

## Audio Inspector

The repository includes a GitHub Pages Audio Inspector at <https://moritzbrantner.github.io/audio-analysis/>. Drop an audio file to inspect file metadata, waveform, levels and dynamics, clipping and near-silence indicators, spectral features, pitch and musical key estimates, and rhythm information.

The public Pages deployment is browser-local: the browser decodes the selected file and the existing Rust package surfaces run through WASM. The report distinguishes whole-file statistics from bounded representative-window analyses so package surface limits are visible instead of hidden. Reports can be exported as JSON and include package/version provenance.

Local deployments retain a separate backend seam for heavier model-backed capabilities. Browser-local analysis remains the default; a backend is discovered on localhost at `http://127.0.0.1:3000` or can be selected with `?backend=<base-url>`. The public Pages deployment never enables backend upload automatically.

Build the exact Pages artifact locally with:

```text
bash scripts/build-pages.sh
```

The script builds the core, Fourier, pitch, and rhythm WASM adapters and assembles the static artifact under `_site/`.

## Development surface

The repository still retains the reviewed historical package inventory for compatibility, but ordinary development is intentionally smaller. The capability library crates are the Cargo workspace `default-members`; per-capability CLI, server, WASM, and app packages are compatibility shells and are not the default feature-development surface.

Use the library loop for normal work:

```text
scripts/check-fast.sh
```

That validates Cargo metadata and the capability libraries. When an adapter shell changes, or before checking distribution compatibility, run:

```text
bash scripts/check-adapters.sh
```

Repository CPU CI deliberately remains broader than the local fast loop: preflight runs the default library checks, the complete workspace with default features, and the important non-CUDA optional feature combinations, followed by documentation and package checks. Reducing local iteration cost must not reduce compatibility coverage.

CUDA is a resource-backed surface rather than a requirement of the ordinary hosted CPU runner. On a CUDA-equipped machine with `nvcc` available, run:

```text
bash scripts/check-cuda.sh
```

That check covers the transcription and TTS CUDA paths plus their transport adapters. CPU CI does not claim CUDA evidence.

## Package-shape direction

Do not multiply transports by creating another CLI/server/WASM/app package for every library. New behavior belongs in the capability libraries first. Existing adapter shells remain until consumer evidence shows that they can be removed or replaced without losing a real deployment boundary.

Repeated co-change between independently versioned library crates is consolidation evidence. Consolidation should be driven by that evidence and a clear ownership/API boundary rather than by adding another facade layer.

## Cross-repository development

Ordinary consumer work is source-first. Native WhisperX and other applications may validate exact `audio-analysis` source revisions without publishing intermediate crates. Registry publication, version bumps, tags, and registry-only consumer verification belong to a separate release/distribution task.

`moenarch-audio-contracts` and the other foundation/NLP contracts remain external dependencies. Committed package manifests must not introduce sibling paths, moving Git references, or visual-analysis dependencies.
