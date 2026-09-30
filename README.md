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

## DJ acceptance evidence

The DJ acceptance corpus is declared in `tests/fixtures/dj/real-music-corpus.v1.json`. The repository-level pinned `librosa/data` source remains the default for existing fixtures; external fixtures may override the audio `sourceUrl`, but must also declare a human-auditable, revision-pinned `provenanceUrl` plus the source-published byte length and SHA-1. Every downloaded file is accepted only when its SHA-256 and, for external fixtures, those independent source metadata pins all match the manifest.

The cheap metadata gate does not download audio:

```text
bun run check:dj-corpus
```

The full evaluator downloads the checksum-pinned corpus and compares the Rust whole-track analysis with librosa, and with Essentia when it is installed. It supports focused diagnosis with `--fixture <name>`; the `DJ Real Music Evidence` workflow exposes the same optional fixture input. Run it manually or opt a PR in with the `dj-real-music-evidence` label. Ordinary PR checks use generated fixtures.

```text
python3.12 -m venv .venv-dj
.venv-dj/bin/python -m pip install librosa==1.0.0 essentia==2.1b6.dev1389
.venv-dj/bin/python scripts/evaluate-dj-goldens.py
.venv-dj/bin/python scripts/evaluate-dj-goldens.py --fixture beethoven-pathetique-adagio-modulation
```

The reference workflow pins Python 3.12.13; the pinned Essentia release has no Python 3.14 wheel. Keep the reference environment isolated from the system Python.

Analyzer disagreement remains evidence rather than ground truth. Analyzer-derived key references declare their source and remain non-gating unless a fixture explicitly enables `assertKey` against an independently justified reference. Coverage completeness means the declared scenario families are represented; it does not by itself establish Mixxx-class accuracy.

### Direct Mixxx comparison benchmark

`scripts/evaluate-dj-mixxx.py` compares the 18 checksum-pinned corpus inputs and eight independently authored PCM scenarios with **actual Mixxx 2.5.4**, using its packaged Queen Mary beat/key plugins. Both whole-track profiles are captured: constant tempo (the default assumption) and variable tempo. Fast analysis is disabled. Each profile uses a fresh library/settings directory twice; Rust's public whole-track example also runs twice per input in release mode. A complete run therefore contains 26 inputs, 52 comparisons, 104 Mixxx captures, and 52 Rust captures. The corpus includes synthetic stress and mixed-source material; its size is not a count of independently annotated real songs.

Prepare the declared exact Foundation source checkout, activate source mode, and run:

```sh
bash scripts/source-deps activate
bash scripts/check-agent-readiness.sh --with-source
docker build --tag audio-analysis-mixxx:2.5.4 scripts/mixxx
python3 scripts/evaluate-dj-mixxx.py
```

The host evaluator needs Python's standard library and Docker. The evaluation image pins the Ubuntu base digest and Mixxx package version; it records the installed dependency versions, Mixxx binary checksum, immutable local image ID, and runtime fingerprint rather than assuming that later image builds have identical dependencies. The runtime has no network access, reads inputs through a read-only mount, and writes only its isolated captures. It loads two tracks into normal decks, closes only known startup dialogs, waits for PCM processing, and quits cleanly so Mixxx flushes its own database. The adapter reads the versioned beat grid/map and normalized chromatic key IDs; it does not substitute a reimplementation of Mixxx's analyzer. Database plugin provenance and persisted profile settings must match the requested whole-track analysis.

JSON and Markdown reports are written to `target/dj-goldens/mixxx/report.{json,md}`. Raw repeated captures, settings, databases, and process logs stay in the ignored target directory. Reports record exact input checksums, harness checksums, source/toolchain/lock provenance, missing outputs, repeated-run consistency, exact and half/double-time BPM agreement, beat F1 at 70 ms, and dominant-key agreement. Authored beat/key/tempo correctness and beat-interval tempo errors are separate from model agreement. Constant grids are expanded over `[0, duration)`; variable maps use their stored beat positions inside the same horizon. No inferred bar phase is manufactured from Mixxx's first beat.

Both whole-track pipelines receive the identical encoded file bytes and own their native decoding/downmixing. Differences can therefore include frontend behavior, especially for the five-channel wind fixture; this is not an isolated comparison of algorithms receiving identical decoded PCM. The authored cases deliberately target known failure modes, so their aggregate is a regression workload, not a representative song-accuracy leaderboard.

For a focused probe, use `--fixture <corpus-name>` and optionally `--no-authored`; the report explicitly marks that coverage incomplete. Validate manifest/adapter regressions without downloading audio or launching Mixxx:

```sh
python3 -m unittest discover -s tests/dj -p 'test_*.py'
python3 scripts/evaluate-dj-mixxx.py --check-manifest
```

The **DJ Mixxx Benchmark** workflow can be dispatched manually or enabled for a PR with the `dj-mixxx-benchmark` label. It prepares the exact source graph and uploads the report and repeated reference captures. A successful run proves capture integrity and repeat consistency; it does not mean every output is accurate. Mixxx agreement is not real beat/key ground truth, and this benchmark makes no Mixxx-parity, structure, waveform, downbeat, confidence-calibration, or speed claim. The adapter's wire-format reference is [Mixxx 2.5.4's beat schema](https://github.com/mixxxdj/mixxx/blob/073e5ff876b6d1ed2cb6843f618cafaa6597daed/src/proto/beats.proto); the two tempo assumptions are described in the [Mixxx manual](https://manual.mixxx.org/2.5/en/chapters/preferences/beat_detection).

The committed `mixxx-baseline.v1.json` keeps measured known matches and failures. Full runs gate regressions only against independently authored beats/key/tempo and the creator-described numeric tempo references. Beat F1 may fall by at most the larger of 0.02 absolute or 3% relative before failing; beat-interval tempo error uses 1 BPM absolute or 5% relative. Losing an established tempo/key match also fails. Equivalent inputs, oracles, harness, compiler, build profile, and dependency lock are required. Mixxx runtime changes are recorded separately and its agreement scores never become ground truth. A deliberate `--establish-baseline tests/fixtures/dj/mixxx-baseline.v1.json` update requires a complete, repeat-consistent run and should be reviewed with the resulting metric changes. Focused probes do not apply a whole-corpus accuracy gate. For drifting PCM, the authored scalar BPM convention is the median of instantaneous tempi; different whole-track summary conventions should be assessed alongside the actual beat grid and interval error.

Independent creator tempo references use numeric `referenceBpm` and sourced `referenceBpmSource` fields. The report compares Rust and both reference analyzers with that tempo, allowing explicit half/double-time equivalence. `assertTempo` gates an independent tempo reference when present; otherwise it gates analyzer comparisons. Solarity's creator-declared 124 BPM is asserted after the shared pulse-family fix. The added trumpet loop's creator-described 90 BPM remains a non-gating diagnostic reference; its declared blues/Dorian tonality is not guessed into a major/minor key oracle.

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
