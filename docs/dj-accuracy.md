# DJ analysis accuracy evidence

This repository treats the Mixxx-class goal as an evidence program, not as an implementation-parity claim. The first evidence slice measures the existing Rust-owned whole-song rhythm and key outputs against established analyzers on immutable reusable music fixtures.

## Ownership and privacy boundary

- Reusable BPM, beat, downbeat, key, and structural semantics remain Rust-owned.
- `scripts/evaluate-dj-goldens.py` calls the repository's Rust `dj_analyze` example. It does not implement a parallel Python BPM, key, or structure model.
- librosa and Essentia are references used only by the opt-in evaluation workflow.
- Evaluation fixtures are public corpus files downloaded into `target/dj-goldens`; they are not browser uploads.
- Audio selected or recorded in the GitHub Pages Audio Inspector remains browser-local. This evidence workflow adds no upload endpoint or backend analysis dependency.

## Baseline corpus

The baseline uses six files from the immutable librosa data revision `38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0`.

| Fixture | Coverage role | License | Pin |
| --- | --- | --- | --- |
| Choice (drum+bass) | electronic/dance | CC-BY-NC-4.0 | SHA-256 |
| Brahms Hungarian Dance #5 | classical/live-tempo behavior | CC-PDM-1.0 | SHA-256 |
| Vibe Ace | jazz arrangement / changing instrumentation | CC-BY-4.0 | Git blob SHA-1 at immutable revision |
| Let's Go Fishin' | pop/folk with vocals | CC-BY-NC-SA-4.0 | Git blob SHA-1 at immutable revision |
| Sweet Waltz | triple meter | CC-BY-NC-4.0 | Git blob SHA-1 at immutable revision |
| Pistachio Ice Cream Ragtime | syncopated piano | CC-BY-NC-4.0 | Git blob SHA-1 at immutable revision |

The corpus is deliberately marked `baseline-incomplete`. It does not yet satisfy all issue #69 coverage categories. The machine-readable report lists the missing human-annotated beat, structure, half/double ambiguity, weak-intro, tempo-drift, hip-hop, and modulation cases.

## Metrics

The report schema is `audio-analysis-dj-accuracy/v1`.

### Tempo

For every reference analyzer, the report records the nearest explicit reference multiplier from `0.5`, `1.0`, or `2.0`, the aligned reference BPM, relative error, and whether that error is within 3.5%.

Half-time and double-time are therefore visible in evidence. They are not silently rewritten into an exact-tempo result.

### Beat positions

Rust beats are matched one-to-one against each reference analyzer's beat positions within a fixed 70 ms window. Each fixture reports precision, recall, F1, and mean absolute timing error for matched beats.

The beat metric does not compensate for half/double-time grids. A track that gets a tempo-equivalent BPM but emits the wrong beat density will therefore show that failure in beat precision/recall.

### Dominant key

Where a live analyzer exposes a key label, the report records exact tonic + major/minor agreement. The two pre-existing Essentia references remain pinned to Essentia `2.1b6.dev1389` and are reported separately from live analyzer agreement.

A pinned analyzer result is not described as human ground truth.

### Structure

No structural accuracy number is emitted in this first slice because the baseline corpus does not yet contain trustworthy structural-boundary annotations and the file-analysis example does not expose a separately evaluated boundary contract. The report writes `structureMetric: null` and identifies this as a known coverage gap.

The evaluator must not recreate structural semantics in Python or JavaScript merely to produce a score. A later slice should expose the Rust-owned structural boundaries through the reusable analysis contract and compare them with reference/human boundaries.

## Running the evidence

Metric semantics require only the standard library:

```sh
python scripts/test-dj-evaluation.py
```

The full real-music comparison requires librosa and optionally Essentia:

```sh
python scripts/evaluate-dj-goldens.py --output target/dj-goldens/dj-accuracy.json
```

The dedicated GitHub Actions workflow installs both reference analyzers, requires the live Essentia comparison, and uploads `dj-accuracy.json` as an artifact. Ordinary repository preflight remains independent of third-party audio downloads and Python analyzer availability.

## Interpretation boundary

This baseline answers a narrow question: what does the current Rust analyzer do on a reproducible set of real music, and how does it compare quantitatively with two established analyzers?

It does **not** justify a Mixxx-parity claim. Improvements to BPM/beat tracking, whole-track key analysis, structural modeling, and the overview waveform should use this report as a before/after evidence surface while expanding the corpus toward human/reference annotations.
