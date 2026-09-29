# ADR 0012: Audio capability repository boundary

## Status

Accepted from the source repository's capability-repository split decision.

## Decision

`moritzbrantner/audio-analysis` owns audio IO/analysis, recognition, separation, transcription execution, synthesis, MIDI, TTS, and native audio adapters. It may depend only on released foundation packages and narrow published NLP/transcript contracts. It must not commit path dependencies to sibling repositories or moving-branch Git dependencies.

Generic media/time, source metadata, cancellation, diagnostics, and model lifecycle remain foundation-owned. Transcript documents/formats remain narrow NLP-owned; this repository produces and consumes them for transcription.

Focused CLI, server, WASM, npm, and app adapters remain until a separately authorized migration decision. Source removal from `rust-packages` is prohibited until release, registry, consumer, compatibility-signpost, and rollback gates are complete.

## Consequences

Publication requires a later exact release issue, reviewed immutable manifest, clean-checkout evidence, package checks, and registry-only consumer proof. This bootstrap authorizes none of those actions.
