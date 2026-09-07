# Audio Analysis Playback

Reusable, deterministic playback and DJ-oriented planning helpers shared by browser, desktop, and media applications.

The crate intentionally owns pure calculations rather than player state or browser/native I/O. Initial coverage includes:

- equal-power crossfader gains;
- tempo-percent and playback-rate mapping;
- effective-BPM and BPM-sync planning;
- beat-grid-aligned loop selection;
- bounded-resolution waveform extrema summaries.

These helpers were extracted from consumer behavior in `dj-party` so other projects such as `media-player` can reuse one implementation. Some product behavior is informed by established DJ-software workflows, including Mixxx, but the Rust implementation here is independent: no Mixxx source code is copied or translated. This repository remains licensed under MIT OR Apache-2.0.

Application-specific policy stays with the consumer. For example, `dj-party` decides its allowed tempo range, permitted loop sizes, maximum waveform point count, and WASM-facing invalid-value behavior, while this crate owns the reusable mathematics.
