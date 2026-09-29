# Checks

The normal incremental gate is `scripts/check-fast.sh`. The PR/release-oriented gate is `scripts/check-preflight.sh`; it adds documentation, package, and adapter-parity checks. An exact release issue additionally requires a clean checkout, `cargo package` for each public crate, and the native-whisperx temporary-patch consumer matrix.

This bootstrap intentionally has no verification receipt. The commands are recorded for later execution; their presence is not evidence that they passed.
