# Candle ZIP 9 source backport

Audio Analysis consumes Candle 0.10.2 at immutable fork revision
`b57f9eb7ccc6bf5cebf0c3822ed69573b06877a9`. It is the upstream `0.10.2` tag plus
the four-file ZIP 9 compatibility backport from huggingface/candle#4041,
`70f72e9cfdbfc570846ca363ead2135394af3e9a`. Package versions and tensor/model
execution are unchanged. No publication is required.

The patch upgrades ZIP to 9 and propagates its fallible filename API in NPZ/PTH
readers. Serialization tests cover borrowed UTF-8, owned CP437, malformed NPZ
and malformed PTH, alongside the existing NPY/NPZ/safetensors round trips.
All seven serialization tests and all CPU core library/integration tests pass.

The workspace lock resolves ZIP 9.0.5. Foundation source-lock fingerprint was
intentionally refreshed with the same Foundation revision after this dependency
transaction. `scripts/check-agent-readiness.sh --with-source` and
`scripts/check-handoff.sh` pass; existing karaoke Clippy warnings remain.
Shared convention source: `6cc05789dab39cee6758b288f83fb48b4ee6dbdb`.

Cargo does not inherit a dependency workspace's patches. Applications using
published Audio Analysis must provide the same root override until a compatible
Candle release includes the fix. Native WhisperX #366 owns this consumer change,
including its separate Candle 0.11 translation family. Remove this override in
a deliberate dependency update once the compatible upstream release exists;
registry-only publication remains a separate gate.
