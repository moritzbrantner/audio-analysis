# audio-generation-tts-cli

Thin command-line adapter for `moritzbrantner-audio-generation-tts`.

```bash
cargo run -p audio-generation-tts-cli -- operations --json
cargo run -p audio-generation-tts-cli -- run --operation audio.tts.synthesize --json '{"text":"Hello"}'
```
