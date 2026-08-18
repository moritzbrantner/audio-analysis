# Audio Analysis Context

**Capability Repository**: this repository is the release-owning home for a coherent audio boundary, independently buildable only against released foundation and narrow NLP contracts.

**Adapter Parity**: library, CLI, REST/server, WASM, npm, and focused app surfaces delegate to the same library-owned operations and preserve their request/response contracts.

**Speaker Diarization**: `moenarch-audio-analysis-speakers` owns diarization and transcript speaker assignment contracts. Transcription may orchestrate those contracts but does not own their schema.

**Speaker-conditioned TTS**: `moenarch-audio-generation-tts` owns Reference Voice Prompt planning and provider diagnostics. Consent and identity policy stay with downstream products; default builds remain model-free and network-free.

**Native Workflow**: a workflow requiring local tools, models, native runtimes, or materialization. It is opt-in and must not be selected silently by default.
