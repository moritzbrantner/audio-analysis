# Domain context

`audio-analysis` owns reusable audio decoding, signal analysis, recognition,
speaker, transcription, synthesis, MIDI, TTS, and focused adapter surfaces.

Generic audio stream metadata and finite audio decoding are owned here. Visual
analysis, video frame decoding, application composition, and broad multimodal
UI remain outside this repository. Shared media/runtime/data contracts are
consumed only through exact published foundation crates; transcript/model
contracts are consumed through exact published NLP crates.

The extraction source remains authoritative until a separately authorized
release and migration complete. This repository does not remove source code or
publish packages as part of issue #2.
