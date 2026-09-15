# audio-analysis-karaoke

Deterministic construction of neutral vocal/karaoke charts from already-aligned lyric fragments and pitch observations.

This crate owns the fusion and karaoke-format boundary. It does not transcribe lyrics, separate vocals, estimate tempo, or detect pitch; those remain owned by the transcription, separation, rhythm, and pitch capabilities.

## First slice

The initial surface supports one lead voice with regular pitched notes:

- aligned lyric fragments with timestamps and optional alignment confidence,
- `audio-analysis-pitch` sung-note segments,
- deterministic best-overlap fusion with retained evidence,
- phrase grouping by a configurable silence gap,
- a neutral time-based `KaraokeChart`, and
- UltraStar v1 text export.

For highest-quality charts, callers should provide syllable-level lyric fragments. Word-level alignments are valid input but intentionally remain word-level; this crate does not guess syllable boundaries.

UltraStar-specific timing and C4-relative pitch encoding stay in the exporter. The neutral chart keeps seconds, MIDI pitch, phrase structure, and evidence so other karaoke/game formats can consume the same analysis later.
