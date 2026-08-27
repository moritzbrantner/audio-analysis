# audio-timed-text-compat

Private, unpublished migration shim for the neutral timed-text ownership cutover.

It re-exports the timed-text DTOs from `moenarch-media-core` under the historical `text_transcripts` Rust crate name so audio source can move off the NLP implementation dependency without introducing a second concrete transcript type.

No parsing, formatting, NLP enrichment, model behavior, or independently versioned API belongs here. Remove this shim after the remaining audio imports have been renamed directly to `media_core`.
