# audio-karaoke-formats

Downstream file-format adapters for the neutral karaoke charts produced by
`audio-generation-midi`.

The neutral chart remains authoritative for timing, phrase structure, and MIDI
pitch. This crate owns format-specific grammar, metadata, and quantization so
UltraStar/SingStar-style compatibility rules do not leak into audio/MIDI note
generation.

## UltraStar v1

`export_ultrastar_v1` writes a deterministic UTF-8 UltraStar v1 text document
for one lead vocal track. The exporter currently emits regular pitched notes
only because the neutral chart does not yet model golden, rap, freestyle, or
duet semantics.

Timing is quantized only at this adapter boundary. Export fails closed when the
integer UltraStar grid would collapse a note to zero duration, make notes
overlap, place a note before `#GAP`, or leave no safe beat for a phrase marker.
It never stretches a collapsed note merely to make the file syntactically
valid.

```rust,ignore
use audio_karaoke_formats::{export_ultrastar_v1, UltraStarV1Metadata};

let metadata = UltraStarV1Metadata::new("Example", "Artist", "song.ogg");
let text = export_ultrastar_v1(&chart, &metadata)?;
# Ok::<(), audio_contracts::DetectError>(())
```
