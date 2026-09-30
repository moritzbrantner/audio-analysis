# Beat placement under an isolated syncopated accent

Issue: [#69](https://github.com/moritzbrantner/audio-analysis/issues/69).
Baseline: merged #134, `8aa98ebd4038ee1454d47ae5c06a4cfdd5b4dbec`.
Foundation: declared source revision `fb51ab465ecd4d8086ac37ad0a9a2268b570e09f`.
Installed conventions: `f8cb06467d00f829d13d0c7c149162d6109f113f`, unchanged.

## Isolated decision failure

A 401-frame novelty envelope has nine independently authored pulses at frames
0, 50, ..., 400 (120 BPM at 100 frames/s), each with strength 0.5. Add one
strength-1 pulse at frame 238. With the correct tempo supplied directly, the
original dynamic program returns `0, 50, 100, 150, 200, 238, 300, 350, 400`.
The stronger anticipation displaces the supported frame-250 beat by 120 ms.
The same error occurs with a delayed accent at frame 262.

This isolates the failure to **beat-placement reward**, after onset extraction
and tempo inference: the tracker rewards an individual strong onset without
asking whether neighboring pulses support its phase. Changing the initial beat
or the global tempo cannot repair this interior displacement. It is not evidence
that every Solarity disagreement has the same cause.

The public PCM regression has nine 70-Hz exponentially decaying pulses at
0.5, 1.0, ..., 4.5 seconds, amplitude 0.2, plus one amplitude-0.8 pulse at
3.0 seconds minus 120 ms, plus 120 ms, or plus 240 ms. It uses 24-kHz PCM and
80-ms pulse widths. The authored beat grid is independent of every analyzer.
The original public path reports the anticipated beat at 2.880 seconds rather
than 3.0 and fails the two-hop timing gate. The candidate preserves all nine
beats within two hops, with the same 122.2826-BPM estimate. These tests are in
`track.rs`; no downloaded music or third-party reference is needed to run them.

## Narrow change

Weight final beat-placement novelty by onset support one local period before
and after each frame. Use the existing 18%-of-period onset-alignment tolerance;
average only available neighbor windows at track edges. An isolated onset is
thus discounted relative to a repeating pulse. Dynamic programming, onset
extraction, global tempo-candidate scoring, local tempo inference, reported
strengths, and confidence contracts retain their existing ownership.

Applying this weighting to tempo-candidate scoring changed the independent
four-against-three pulse-level result, so the change is deliberately confined to
placement after tempo selection. No threshold, corpus entry, authored gate,
Mixxx baseline, or tool/source pin was changed.

## Solarity diagnosis and limits

Unmodified source reproduces 123.626373 BPM and 423 beats. The candidate still
reports 123.626373 BPM and emits 404 beats. Its early displaced beat changes from
1.824 to 1.706667 seconds; that improvement in local pulse continuity does not
establish real-song beat truth. Local tempo excursions near the end toward
165.44 BPM and competing periodic phases remain further diagnostic leads.

The original Mixxx constant profile starts at 0.245458 seconds; the variable
profile starts at 0.493167 seconds, approximately half a beat apart at 124 BPM.
They are contradictory phase references, not annotations. Candidate agreement
with those profiles falls from 0.340321/0.318126 to 0.272727/0.250000. Agreement
with librosa changes from 0.387097 to 0.327827, while Essentia agreement rises
from 0.449383 to 0.485461. **Solarity's true phase remains unresolved.** Do not
claim that lower path irregularity, aggregate agreement, or this authored test
resolves its real-song phase. Independent beat annotations are still needed.

## Preserved acceptance evidence

The full 18-input corpus was evaluated with librosa 1.0.0 and Essentia
2.1b6.dev1389 before and after the change. All declared gates passed. Mean beat
F1 changes from 0.689146/0.722114 to 0.692556/0.735716. Tempo-family counts stay
12/18 and 14/18; asserted creator tempo/key gates and modulation coverage pass.
Individual disagreements remain visible below.

The complete eight authored Mixxx scenarios were freshly analyzed by Rust
against unchanged independently authored truth; mean authored F1 remains
0.954545. All ten independent measurements pass the unchanged versioned Mixxx
baseline, including its existing noise thresholds. Local model comparisons
reuse checksum-matched Mixxx captures from [#134's hosted artifact](https://github.com/moritzbrantner/audio-analysis/actions/runs/36668711418),
artifact `11076724918`; this is **not a fresh local Mixxx capture**. The existing
`DJ Mixxx Benchmark` and `DJ Real Music Evidence` workflows provide fresh hosted
verification when their PR labels are applied.

Repeat these checks in the declared source-development environment:

```sh
cargo test --locked -p moenarch-audio-analysis-rhythm
bash scripts/check-handoff.sh
python3 -m unittest discover -s tests/dj -p 'test_*.py'
python3 scripts/evaluate-dj-goldens.py
python3 scripts/evaluate-dj-mixxx.py
```

Local corpus runs used the same evaluator with the release-profile example to
avoid repeating full-song DSP in debug mode; the before run used an unchanged
baseline binary. The source graph and input checksums were held constant.

| Fixture | librosa F1 before | after | Essentia F1 before | after |
| --- | ---: | ---: | ---: | ---: |
| choice-drum-bass | 0.877193 | 0.982456 | 0.588235 | 0.658824 |
| brahms-hungarian-dance-5 | 0.467066 | 0.463415 | 0.460526 | 0.469799 |
| sweet-waltz | 0.979253 | 0.979253 | 0.995918 | 0.995918 |
| pistachio-ragtime | 0.926254 | 0.997050 | 0.943953 | 0.991150 |
| sugar-plum-fairy | 0.911215 | 0.941725 | 0.920930 | 0.918794 |
| vibe-ace | 0.574359 | 0.646154 | 0.505051 | 0.484848 |
| accelerating-snare | 0.592593 | 0.395062 | 0.625000 | 0.611765 |
| friendly-evil-gangsta-synth-hip-hop | 0.955499 | 0.931806 | 0.940470 | 0.984224 |
| pop-rock-wikiservicio | 0.959897 | 0.968831 | 0.956072 | 0.962387 |
| beethoven-pathetique-adagio-modulation | 0.365105 | 0.343369 | 0.677305 | 0.686327 |
| mj-pia-solarity-acid-techno | 0.387097 | 0.327827 | 0.449383 | 0.485461 |
| crooked-cop-reggae | 0.881671 | 0.913462 | 0.827982 | 0.855107 |
| dream-vilanculos-mini-light-trance | 0.793926 | 0.795652 | 0.993576 | 0.991416 |
| wikipedia-guitar-solo-f-sharp-minor | 0.975610 | 0.975610 | 0.958084 | 0.970060 |
| karissa-hobbs-lets-go-fishin | 0.352423 | 0.306667 | 0.616915 | 0.618090 |
| sorohanro-trumpet-loop | 0.250000 | 0.250000 | 0.631579 | 0.736842 |
| drese-wind-ensemble | 0.579710 | 0.716418 | 0.407080 | 0.450450 |
| drese-expressive-trumpet | 0.575758 | 0.531250 | 0.500000 | 0.371429 |
