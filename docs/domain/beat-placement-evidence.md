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

## November 2, 2026: steady-trajectory phase continuity

Issue [#223](https://github.com/moritzbrantner/audio-analysis/issues/223) remained open after
merged [#228](https://github.com/moritzbrantner/audio-analysis/pull/228) corrected a late 4:3
tempo alias on Solarity. The new baseline was 388 beats, with beat F1
0.3709 against librosa and 0.5290 against Essentia; 65 oracle gaps remained.

An independently authored 64-beat, 124-BPM PCM track reproduces a separate
phase error. Quiet 70-Hz quarter-note kicks continue through four sections
with prominent, changing 1,400-Hz offbeat layers. The original tracker picks
the correct tempo and beat count, but beat 35 is displaced by 117 ms. This
distinguishes placement from octave selection and absence of kick evidence.

The repair doubles the period-deviation penalty **only when a local tempo path
exists and at least 95% of its frames remain within 4% of the anchor tempo**.
Tempo-candidate rescoring and genuinely variable local trajectories keep the
original penalty. The authored multisection PCM test fails before the change,
passes afterward, and retains the existing real-tempo-change regressions.

The checksum-pinned 18-track DJ oracle comparison found:

| Measure | Merged #228 | Steady-path correction |
| --- | ---: | ---: |
| Solarity beat F1 vs librosa | 0.3709 | 0.4430 |
| Solarity beat F1 vs Essentia | 0.5290 | 0.6134 |
| Mean corpus beat F1 vs librosa | 0.7675 | 0.7723 |
| Mean corpus beat F1 vs Essentia | 0.7870 | 0.7931 |
| Known oracle gaps | 65 | 63 |
| New oracle failures | — | 0 |

The two retired gaps are `karissa-hobbs-lets-go-fishin/sections:essentia`
and `mj-pia-solarity-acid-techno/sections:librosa`. No previously passing
tempo, beat, key, or section check failed; 14 `beats:*` gaps remain. The
downbeat unit tests pass, but no independently annotated downbeat corpus is
available to establish real-song downbeat accuracy.

The existing `dj_analyze` example and `dj-oracle` report now retain beat
strength and 20-second timing windows alongside each local BPM range. These
allow a correct-tempo, wrong-phase segment to be distinguished from pulse-count,
low-onset-strength, or local-trajectory concerns without changing the gating
oracle. In Solarity's 120–160-second interval, a separate 124-BPM low-frequency
transient-grid diagnostic agrees with the Rust beats but disagrees with both
reference lists; this is **not** a human-authored ground truth. Librosa and
Essentia agree with each other on only about 0.613 beat F1 over the full
Solarity track. Further phase changes require independent tapped kick/quarter-
note and bar-phase annotations rather than tuning one oracle's timestamps into
the Rust analyzer.

## Further November 2 investigation: short returning-phase excursions

The 18-track oracle remains necessary for regression detection, but competing
reference phases on Solarity are not reliable independent beat truth. An
additional diagnostic uses [Google Magenta's Groove MIDI Dataset](https://magenta.withgoogle.com/datasets/groove),
licensed CC BY 4.0. The published MIDI contains human-played drum timings and
the recording's metronome tempo. These checks used the official MIDI-only
`groove-v1.0.0-midionly.zip` (SHA-256
`651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e`).
Each MIDI note-on was deterministically synthesized into a short decaying
sinusoidal drum transient at 22,050 Hz. These are **newly synthesized test
signals, not the dataset's recorded audio**. Expected beats follow the MIDI
quarter-note metronome grid; the F1 tolerance is the same ±70 ms as the oracle.

The original tracker was almost entirely aligned on four human performances,
but shifted twelve consecutive beats by approximately 80–130 ms in a steady
137-BPM disco recording, between 189.962 and 194.769 seconds. At the end of the
phrase it returned to the established phase, while the selected BPM was still
about 136. An independently authored 160-beat, 120-BPM novelty regression
reproduces that same failure with twelve stronger 110-ms anticipations after
eighty correctly aligned quarter notes. The raw dynamic program follows the
wrong phase for the phrase even though the metronome grid has an onset on
every beat.

The new post-tracking guard runs only for a steady local tempo and at least 48
beats. It robustly fits an interval and phase to the tracker output, requires
at least 90% of beats to agree with that phase, and considers only returning
departures of at most 16 beats bracketed by aligned beats. It can move a
displaced beat only to an **observed onset** within 18% of a period of the
established grid, subject to absolute and relative onset-strength floors. It
does not insert beats or alter global tempo selection, candidate scores,
novelty extraction, or the dynamic-programming path. An additional regression
ensures a missing grid onset is never synthesized by this correction.

| Human MIDI performance | Baseline beat F1 | New beat F1 | Important window |
| --- | ---: | ---: | --- |
| `drummer1/session3/7_dance-disco_120_beat_4-4` | 0.996 | 0.996 | unchanged |
| `drummer1/session1/101_dance-disco_120_beat_4-4` | 0.987 | 0.987 | unchanged |
| `drummer3/session2/25_hiphop_92_beat_4-4` | 0.975 | 0.975 | unchanged |
| `drummer8/session2/26_dance-disco_137_beat_4-4` | 0.975 | **0.997** | 180–200 s: **0.739 → 0.978** |

All 18 checksum-pinned DJ fixtures were freshly analyzed with the same Rust
release-profile example before and after. The table reports both fixed oracle
comparisons, including failures, rather than hiding unchanged metrics. “—”
denotes an analyzer that returned no scoreable beats.

| DJ fixture | librosa F1 before → after | Essentia F1 before → after | Gaps before → after |
| --- | ---: | ---: | ---: |
| choice-drum-bass | 0.983 → 0.983 | 0.659 → 0.659 | 4 → 4 |
| brahms-hungarian-dance-5 | 0.648 → 0.648 | 0.707 → 0.707 | 2 → 2 |
| sweet-waltz | 0.983 → 0.983 | 1.000 → 1.000 | 2 → 2 |
| pistachio-ragtime | 0.997 → 0.997 | 0.991 → 0.991 | 3 → 3 |
| sugar-plum-fairy | 0.942 → 0.942 | 0.919 → 0.919 | 1 → 1 |
| vibe-ace | 0.859 → 0.859 | 0.819 → 0.819 | 1 → 1 |
| accelerating-snare | 0.395 → 0.395 | 0.612 → 0.612 | 8 → 8 |
| friendly-evil-gangsta-synth-hip-hop | 0.931 → 0.931 | 0.984 → 0.984 | 0 → 0 |
| pop-rock-wikiservicio | 0.969 → 0.969 | 0.962 → 0.962 | 1 → 1 |
| beethoven-pathetique-adagio-modulation | 0.339 → 0.339 | 0.684 → 0.684 | 8 → 8 |
| mj-pia-solarity-acid-techno | 0.443 → 0.443 | 0.613 → 0.613 | 4 → 4 |
| crooked-cop-reggae | 0.909 → 0.909 | 0.862 → 0.862 | 1 → 1 |
| dream-vilanculos-mini-light-trance | 0.796 → 0.796 | 0.991 → 0.991 | 1 → 1 |
| wikipedia-guitar-solo-f-sharp-minor | 0.976 → 0.976 | 0.994 → 0.994 | 2 → 2 |
| karissa-hobbs-lets-go-fishin | 0.295 → 0.295 | 0.593 → 0.593 | 5 → 5 |
| sorohanro-trumpet-loop | — → — | — → — | 8 → 8 |
| drese-wind-ensemble | 0.716 → 0.716 | 0.451 → 0.451 | 7 → 7 |
| drese-expressive-trumpet | 0.949 → 0.949 | 0.643 → 0.643 | 5 → 5 |

The corpus means remain 0.7723 against librosa and 0.7931 against Essentia.
There are still **63 known oracle gaps, including 14 `beats:*` gaps**, and no
new failures. Tempo family counts (12/18 and 12/18), key matches (13/18 and
10/18), and section-boundary mean F-scores (0.3930 and 0.6547) are unchanged.
The one observed section boundary adjustment is 23 ms on Pistachio Ragtime,
without affecting its section F-score. Downbeat-phase regressions and genuine
tempo-change tests pass in Rust, but no independently annotated Solarity bar
phase has been established. Solarity still measures **0.4430 / 0.6134** against
the conflicting references; issue #223 must remain open for that investigation.
