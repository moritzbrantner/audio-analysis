#!/usr/bin/env python3
"""Evaluate the Rust DJ analysis path against established music analyzers.

This is intentionally opt-in rather than ordinary CI: it downloads a checksum-
pinned, reusable music corpus, runs the Rust whole-track analyzer, and compares
its musical outputs with librosa and, when installed, Essentia.

Python requirements:
    python -m pip install numpy librosa
Optional stronger cross-check:
    python -m pip install essentia

The downloaded audio is stored under target/dj-goldens and never committed.
Pinned key references were captured from Essentia 2.1b6.dev1389 against the
exact fixture bytes. Other fixtures deliberately retain analyzer disagreement
as evidence rather than manufacturing ground truth from filenames or titles.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "target" / "dj-goldens"
DATA_REVISION = "38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0"
RAW_AUDIO_ROOT = f"https://raw.githubusercontent.com/librosa/data/{DATA_REVISION}/audio"
BEAT_TOLERANCE_SECONDS = 0.07


@dataclass(frozen=True)
class Fixture:
    name: str
    filename: str
    sha256: str
    license: str
    category: str
    focus: tuple[str, ...]
    reference_key: str | None = None
    assert_tempo: bool = False

    @property
    def url(self) -> str:
        return f"{RAW_AUDIO_ROOT}/{self.filename}"


FIXTURES = (
    Fixture(
        name="choice-drum-bass",
        filename="admiralbob77_-_Choice_-_Drum-bass.ogg",
        sha256="ac644f9645e7c15174e4a4f8561e4d1448d7f6e59ff6b0556b310ebbced879bc",
        license="CC-BY-NC-4.0",
        category="electronic-dance",
        focus=("steady-tempo", "bass-heavy", "half-double-time"),
        reference_key="G major",
        assert_tempo=True,
    ),
    Fixture(
        name="brahms-hungarian-dance-5",
        filename="Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).ogg",
        sha256="919b48aa4cc66a0357d2cd5728664c5ab8f15c4b3469460df4b59470d35d3e49",
        license="CC-PDM-1.0",
        category="classical-live-tempo",
        focus=("tempo-drift", "orchestral", "weak-percussion"),
        reference_key="G minor",
        assert_tempo=True,
    ),
    Fixture(
        name="sweet-waltz",
        filename="147793__setuniman__sweet-waltz-0i-22mi.ogg",
        sha256="4baff8ebf1771c33618b58aa12ac3fbac1e0462894ae74247f2fb8e649d1c63b",
        license="CC-BY-NC-4.0",
        category="synth-waltz",
        focus=("three-beat-feel", "layered-arrangement", "structure-change"),
        assert_tempo=True,
    ),
    Fixture(
        name="pistachio-ragtime",
        filename="442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime.ogg",
        sha256="9617c9be55c128177b13c20fbc52178ed482e3545094517efb30a7db2798991e",
        license="CC-BY-NC-4.0",
        category="acoustic-piano",
        focus=("syncopation", "weak-kick", "harmonic-motion"),
        assert_tempo=True,
    ),
    Fixture(
        name="sugar-plum-fairy",
        filename="Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy.ogg",
        sha256="b5c1a3e26310e6618d3c124f458654cd235650fcb9db7d711302644566600484",
        license="CC-BY-4.0",
        category="orchestral-arrangement",
        focus=("sparse-intro", "dynamic-range", "structural-contrast"),
        assert_tempo=True,
    ),
    Fixture(
        name="vibe-ace",
        filename="Kevin_MacLeod_-_Vibe_Ace.ogg",
        sha256="6c23aed3dd5aa57f2b1652ecab68d15d9b82ad257f54e639eb2880ca09bc118a",
        license="CC-BY-4.0",
        category="jazz-electronic",
        focus=("instrumentation-change", "syncopation", "structure-change"),
        assert_tempo=True,
    ),
    Fixture(
        name="accelerating-snare",
        filename="snare-accelerate.ogg",
        sha256="c4b237c784504cec7896e5c4b4e99e0774cf249129b627da94bbbea9e2c6605a",
        license="CC-BY-4.0",
        category="tempo-drift-stress",
        focus=("tempo-drift", "variable-beat-map"),
        assert_tempo=False,
    ),
)


def download_fixture(fixture: Fixture) -> pathlib.Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / fixture.filename
    if path.exists() and sha256(path) == fixture.sha256:
        return path
    if path.exists():
        path.unlink()
    print(
        f"downloading {fixture.name} ({fixture.category}, {fixture.license})",
        file=sys.stderr,
    )
    urllib.request.urlretrieve(fixture.url, path)
    digest = sha256(path)
    if digest != fixture.sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"fixture checksum mismatch for {fixture.name}: {digest} != {fixture.sha256}"
        )
    return path


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rust_analysis(path: pathlib.Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "moenarch-audio-analysis-rhythm",
            "--example",
            "dj_analyze",
            "--",
            str(path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def librosa_analysis(path: pathlib.Path) -> dict[str, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "librosa golden requires `python -m pip install numpy librosa`"
        ) from error

    audio, sample_rate = librosa.load(path, sr=44_100, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    return {
        "tempo": tempo_value,
        "beats": [float(value) for value in beat_times],
        "meanChroma": [float(value) for value in np.mean(chroma, axis=1)],
    }


def essentia_analysis(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        import essentia.standard as es
    except ImportError:
        return None

    audio = es.MonoLoader(filename=str(path), sampleRate=44_100)()
    bpm, ticks, confidence, estimates, intervals = es.RhythmExtractor2013(
        method="multifeature"
    )(audio)
    key, scale, strength = es.KeyExtractor(sampleRate=44_100)(audio)
    return {
        "tempo": float(bpm),
        "rhythmConfidence": float(confidence),
        "beats": [float(value) for value in ticks],
        "tempoCandidates": [float(value) for value in estimates],
        "intervals": [float(value) for value in intervals],
        "key": f"{key} {scale}",
        "keyStrength": float(strength),
    }


def tempo_relative_error(actual: float, golden: float) -> float:
    if actual <= 0.0 or golden <= 0.0:
        return float("inf")
    return min(
        abs(actual - golden * ratio) / max(actual, golden * ratio)
        for ratio in (0.5, 1.0, 2.0)
    )


def tempo_equivalent(actual: float, golden: float, tolerance: float = 0.035) -> bool:
    return tempo_relative_error(actual, golden) <= tolerance


def beat_f1(
    actual: list[float],
    reference: list[float],
    tolerance: float = BEAT_TOLERANCE_SECONDS,
) -> float | None:
    if not actual or not reference:
        return None
    used: set[int] = set()
    matches = 0
    for beat in actual:
        best_index = None
        best_distance = tolerance
        for index, expected in enumerate(reference):
            if index in used:
                continue
            distance = abs(beat - expected)
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None:
            used.add(best_index)
            matches += 1
    precision = matches / len(actual)
    recall = matches / len(reference)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def fixture_metrics(
    rust: dict[str, Any],
    librosa: dict[str, Any],
    essentia: dict[str, Any] | None,
) -> dict[str, Any]:
    rhythm = rust.get("rhythm", {})
    rust_bpm = rhythm.get("bpm")
    rust_beats = [float(value) for value in rhythm.get("beats", [])]
    metrics: dict[str, Any] = {
        "tempoEquivalentLibrosa": False,
        "tempoErrorLibrosaPercent": None,
        "beatF1Librosa": beat_f1(rust_beats, librosa["beats"]),
        "tempoEquivalentEssentia": None,
        "tempoErrorEssentiaPercent": None,
        "beatF1Essentia": None,
    }
    if rust_bpm is not None:
        error = tempo_relative_error(float(rust_bpm), librosa["tempo"])
        metrics["tempoEquivalentLibrosa"] = error <= 0.035
        metrics["tempoErrorLibrosaPercent"] = error * 100.0
        if essentia is not None:
            error = tempo_relative_error(float(rust_bpm), essentia["tempo"])
            metrics["tempoEquivalentEssentia"] = error <= 0.035
            metrics["tempoErrorEssentiaPercent"] = error * 100.0
    if essentia is not None:
        metrics["beatF1Essentia"] = beat_f1(rust_beats, essentia["beats"])
    return metrics


def mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def aggregate_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [report["metrics"] for report in reports]
    pinned_key_reports = [report for report in reports if report["referenceKey"] is not None]
    pinned_key_matches = sum(
        (report["rust"].get("key") or {}).get("label") == report["referenceKey"]
        for report in pinned_key_reports
    )
    return {
        "fixtureCount": len(reports),
        "categories": sorted({report["category"] for report in reports}),
        "tempoEquivalentLibrosa": sum(item["tempoEquivalentLibrosa"] is True for item in metrics),
        "tempoEquivalentEssentia": sum(item["tempoEquivalentEssentia"] is True for item in metrics),
        "essentiaFixtureCount": sum(report["essentia"] is not None for report in reports),
        "meanBeatF1Librosa": mean_present([item["beatF1Librosa"] for item in metrics]),
        "meanBeatF1Essentia": mean_present([item["beatF1Essentia"] for item in metrics]),
        "pinnedKeyMatches": pinned_key_matches,
        "pinnedKeyFixtureCount": len(pinned_key_reports),
    }


def main() -> int:
    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for fixture in FIXTURES:
        path = download_fixture(fixture)
        rust = rust_analysis(path)
        librosa = librosa_analysis(path)
        essentia = essentia_analysis(path)
        metrics = fixture_metrics(rust, librosa, essentia)
        report = {
            "fixture": fixture.name,
            "category": fixture.category,
            "focus": fixture.focus,
            "license": fixture.license,
            "sha256": fixture.sha256,
            "referenceKey": fixture.reference_key,
            "metrics": metrics,
            "rust": rust,
            "librosa": librosa,
            "essentia": essentia,
        }
        reports.append(report)

        if fixture.assert_tempo:
            rust_bpm = rust.get("rhythm", {}).get("bpm")
            if rust_bpm is None or not tempo_equivalent(float(rust_bpm), librosa["tempo"]):
                failures.append(
                    f"{fixture.name}: Rust BPM {rust_bpm} is not equivalent to "
                    f"librosa BPM {librosa['tempo']:.3f}"
                )
            if essentia is not None and (
                rust_bpm is None
                or not tempo_equivalent(float(rust_bpm), essentia["tempo"])
            ):
                failures.append(
                    f"{fixture.name}: Rust BPM {rust_bpm} is not equivalent to "
                    f"Essentia BPM {essentia['tempo']:.3f}"
                )

        if fixture.reference_key is not None:
            rust_key = (rust.get("key") or {}).get("label")
            if rust_key != fixture.reference_key:
                failures.append(
                    f"{fixture.name}: Rust key {rust_key!r} != pinned Essentia reference "
                    f"{fixture.reference_key!r}"
                )
            if essentia is not None and essentia["key"] != fixture.reference_key:
                failures.append(
                    f"{fixture.name}: Essentia key {essentia['key']!r} differs from pinned "
                    f"reference {fixture.reference_key!r}"
                )

    output = {
        "schemaVersion": "audio-analysis-dj-goldens/v2",
        "beatToleranceSeconds": BEAT_TOLERANCE_SECONDS,
        "aggregate": aggregate_metrics(reports),
        "fixtures": reports,
        "failures": failures,
    }
    print(json.dumps(output, indent=2))
    if failures:
        for failure in failures:
            print(f"golden failure: {failure}", file=sys.stderr)
        return 1
    if all(report["essentia"] is None for report in reports):
        print(
            "note: Essentia is not installed; librosa tempo/beat evidence and pinned "
            "Essentia key references ran, but the live second implementation cross-check "
            "was skipped",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
