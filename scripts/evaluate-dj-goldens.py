#!/usr/bin/env python3
"""Compare the Rust DJ analysis path with established music analyzers.

This is intentionally opt-in rather than ordinary CI: it downloads two reusable
music fixtures, runs the Rust whole-track analyzer, and compares its musical
outputs with librosa and, when installed, Essentia.

Python requirements:
    python -m pip install numpy librosa
Optional stronger cross-check:
    python -m pip install essentia

The downloaded audio is stored under target/dj-goldens and never committed.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "target" / "dj-goldens"


@dataclass(frozen=True)
class Fixture:
    name: str
    url: str
    sha256: str
    license: str
    known_key: str | None = None
    assert_tempo: bool = False


FIXTURES = (
    Fixture(
        name="choice-drum-bass",
        url=(
            "https://raw.githubusercontent.com/librosa/data/"
            "38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0/audio/"
            "admiralbob77_-_Choice_-_Drum-bass.ogg"
        ),
        sha256="ac644f9645e7c15174e4a4f8561e4d1448d7f6e59ff6b0556b310ebbced879bc",
        license="CC-BY-NC-4.0",
        assert_tempo=True,
    ),
    Fixture(
        name="brahms-hungarian-dance-5",
        url=(
            "https://raw.githubusercontent.com/librosa/data/"
            "38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0/audio/"
            "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).ogg"
        ),
        sha256="919b48aa4cc66a0357d2cd5728664c5ab8f15c4b3469460df4b59470d35d3e49",
        license="CC-PDM-1.0",
        known_key="F# minor",
    ),
)


def download_fixture(fixture: Fixture) -> pathlib.Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{fixture.name}.ogg"
    if path.exists() and sha256(path) == fixture.sha256:
        return path
    if path.exists():
        path.unlink()
    print(f"downloading {fixture.name} ({fixture.license})", file=sys.stderr)
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


def tempo_equivalent(actual: float, golden: float, tolerance: float = 0.035) -> bool:
    if actual <= 0.0 or golden <= 0.0:
        return False
    return any(
        abs(actual - golden * ratio) / max(actual, golden * ratio) <= tolerance
        for ratio in (0.5, 1.0, 2.0)
    )


def main() -> int:
    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for fixture in FIXTURES:
        path = download_fixture(fixture)
        rust = rust_analysis(path)
        librosa = librosa_analysis(path)
        essentia = essentia_analysis(path)
        report = {
            "fixture": fixture.name,
            "license": fixture.license,
            "sha256": fixture.sha256,
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

        if fixture.known_key is not None:
            rust_key = (rust.get("key") or {}).get("label")
            if rust_key != fixture.known_key:
                failures.append(
                    f"{fixture.name}: Rust key {rust_key!r} != known key {fixture.known_key!r}"
                )
            if essentia is not None and essentia["key"] != fixture.known_key:
                failures.append(
                    f"{fixture.name}: Essentia key {essentia['key']!r} differs from "
                    f"known key {fixture.known_key!r}"
                )

    print(json.dumps({"fixtures": reports, "failures": failures}, indent=2))
    if failures:
        for failure in failures:
            print(f"golden failure: {failure}", file=sys.stderr)
        return 1
    if all(report["essentia"] is None for report in reports):
        print(
            "note: Essentia is not installed; librosa and known-key checks ran, "
            "but the second independent golden was skipped",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
