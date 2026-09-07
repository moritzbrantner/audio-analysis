#!/usr/bin/env python3
"""Measure the Rust whole-song DJ analysis path against established analyzers.

The real-music run is intentionally separate from ordinary deterministic CI. It
fetches immutable, checksum-pinned reusable music fixtures into ``target/``, runs
the repository's Rust analyzers, and records comparable evidence from librosa and
Essentia. Analyzer agreement is evidence, not human ground truth.

Python requirements:
    python -m pip install numpy librosa
Optional second live analyzer:
    python -m pip install essentia

Uploaded Audio Inspector audio is not involved in this evaluator and remains
browser-local. This script only fetches the public corpus fixtures declared below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any

from dj_evaluation import (
    DEFAULT_BEAT_TOLERANCE_SECONDS,
    DEFAULT_TEMPO_TOLERANCE,
    TEMPO_EQUIVALENCE_RATIOS,
    aggregate_metrics,
    analyzer_metrics,
    tempo_comparison,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "target" / "dj-goldens"
LIBROSA_DATA_REV = "38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0"
PINNED_ESSENTIA_VERSION = "2.1b6.dev1389"


@dataclass(frozen=True)
class Fixture:
    name: str
    category: str
    source_path: str
    license: str
    sha256: str | None = None
    git_blob_sha1: str | None = None
    reference_key: str | None = None
    assert_tempo: bool = False

    @property
    def url(self) -> str:
        return (
            "https://raw.githubusercontent.com/librosa/data/"
            f"{LIBROSA_DATA_REV}/audio/{self.source_path}"
        )


FIXTURES = (
    Fixture(
        name="choice-drum-bass",
        category="electronic-dance",
        source_path="admiralbob77_-_Choice_-_Drum-bass.ogg",
        sha256="ac644f9645e7c15174e4a4f8561e4d1448d7f6e59ff6b0556b310ebbced879bc",
        license="CC-BY-NC-4.0",
        reference_key="G major",
        assert_tempo=True,
    ),
    Fixture(
        name="brahms-hungarian-dance-5",
        category="classical-live-tempo",
        source_path=(
            "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_"
            "(string_orchestra).ogg"
        ),
        sha256="919b48aa4cc66a0357d2cd5728664c5ab8f15c4b3469460df4b59470d35d3e49",
        license="CC-PDM-1.0",
        reference_key="G minor",
        assert_tempo=True,
    ),
    Fixture(
        name="vibe-ace",
        category="jazz-arrangement",
        source_path="Kevin_MacLeod_-_Vibe_Ace.ogg",
        git_blob_sha1="d546326298b020752fcb7cc96d3f7146e43d6b99",
        license="CC-BY-4.0",
    ),
    Fixture(
        name="lets-go-fishin",
        category="pop-folk-vocals",
        source_path="Karissa_Hobbs_-_Lets_Go_Fishin.ogg",
        git_blob_sha1="c058853e327889fd44f31d97c957dfdfba97f959",
        license="CC-BY-NC-SA-4.0",
    ),
    Fixture(
        name="sweet-waltz",
        category="triple-meter",
        source_path="147793__setuniman__sweet-waltz-0i-22mi.ogg",
        git_blob_sha1="02493bc19430a120b7c0819c6790de23016e7a4b",
        license="CC-BY-NC-4.0",
    ),
    Fixture(
        name="pistachio-ragtime",
        category="syncopated-piano",
        source_path="442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime.ogg",
        git_blob_sha1="326bec9ad2690565a606d639b19c99ac38d64d3f",
        license="CC-BY-NC-4.0",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="write the complete JSON evidence document to this path",
    )
    parser.add_argument(
        "--require-essentia",
        action="store_true",
        help="fail when the live Essentia cross-check is unavailable",
    )
    return parser.parse_args()


def download_fixture(fixture: Fixture) -> pathlib.Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{fixture.name}.ogg"
    if path.exists() and fixture_checksum_matches(path, fixture):
        return path
    if path.exists():
        path.unlink()
    print(f"downloading {fixture.name} ({fixture.license})", file=sys.stderr)
    urllib.request.urlretrieve(fixture.url, path)
    if not fixture_checksum_matches(path, fixture):
        observed = fixture_checksums(path)
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"fixture checksum mismatch for {fixture.name}: observed {observed}"
        )
    return path


def fixture_checksum_matches(path: pathlib.Path, fixture: Fixture) -> bool:
    if fixture.sha256 is not None and sha256(path) != fixture.sha256:
        return False
    if fixture.git_blob_sha1 is not None and git_blob_sha1(path) != fixture.git_blob_sha1:
        return False
    if fixture.sha256 is None and fixture.git_blob_sha1 is None:
        raise RuntimeError(f"fixture {fixture.name} has no pinned checksum")
    return True


def fixture_checksums(path: pathlib.Path) -> dict[str, str]:
    return {"sha256": sha256(path), "gitBlobSha1": git_blob_sha1(path)}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: pathlib.Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rust_analysis(path: pathlib.Path) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "moenarch-audio-analysis-rhythm",
        "--example",
        "dj_analyze",
        "--",
        str(path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Rust DJ analyzer failed for "
            f"{path.name} with exit code {completed.returncode}:\n{completed.stderr.strip()}"
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


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def fixture_report(
    fixture: Fixture,
    rust: dict[str, Any],
    librosa: dict[str, Any],
    essentia: dict[str, Any] | None,
) -> dict[str, Any]:
    reference_key = (
        {
            "value": fixture.reference_key,
            "kind": "pinned-analyzer-reference",
            "analyzer": "Essentia KeyExtractor",
            "analyzerVersion": PINNED_ESSENTIA_VERSION,
        }
        if fixture.reference_key is not None
        else None
    )
    return {
        "fixture": fixture.name,
        "category": fixture.category,
        "license": fixture.license,
        "source": {
            "url": fixture.url,
            "revision": LIBROSA_DATA_REV,
            "sha256": fixture.sha256,
            "gitBlobSha1": fixture.git_blob_sha1,
        },
        "reference": {
            "dominantKey": reference_key,
            "structuralBoundariesSeconds": None,
        },
        "rust": rust,
        "librosa": librosa,
        "essentia": essentia,
        "metrics": {
            "librosa": analyzer_metrics(rust, librosa),
            "essentia": analyzer_metrics(rust, essentia) if essentia is not None else None,
        },
    }


def regression_failures(
    fixture: Fixture,
    rust: dict[str, Any],
    librosa: dict[str, Any],
    essentia: dict[str, Any] | None,
) -> list[str]:
    failures: list[str] = []
    if fixture.assert_tempo:
        rust_bpm = (rust.get("rhythm") or {}).get("bpm")
        if rust_bpm is None or not tempo_comparison(
            float(rust_bpm), float(librosa["tempo"])
        )["equivalent"]:
            failures.append(
                f"{fixture.name}: Rust BPM {rust_bpm} is not equivalent to "
                f"librosa BPM {librosa['tempo']:.3f}"
            )
        if essentia is not None and (
            rust_bpm is None
            or not tempo_comparison(float(rust_bpm), float(essentia["tempo"]))[
                "equivalent"
            ]
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
    return failures


def evidence_document(
    reports: list[dict[str, Any]], failures: list[str]
) -> dict[str, Any]:
    return {
        "schemaVersion": "audio-analysis-dj-accuracy/v1",
        "claimBoundary": (
            "Baseline analyzer-agreement evidence only; this report does not claim "
            "Mixxx parity or substitute analyzer agreement for human ground truth."
        ),
        "metricDefinitions": {
            "tempoToleranceFraction": DEFAULT_TEMPO_TOLERANCE,
            "tempoReferenceMultipliers": list(TEMPO_EQUIVALENCE_RATIOS),
            "beatToleranceMs": DEFAULT_BEAT_TOLERANCE_SECONDS * 1000.0,
            "beatMetric": "one-to-one precision/recall/F1 within the time tolerance",
            "keyMetric": "exact tonic + major/minor label agreement when available",
            "structureMetric": None,
        },
        "coverage": {
            "status": "baseline-incomplete",
            "knownGaps": [
                "human-annotated beat positions",
                "human/reference structural boundaries",
                "hip-hop fixture",
                "explicit half-time/double-time ambiguity fixture",
                "weak-intro fixture",
                "tempo-drift fixture with annotated beat map",
                "modulation fixture with annotated key timeline",
            ],
        },
        "analyzers": {
            "rust": "repository dj_analyze example using Rust rhythm + pitch crates",
            "librosaVersion": installed_version("librosa"),
            "essentiaVersion": installed_version("essentia"),
            "pinnedEssentiaReferenceVersion": PINNED_ESSENTIA_VERSION,
        },
        "aggregate": aggregate_metrics(reports),
        "fixtures": reports,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    essentia_missing = False

    for fixture in FIXTURES:
        path = download_fixture(fixture)
        rust = rust_analysis(path)
        librosa = librosa_analysis(path)
        essentia = essentia_analysis(path)
        essentia_missing = essentia_missing or essentia is None
        reports.append(fixture_report(fixture, rust, librosa, essentia))
        failures.extend(regression_failures(fixture, rust, librosa, essentia))

    if args.require_essentia and essentia_missing:
        failures.append("Essentia is required for this evidence run but is not installed")

    document = evidence_document(reports, failures)
    serialized = json.dumps(document, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")

    if failures:
        for failure in failures:
            print(f"golden failure: {failure}", file=sys.stderr)
        return 1
    if essentia_missing:
        print(
            "note: Essentia is not installed; librosa metrics and pinned Essentia key "
            "references ran, but live second-analyzer metrics were skipped",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
