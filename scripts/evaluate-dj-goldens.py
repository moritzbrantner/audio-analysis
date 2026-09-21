#!/usr/bin/env python3
"""Evaluate the Rust DJ analysis path against established music analyzers.

This is intentionally opt-in rather than ordinary CI: it downloads a checksum-
pinned, reusable music corpus, runs the Rust whole-track analyzer, and compares
its musical outputs with librosa and, when installed, Essentia.

Python requirements:
    python -m pip install numpy librosa
Optional stronger cross-check:
    python -m pip install essentia

Corpus metadata and coverage requirements live in the versioned DJ fixture manifest.
Use --check-manifest for the cheap offline validation path. Downloaded audio is
stored under target/dj-goldens and never committed. Pinned key references were
captured from Essentia 2.1b6.dev1389 against the
exact fixture bytes. Other fixtures deliberately retain analyzer disagreement
as evidence rather than manufacturing ground truth from filenames or titles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "target" / "dj-goldens"
CORPUS_MANIFEST = ROOT / "tests" / "fixtures" / "dj" / "real-music-corpus.v1.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
BEAT_TOLERANCE_SECONDS = 0.07


@dataclass(frozen=True)
class Fixture:
    name: str
    filename: str
    sha256: str
    license: str
    category: str
    coverage: tuple[str, ...]
    focus: tuple[str, ...]
    source_url: str | None = None
    provenance_url: str | None = None
    reference_key: str | None = None
    reference_key_source: str | None = None
    assert_key: bool = False
    assert_tempo: bool = False


@dataclass(frozen=True)
class Corpus:
    source_repository: str
    source_revision: str
    audio_root: str
    required_coverage: tuple[str, ...]
    fixtures: tuple[Fixture, ...]

    @property
    def raw_audio_root(self) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.source_repository}/"
            f"{self.source_revision}/{self.audio_root}"
        )


def string_list(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    if not allow_empty and not value:
        raise RuntimeError(f"{label} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise RuntimeError(f"{label} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise RuntimeError(f"{label} must not contain duplicates")
    return tuple(value)


def optional_https_url(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} must be null or a non-empty HTTPS URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(f"{label} must use an absolute HTTPS URL")
    return value


def load_corpus() -> Corpus:
    payload = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != "audio-analysis-dj-corpus/v1":
        raise RuntimeError("unsupported DJ corpus manifest schema")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("DJ corpus manifest source must be an object")
    repository = source.get("repository")
    revision = source.get("revision")
    audio_root = source.get("audioRoot")
    if (
        not isinstance(repository, str)
        or repository.count("/") != 1
        or not all(repository.split("/"))
    ):
        raise RuntimeError("DJ corpus source repository must use owner/repository form")
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise RuntimeError("DJ corpus source revision must be a full 40-character commit SHA")
    if (
        not isinstance(audio_root, str)
        or not audio_root.strip("/")
        or pathlib.PurePosixPath(audio_root).is_absolute()
        or ".." in pathlib.PurePosixPath(audio_root).parts
    ):
        raise RuntimeError("DJ corpus audioRoot must be a safe relative repository path")

    required_coverage = string_list(payload.get("requiredCoverage"), "requiredCoverage")
    required_set = set(required_coverage)
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise RuntimeError("DJ corpus manifest must contain fixtures")

    fixtures: list[Fixture] = []
    names: set[str] = set()
    filenames: set[str] = set()
    for index, item in enumerate(raw_fixtures):
        label = f"fixtures[{index}]"
        if not isinstance(item, dict):
            raise RuntimeError(f"{label} must be an object")

        name = item.get("name")
        filename = item.get("filename")
        checksum = item.get("sha256")
        license_name = item.get("license")
        category = item.get("category")
        source_url = optional_https_url(item.get("sourceUrl"), f"{label}.sourceUrl")
        provenance_url = optional_https_url(
            item.get("provenanceUrl"), f"{label}.provenanceUrl"
        )
        reference_key = item.get("referenceKey")
        reference_key_source = item.get("referenceKeySource")
        assert_key = item.get("assertKey", False)
        assert_tempo = item.get("assertTempo", False)

        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"{label}.name must be a non-empty string")
        if name in names:
            raise RuntimeError(f"duplicate DJ corpus fixture name: {name}")
        names.add(name)

        if (
            not isinstance(filename, str)
            or not filename.strip()
            or pathlib.PurePosixPath(filename).name != filename
            or filename in {".", ".."}
        ):
            raise RuntimeError(f"{label}.filename must be a plain filename")
        if filename in filenames:
            raise RuntimeError(f"duplicate DJ corpus filename: {filename}")
        filenames.add(filename)

        if not isinstance(checksum, str) or SHA256_PATTERN.fullmatch(checksum) is None:
            raise RuntimeError(f"{label}.sha256 must be a lowercase SHA-256 digest")
        if not isinstance(license_name, str) or not license_name.strip():
            raise RuntimeError(f"{label}.license must be a non-empty string")
        if not isinstance(category, str) or not category.strip():
            raise RuntimeError(f"{label}.category must be a non-empty string")
        if (source_url is None) != (provenance_url is None):
            raise RuntimeError(
                f"{label}.sourceUrl and {label}.provenanceUrl must be declared together"
            )
        if reference_key is not None and (
            not isinstance(reference_key, str) or not reference_key.strip()
        ):
            raise RuntimeError(f"{label}.referenceKey must be null or a non-empty string")
        if reference_key_source is not None and (
            not isinstance(reference_key_source, str) or not reference_key_source.strip()
        ):
            raise RuntimeError(
                f"{label}.referenceKeySource must be null or a non-empty string"
            )
        if reference_key is None and reference_key_source is not None:
            raise RuntimeError(
                f"{label}.referenceKeySource requires {label}.referenceKey"
            )
        if not isinstance(assert_key, bool):
            raise RuntimeError(f"{label}.assertKey must be a boolean")
        if assert_key and reference_key is None:
            raise RuntimeError(f"{label}.assertKey requires {label}.referenceKey")
        if not isinstance(assert_tempo, bool):
            raise RuntimeError(f"{label}.assertTempo must be a boolean")

        coverage = string_list(item.get("coverage", []), f"{label}.coverage", allow_empty=True)
        unknown_coverage = sorted(set(coverage) - required_set)
        if unknown_coverage:
            raise RuntimeError(
                f"{label}.coverage contains undeclared requirements: {unknown_coverage}"
            )
        focus = string_list(item.get("focus"), f"{label}.focus")

        fixtures.append(
            Fixture(
                name=name,
                filename=filename,
                sha256=checksum,
                license=license_name,
                category=category,
                coverage=coverage,
                focus=focus,
                source_url=source_url,
                provenance_url=provenance_url,
                reference_key=reference_key,
                reference_key_source=reference_key_source,
                assert_key=assert_key,
                assert_tempo=assert_tempo,
            )
        )

    return Corpus(
        source_repository=repository,
        source_revision=revision,
        audio_root=audio_root.strip("/"),
        required_coverage=required_coverage,
        fixtures=tuple(fixtures),
    )


def coverage_summary(corpus: Corpus) -> dict[str, Any]:
    covered_set = {tag for fixture in corpus.fixtures for tag in fixture.coverage}
    covered = [tag for tag in corpus.required_coverage if tag in covered_set]
    missing = [tag for tag in corpus.required_coverage if tag not in covered_set]
    return {
        "required": list(corpus.required_coverage),
        "covered": covered,
        "missing": missing,
        "complete": not missing,
    }


def fixture_source_url(corpus: Corpus, fixture: Fixture) -> str:
    return fixture.source_url or f"{corpus.raw_audio_root}/{fixture.filename}"


def download_fixture(corpus: Corpus, fixture: Fixture) -> pathlib.Path:
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
    request = urllib.request.Request(
        fixture_source_url(corpus, fixture),
        headers={
            "User-Agent": (
                "audio-analysis-dj-corpus/1.0 "
                "(+https://github.com/moritzbrantner/audio-analysis)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
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


def key_timeline_metrics(rust: dict[str, Any]) -> dict[str, Any]:
    timeline = rust.get("keyTimeline")
    timeline = timeline if isinstance(timeline, list) else []
    segments = rust.get("keySegments")
    segments = segments if isinstance(segments, list) else []
    labels = sorted(
        {
            label
            for segment in segments
            if isinstance(segment, dict)
            for key in [segment.get("key")]
            if isinstance(key, dict)
            for label in [key.get("label")]
            if isinstance(label, str) and label
        }
    )
    known_windows = sum(
        isinstance(window, dict) and isinstance(window.get("key"), dict)
        for window in timeline
    )
    return {
        "keyTimelineWindowCount": len(timeline),
        "keyTimelineKnownWindowCount": known_windows,
        "keySegmentCount": len(segments),
        "distinctStableKeys": labels,
        "keyBoundaryAligned": rust.get("keyBoundaryAligned") is True,
        "keyTimelineComplete": rust.get("keyTimelineComplete") is True,
    }


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
        **key_timeline_metrics(rust),
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


def aggregate_metrics(
    reports: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    metrics = [report["metrics"] for report in reports]
    pinned_key_reports = [report for report in reports if report["referenceKey"] is not None]
    pinned_key_matches = sum(
        (report["rust"].get("key") or {}).get("label") == report["referenceKey"]
        for report in pinned_key_reports
    )
    asserted_key_reports = [report for report in pinned_key_reports if report.get("assertKey")]
    modulation_reports = [
        report for report in reports if "modulation" in report.get("coverage", ())
    ]
    modulation_multi_key = sum(
        len(report["metrics"]["distinctStableKeys"]) > 1
        for report in modulation_reports
    )
    return {
        "fixtureCount": len(reports),
        "categories": sorted({report["category"] for report in reports}),
        "coverage": coverage,
        "modulationFixtureCount": len(modulation_reports),
        "modulationFixturesWithMultipleStableKeys": modulation_multi_key,
        "tempoEquivalentLibrosa": sum(item["tempoEquivalentLibrosa"] is True for item in metrics),
        "tempoEquivalentEssentia": sum(item["tempoEquivalentEssentia"] is True for item in metrics),
        "essentiaFixtureCount": sum(report["essentia"] is not None for report in reports),
        "meanBeatF1Librosa": mean_present([item["beatF1Librosa"] for item in metrics]),
        "meanBeatF1Essentia": mean_present([item["beatF1Essentia"] for item in metrics]),
        "pinnedKeyMatches": pinned_key_matches,
        "pinnedKeyFixtureCount": len(pinned_key_reports),
        "assertedKeyFixtureCount": len(asserted_key_reports),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-manifest",
        action="store_true",
        help="validate corpus metadata and report coverage without downloading audio",
    )
    args = parser.parse_args()
    corpus = load_corpus()
    coverage = coverage_summary(corpus)

    if args.check_manifest:
        print(
            json.dumps(
                {
                    "schemaVersion": "audio-analysis-dj-corpus-check/v1",
                    "manifest": str(CORPUS_MANIFEST.relative_to(ROOT)),
                    "source": {
                        "repository": corpus.source_repository,
                        "revision": corpus.source_revision,
                        "audioRoot": corpus.audio_root,
                    },
                    "fixtureCount": len(corpus.fixtures),
                    "externalFixtureCount": sum(
                        fixture.source_url is not None for fixture in corpus.fixtures
                    ),
                    "externalSources": [
                        {
                            "fixture": fixture.name,
                            "sourceUrl": fixture.source_url,
                            "provenanceUrl": fixture.provenance_url,
                        }
                        for fixture in corpus.fixtures
                        if fixture.source_url is not None
                    ],
                    "coverage": coverage,
                },
                indent=2,
            )
        )
        return 0

    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for fixture in corpus.fixtures:
        path = download_fixture(corpus, fixture)
        rust = rust_analysis(path)
        librosa = librosa_analysis(path)
        essentia = essentia_analysis(path)
        metrics = fixture_metrics(rust, librosa, essentia)
        report = {
            "fixture": fixture.name,
            "category": fixture.category,
            "coverage": fixture.coverage,
            "focus": fixture.focus,
            "license": fixture.license,
            "sha256": fixture.sha256,
            "sourceUrl": fixture_source_url(corpus, fixture),
            "provenanceUrl": fixture.provenance_url,
            "referenceKey": fixture.reference_key,
            "referenceKeySource": fixture.reference_key_source,
            "assertKey": fixture.assert_key,
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
            metrics["referenceKeyMatch"] = rust_key == fixture.reference_key
            if fixture.assert_key and rust_key != fixture.reference_key:
                failures.append(
                    f"{fixture.name}: Rust key {rust_key!r} != authoritative reference "
                    f"{fixture.reference_key!r}"
                )
            if essentia is not None:
                metrics["liveEssentiaReferenceKeyMatch"] = (
                    essentia["key"] == fixture.reference_key
                )

    output = {
        "schemaVersion": "audio-analysis-dj-goldens/v3",
        "beatToleranceSeconds": BEAT_TOLERANCE_SECONDS,
        "corpus": {
            "manifest": str(CORPUS_MANIFEST.relative_to(ROOT)),
            "sourceRepository": corpus.source_repository,
            "sourceRevision": corpus.source_revision,
        },
        "aggregate": aggregate_metrics(reports, coverage),
        "fixtures": reports,
        "failures": failures,
    }
    print(json.dumps(output, indent=2))
    if coverage["missing"]:
        print(
            "note: DJ acceptance coverage is incomplete; missing "
            + ", ".join(coverage["missing"]),
            file=sys.stderr,
        )
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
