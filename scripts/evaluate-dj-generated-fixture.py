#!/usr/bin/env python3
"""Evaluate one deterministic DJ fixture generated through pinned asset-tooling.

This complements the real-music golden corpus with authored ground truth. It is
synthetic evidence and must not be counted as real-song accuracy evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
RECIPE = ROOT / "tests" / "fixtures" / "dj" / "asset-tooling-half-time-song.v1.json"
GENERATOR = ROOT / "scripts" / "generate-dj-asset-tooling-fixture.mjs"
DEFAULT_REPORT = ROOT / "target" / "dj-goldens" / "generated" / "report.json"
BEAT_TOLERANCE_SECONDS = 0.07
DOWNBEAT_TOLERANCE_SECONDS = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-tooling",
        type=pathlib.Path,
        default=(ROOT.parent / "asset-tooling"),
        help="checkout of the exact asset-tooling revision pinned by the fixture",
    )
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_generation(
    asset_tooling: pathlib.Path,
    output: pathlib.Path,
    manifest: pathlib.Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "bun",
            str(GENERATOR),
            "--asset-tooling",
            str(asset_tooling),
            "--recipe",
            str(RECIPE),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return json.loads(manifest.read_text(encoding="utf-8"))


def rust_analysis(path: pathlib.Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
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


def beat_f1(
    actual: list[float],
    reference: list[float],
    tolerance: float,
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


def tempo_error(actual: float, expected: float) -> float:
    if actual <= 0.0 or expected <= 0.0:
        return float("inf")
    return abs(actual - expected) / expected


def tempo_family_error(actual: float, expected: float) -> float:
    if actual <= 0.0 or expected <= 0.0:
        return float("inf")
    return min(
        abs(actual - expected * ratio) / max(actual, expected * ratio)
        for ratio in (0.5, 1.0, 2.0)
    )


def main() -> int:
    args = parse_args()
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    audio = recipe["audio"]
    acceptance = recipe["acceptance"]
    ground_truth = recipe["groundTruth"]

    target = args.report.resolve().parent
    target.mkdir(parents=True, exist_ok=True)
    first_wav = target / "asset-tooling-half-time-song-a.wav"
    second_wav = target / "asset-tooling-half-time-song-b.wav"
    first_manifest_path = target / "asset-tooling-half-time-song-a.json"
    second_manifest_path = target / "asset-tooling-half-time-song-b.json"

    first_manifest = run_generation(
        args.asset_tooling.resolve(), first_wav, first_manifest_path
    )
    second_manifest = run_generation(
        args.asset_tooling.resolve(), second_wav, second_manifest_path
    )
    first_sha = sha256(first_wav)
    second_sha = sha256(second_wav)
    reproducible = (
        first_sha == second_sha
        and first_manifest["outputAsset"] == second_manifest["outputAsset"]
        and first_manifest["recipeSha256"] == second_manifest["recipeSha256"]
    )

    rust = rust_analysis(first_wav)
    expected_bpm = float(audio["bpm"])
    total_beats = int(audio["bars"]) * int(audio["beatsPerBar"])
    beat_period = 60.0 / expected_bpm
    beat_phase = float(ground_truth["beatPhaseSeconds"])
    reference_beats = [beat_phase + index * beat_period for index in range(total_beats)]
    reference_downbeats = [
        beat_phase + index * beat_period
        for index in range(0, total_beats, int(audio["beatsPerBar"]))
    ]

    rhythm = rust.get("rhythm", {})
    rust_bpm = rhythm.get("bpm")
    rust_beats = [float(value) for value in rhythm.get("beats", [])]
    rust_downbeats = [float(value) for value in rhythm.get("downbeats", [])]
    exact_tempo_error = (
        tempo_error(float(rust_bpm), expected_bpm) if rust_bpm is not None else float("inf")
    )
    family_tempo_error = (
        tempo_family_error(float(rust_bpm), expected_bpm)
        if rust_bpm is not None
        else float("inf")
    )
    authored_beat_f1 = beat_f1(
        rust_beats, reference_beats, BEAT_TOLERANCE_SECONDS
    )
    authored_downbeat_f1 = beat_f1(
        rust_downbeats, reference_downbeats, DOWNBEAT_TOLERANCE_SECONDS
    )
    rust_key = (rust.get("key") or {}).get("label")
    key_matches = rust_key == ground_truth["key"]
    tempo_tolerance = float(acceptance["tempoTolerancePercent"]) / 100.0

    metrics = {
        "reproducible": reproducible,
        "tempoExact": exact_tempo_error <= tempo_tolerance,
        "tempoFamilyEquivalent": family_tempo_error <= tempo_tolerance,
        "tempoExactErrorPercent": exact_tempo_error * 100.0,
        "tempoFamilyErrorPercent": family_tempo_error * 100.0,
        "beatF1Authored": authored_beat_f1,
        "downbeatF1Authored": authored_downbeat_f1,
        "keyMatchesAuthored": key_matches,
    }

    failures: list[str] = []
    findings: list[str] = []
    if not reproducible:
        failures.append("asset-tooling did not reproduce byte-identical audio and AssetRef identity")
    if not metrics["tempoFamilyEquivalent"]:
        failures.append(
            f"Rust BPM {rust_bpm!r} is outside the authored 120 BPM half/exact/double family"
        )
    elif not metrics["tempoExact"]:
        findings.append(
            f"tempo-family ambiguity remains: Rust selected {rust_bpm!r} BPM for authored 120 BPM"
        )
    if authored_beat_f1 is None or authored_beat_f1 < float(acceptance["minBeatF1"]):
        failures.append(
            f"authored beat F1 {authored_beat_f1!r} is below {acceptance['minBeatF1']}"
        )
    elif authored_beat_f1 < 0.95:
        findings.append(
            f"beat tracking is accepted but not near-perfect on exact authored beats: F1={authored_beat_f1:.3f}"
        )
    if authored_downbeat_f1 is None or authored_downbeat_f1 < float(
        acceptance["minDownbeatF1"]
    ):
        failures.append(
            f"authored downbeat F1 {authored_downbeat_f1!r} is below {acceptance['minDownbeatF1']}"
        )
    elif authored_downbeat_f1 < 0.85:
        findings.append(
            "bar-phase/downbeat tracking remains weak despite accepted beat tracking: "
            f"F1={authored_downbeat_f1:.3f}"
        )
    if acceptance["requireKey"] and not key_matches:
        failures.append(
            f"Rust key {rust_key!r} does not match authored key {ground_truth['key']!r}"
        )

    report = {
        "schemaVersion": "audio-analysis-dj-generated-evidence/v1",
        "evidenceKind": "synthetic-authored-ground-truth",
        "fixture": recipe["name"],
        "claimBoundary": (
            "Synthetic evidence exercises deterministic failure modes and must not be "
            "counted as real-music accuracy evidence."
        ),
        "generator": {
            "repository": recipe["generator"]["repository"],
            "revision": recipe["generator"]["revision"],
            "firstOutputSha256": first_sha,
            "secondOutputSha256": second_sha,
            "assetRef": first_manifest["outputAsset"],
            "placementCount": first_manifest["placementCount"],
            "sourceCount": first_manifest["sourceCount"],
        },
        "groundTruth": {
            "tempoBpm": expected_bpm,
            "key": ground_truth["key"],
            "beatCount": len(reference_beats),
            "downbeatCount": len(reference_downbeats),
            "sections": ground_truth["sections"],
        },
        "metrics": metrics,
        "findings": findings,
        "failures": failures,
        "rust": rust,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    for finding in findings:
        print(f"generated evidence finding: {finding}", file=sys.stderr)
    for failure in failures:
        print(f"generated evidence failure: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
