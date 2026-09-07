"""Pure metric helpers for DJ analysis accuracy evidence.

This module intentionally has no third-party dependencies so its scoring rules can
be tested in ordinary hosted validation independently of the opt-in real-music
analyzers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


TEMPO_EQUIVALENCE_RATIOS = (0.5, 1.0, 2.0)
DEFAULT_TEMPO_TOLERANCE = 0.035
DEFAULT_BEAT_TOLERANCE_SECONDS = 0.070


def tempo_comparison(
    actual_bpm: float,
    reference_bpm: float,
    tolerance: float = DEFAULT_TEMPO_TOLERANCE,
) -> dict[str, float | bool]:
    """Compare BPM while exposing the selected half/exact/double relationship."""
    if actual_bpm <= 0.0 or reference_bpm <= 0.0:
        return {
            "actualBpm": actual_bpm,
            "referenceBpm": reference_bpm,
            "referenceMultiplier": 1.0,
            "alignedReferenceBpm": reference_bpm,
            "relativeError": 1.0,
            "equivalent": False,
        }

    candidates = []
    for ratio in TEMPO_EQUIVALENCE_RATIOS:
        aligned = reference_bpm * ratio
        relative_error = abs(actual_bpm - aligned) / max(actual_bpm, aligned)
        candidates.append((relative_error, ratio, aligned))
    relative_error, ratio, aligned = min(candidates, key=lambda candidate: candidate[0])
    return {
        "actualBpm": actual_bpm,
        "referenceBpm": reference_bpm,
        "referenceMultiplier": ratio,
        "alignedReferenceBpm": aligned,
        "relativeError": relative_error,
        "equivalent": relative_error <= tolerance,
    }


def beat_alignment(
    actual_beats: Iterable[float],
    reference_beats: Iterable[float],
    tolerance_seconds: float = DEFAULT_BEAT_TOLERANCE_SECONDS,
) -> dict[str, float | int]:
    """Compute one-to-one beat precision/recall/F1 within a fixed time tolerance."""
    actual = sorted(float(value) for value in actual_beats)
    reference = sorted(float(value) for value in reference_beats)
    actual_index = 0
    reference_index = 0
    errors: list[float] = []

    while actual_index < len(actual) and reference_index < len(reference):
        delta = actual[actual_index] - reference[reference_index]
        if abs(delta) <= tolerance_seconds:
            errors.append(abs(delta))
            actual_index += 1
            reference_index += 1
        elif delta < 0.0:
            actual_index += 1
        else:
            reference_index += 1

    matches = len(errors)
    precision = matches / len(actual) if actual else 0.0
    recall = matches / len(reference) if reference else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    mean_error_ms = sum(errors) * 1000.0 / matches if matches else 0.0
    return {
        "toleranceMs": tolerance_seconds * 1000.0,
        "actualCount": len(actual),
        "referenceCount": len(reference),
        "matchedCount": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "meanAbsoluteErrorMs": mean_error_ms,
    }


def analyzer_metrics(
    rust: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Project comparable rhythm/key evidence from one analyzer result."""
    rust_rhythm = rust.get("rhythm") or {}
    metrics: dict[str, Any] = {}
    rust_bpm = rust_rhythm.get("bpm")
    reference_bpm = reference.get("tempo")
    if rust_bpm is not None and reference_bpm is not None:
        metrics["tempo"] = tempo_comparison(float(rust_bpm), float(reference_bpm))

    rust_beats = rust_rhythm.get("beats")
    reference_beats = reference.get("beats")
    if rust_beats is not None and reference_beats is not None:
        metrics["beats"] = beat_alignment(rust_beats, reference_beats)

    rust_key = (rust.get("key") or {}).get("label")
    reference_key = reference.get("key")
    if rust_key is not None and reference_key is not None:
        metrics["key"] = {
            "rust": rust_key,
            "reference": reference_key,
            "exactMatch": rust_key == reference_key,
        }
    return metrics


def aggregate_metrics(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate comparable evidence without converting ambiguity into certainty."""
    reports = list(reports)
    analyzers: dict[str, dict[str, Any]] = {}
    for analyzer_name in ("librosa", "essentia"):
        analyzer_reports = [
            report.get("metrics", {}).get(analyzer_name)
            for report in reports
            if report.get("metrics", {}).get(analyzer_name) is not None
        ]
        tempos = [entry["tempo"] for entry in analyzer_reports if "tempo" in entry]
        beats = [entry["beats"] for entry in analyzer_reports if "beats" in entry]
        keys = [entry["key"] for entry in analyzer_reports if "key" in entry]
        analyzers[analyzer_name] = {
            "fixtureCount": len(analyzer_reports),
            "tempoComparisonCount": len(tempos),
            "tempoEquivalentRate": mean(
                1.0 if entry["equivalent"] else 0.0 for entry in tempos
            ),
            "meanTempoRelativeError": mean(entry["relativeError"] for entry in tempos),
            "beatComparisonCount": len(beats),
            "meanBeatF1": mean(entry["f1"] for entry in beats),
            "meanBeatAbsoluteErrorMs": mean(
                entry["meanAbsoluteErrorMs"] for entry in beats
            ),
            "keyComparisonCount": len(keys),
            "exactKeyAgreementRate": mean(
                1.0 if entry["exactMatch"] else 0.0 for entry in keys
            ),
        }

    pinned_keys = [
        report
        for report in reports
        if report.get("reference", {}).get("dominantKey") is not None
    ]
    pinned_key_matches = [
        (report.get("rust", {}).get("key") or {}).get("label")
        == report["reference"]["dominantKey"]["value"]
        for report in pinned_keys
    ]
    return {
        "fixtureCount": len(reports),
        "categories": sorted({report["category"] for report in reports}),
        "pinnedDominantKeyCount": len(pinned_keys),
        "pinnedDominantKeyAccuracy": mean(
            1.0 if matched else 0.0 for matched in pinned_key_matches
        ),
        "analyzers": analyzers,
    }


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None
