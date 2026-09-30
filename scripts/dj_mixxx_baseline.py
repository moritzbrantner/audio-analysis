"""Versioned accuracy baseline: independent oracles, never model agreement."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib


def independent_measurements(report: dict) -> dict:
    cases = {case["name"]: case for case in report["cases"]}
    measurements = {}
    for row in report["comparisons"]:
        name = row["case"]
        if name in measurements:
            continue
        metrics = row["metrics"]
        case = cases[name]
        if "authored" in metrics:
            authored = metrics["authored"]
            measurements[name] = dict(
                evidenceKind="synthetic-authored-ground-truth",
                oracleSha256=hashlib.sha256(json.dumps(case["groundTruth"], sort_keys=True).encode()).hexdigest(),
                beatF1=authored["rustBeatF1"],
                tempoExactMatch=authored["rustTempo"]["exactMatch"],
                tempoFamilyMatch=authored["rustTempo"]["familyMatch"],
                keyMatch=authored["rustKeyMatch"],
                tempoIntervalMeanAbsoluteErrorBpm=authored["rustTempoIntervals"]["meanAbsoluteErrorBpm"],
            )
        elif "creatorTempo" in metrics:
            measurements[name] = dict(
                evidenceKind="creator-described-tempo", referenceBpm=case["referenceBpm"],
                referenceBpmSource=case["referenceBpmSource"],
                tempoExactMatch=metrics["creatorTempo"]["rust"]["exactMatch"],
                tempoFamilyMatch=metrics["creatorTempo"]["rust"]["familyMatch"],
            )
    return measurements


def establish_baseline(report: dict, path: pathlib.Path) -> dict:
    if not report["coverage"]["fullCorpus"] or not report["coverage"]["authoredIncluded"] or report["failures"]:
        raise RuntimeError("a baseline requires a complete, repeat-consistent benchmark")
    baseline = {
        "schemaVersion": "audio-analysis-dj-mixxx-baseline/v1",
        "claimBoundary": "These are measured independent-oracle results, including known failures. Model agreement is not an accuracy gate.",
        "source": report["audioSource"], "referenceRuntime": report["runtime"],
        "inputSha256": {case["name"]: case["sha256"] for case in report["cases"]},
        "noiseThresholds": {
            "beatF1Absolute": 0.02, "beatF1Relative": 0.03,
            "tempoIntervalAbsoluteBpm": 1.0, "tempoIntervalRelative": 0.05,
        },
        "independentMeasurements": independent_measurements(report),
        "modelAgreementSummary": report["aggregate"],
    }
    # Package lists remain in the full runtime artifact; the baseline keeps the
    # exact binary/dependency fingerprints without copying hundreds of entries.
    baseline["referenceRuntime"] = {key: value for key, value in report["runtime"].items() if key != "packages"}
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    return dict(status="established", path=str(path), baselineSha256=hashlib.sha256(path.read_bytes()).hexdigest(), regressions=[])


def compare_baseline(report: dict, path: pathlib.Path) -> dict:
    if not report["coverage"]["fullCorpus"] or not report["coverage"]["authoredIncluded"]:
        return dict(status="focused-no-accuracy-gate", regressions=[])
    baseline = json.loads(path.read_text())
    if baseline.get("schemaVersion") != "audio-analysis-dj-mixxx-baseline/v1":
        raise RuntimeError("unsupported Mixxx accuracy baseline")
    inputs = {case["name"]: case["sha256"] for case in report["cases"]}
    if inputs != baseline["inputSha256"]:
        raise RuntimeError("benchmark inputs differ from the baseline; review a deliberate baseline update")
    for field in ["cargoLockSha256", "buildProfile", "rustCompiler", "harnessSha256", "manifestSha256", "corpusManifestSha256"]:
        if report["audioSource"][field] != baseline["source"][field]:
            raise RuntimeError(f"benchmark {field} differs from the baseline; review a deliberate baseline update")
    actual = independent_measurements(report)
    expected = baseline["independentMeasurements"]
    if set(actual) != set(expected):
        raise RuntimeError("independent oracle set differs from the baseline")
    noise = baseline["noiseThresholds"]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in noise.values()):
        raise RuntimeError("baseline noise thresholds must be finite nonnegative numbers")
    regressions = []
    for name, old in expected.items():
        new = actual[name]
        for field in ["evidenceKind", "oracleSha256", "referenceBpm", "referenceBpmSource"]:
            if old.get(field) != new.get(field):
                raise RuntimeError(f"independent oracle changed for {name}")
        for field in ["tempoExactMatch", "tempoFamilyMatch", "keyMatch"]:
            if old.get(field) is True and new.get(field) is not True:
                regressions.append(f"{name}: {field} regressed from a known match")
        for field, absolute, relative, direction in [
            ("beatF1", noise["beatF1Absolute"], noise["beatF1Relative"], -1),
            ("tempoIntervalMeanAbsoluteErrorBpm", noise["tempoIntervalAbsoluteBpm"], noise["tempoIntervalRelative"], 1),
        ]:
            previous = old.get(field)
            current = new.get(field)
            if previous is None:
                continue
            if type(previous) not in (int, float) or not math.isfinite(previous) or previous < 0:
                raise RuntimeError(f"invalid baseline measurement for {name}/{field}")
            if current is None or direction * (current - previous) > max(absolute, relative * abs(previous)) + 1e-12:
                regressions.append(f"{name}: {field} regressed beyond absolute and relative noise thresholds")
    runtime = report["runtime"]
    old_runtime = baseline["referenceRuntime"]
    equivalent = all(runtime[key] == old_runtime[key] for key in ["mixxxBinarySha256", "packagesSha256", "architecture"])
    return dict(status="regression" if regressions else "passed", baselineSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sourceRevision=baseline["source"]["revision"], mixxxRuntimeEquivalent=equivalent,
                independentOracleCount=len(expected), regressions=regressions)
