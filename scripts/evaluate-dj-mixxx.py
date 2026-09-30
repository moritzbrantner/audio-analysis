#!/usr/bin/env python3
"""Compare the expanded DJ corpus and authored PCM with actual Mixxx analysis."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile

from dj_benchmark_cases import generate_case
from dj_mixxx_reference import capture_job
from dj_mixxx_baseline import compare_baseline, establish_baseline


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/dj/mixxx-benchmark.v1.json"
BASELINE = ROOT / "tests/fixtures/dj/mixxx-baseline.v1.json"
SPEC = importlib.util.spec_from_file_location("dj_goldens", ROOT / "scripts/evaluate-dj-goldens.py")
goldens = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goldens
SPEC.loader.exec_module(goldens)


def load_config() -> dict:
    config = json.loads(MANIFEST.read_text())
    if config.get("schemaVersion") != "audio-analysis-mixxx-benchmark/v1":
        raise RuntimeError("unsupported Mixxx benchmark manifest")
    if config.get("mixxxVersion") != "2.5.4" or config.get("mixxxPackageVersion") != "2.5.4+dfsg-1build1":
        raise RuntimeError("Mixxx adapter supports only the reviewed 2.5.4 runtime")
    if config.get("mixxxSourceRevision") != "073e5ff876b6d1ed2cb6843f618cafaa6597daed":
        raise RuntimeError("Mixxx source revision does not match the database adapter")
    if config.get("repetitions") != 2:
        raise RuntimeError("Mixxx benchmark requires two independent fresh captures")
    if any(type(profile.get("fixedTempo")) is not bool for profile in config.get("profiles", [])):
        raise RuntimeError("Mixxx fixedTempo must be boolean")
    if config.get("profiles") != [
        {"name": "whole-track-constant", "fixedTempo": True},
        {"name": "whole-track-variable", "fixedTempo": False},
    ]:
        raise RuntimeError("Mixxx benchmark requires both whole-track profiles")
    for field in ["beatToleranceSeconds", "tempoToleranceRelative"]:
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise RuntimeError(f"invalid benchmark {field}")
    cases = config.get("authoredCases")
    if not isinstance(cases, list) or len(cases) < 8:
        raise RuntimeError("Mixxx benchmark requires at least eight authored cases")
    names = set()
    patterns = {"steady", "half-time", "weak-intro", "offbeat", "syncopated", "sparse", "drift", "major-chords"}
    for case in cases:
        name = case.get("name")
        if not isinstance(name, str) or not name.startswith("authored-") or re.fullmatch(r"[a-z0-9-]+", name) is None or name in names:
            raise RuntimeError("invalid or duplicate authored case name")
        names.add(name)
        if case.get("pattern") not in patterns:
            raise RuntimeError("unknown authored PCM pattern")
        for field in ["bpm", "startSeconds"]:
            value = case.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"invalid authored {field}")
        count = case.get("beatCount")
        if type(count) is not int or not 32 <= count <= 128 or not 55 <= case["bpm"] <= 220:
            raise RuntimeError("authored case exceeds bounded duration/tempo contract")
        if case["pattern"] == "drift" and (type(case.get("endBpm")) not in (int, float) or not 55 <= case["endBpm"] <= 220):
            raise RuntimeError("invalid authored drift endpoint")
        if case["pattern"] == "major-chords" and case.get("key") != "G major":
            raise RuntimeError("authored chord generator supports G major")
    return config


def beat_score(actual: list, reference: list, tolerance: float) -> float | None:
    if not actual and not reference:
        return None
    return goldens.beat_f1(actual, reference, tolerance) or 0.0


def tempo_metrics(actual: float | None, reference: float | None, tolerance: float) -> dict:
    if not actual or not reference:
        return dict(exactMatch=False if reference else None, familyMatch=False if reference else None,
                    exactErrorPercent=None, familyErrorPercent=None)
    exact_error = abs(actual - reference) / reference
    family_error = goldens.tempo_relative_error(actual, reference)
    return dict(exactMatch=exact_error <= tolerance, familyMatch=family_error <= tolerance,
                exactErrorPercent=100 * exact_error, familyErrorPercent=100 * family_error)


def tempo_interval_error(beats: list[float], truth: dict) -> dict:
    errors = []
    for first, second in zip(beats, beats[1:]):
        center = (first + second) / 2
        if truth["beats"][0] <= center < truth["beats"][-1] and second > first:
            index = bisect.bisect_right(truth["beats"], center) - 1
            errors.append(abs(60 / (second - first) - truth["tempoMap"][index]["bpm"]))
    return dict(meanAbsoluteErrorBpm=goldens.mean_present(errors), intervalCount=len(errors))


def compare(case: dict, rust: dict, mixxx: dict, config: dict) -> dict:
    rhythm = rust["rhythm"]
    rust_key = (rust.get("key") or {}).get("label")
    tolerance = config["beatToleranceSeconds"]
    metrics = {
        "tempoMixxx": tempo_metrics(rhythm.get("bpm"), mixxx["bpm"], config["tempoToleranceRelative"]),
        "beatF1Mixxx": beat_score(rhythm.get("beats", []), mixxx["beats"], tolerance),
        "keyMatchesMixxx": None if not rust_key or not mixxx["key"] else rust_key == mixxx["key"],
        "rustHasTempo": bool(rhythm.get("bpm")), "mixxxHasTempo": bool(mixxx["bpm"]),
        "rustBeatCount": len(rhythm.get("beats", [])), "mixxxBeatCount": len(mixxx["beats"]),
        "rustHasKey": bool(rust_key), "mixxxHasKey": bool(mixxx["key"]),
    }
    truth = case.get("groundTruth")
    if truth:
        metrics["authored"] = {
            "rustTempo": tempo_metrics(rhythm.get("bpm"), truth["bpm"], config["tempoToleranceRelative"]),
            "mixxxTempo": tempo_metrics(mixxx["bpm"], truth["bpm"], config["tempoToleranceRelative"]),
            "rustBeatF1": beat_score(rhythm.get("beats", []), truth["beats"], tolerance),
            "mixxxBeatF1": beat_score(mixxx["beats"], truth["beats"], tolerance),
            "rustKeyMatch": None if not truth["key"] else rust_key == truth["key"],
            "mixxxKeyMatch": None if not truth["key"] else mixxx["key"] == truth["key"],
            "rustTempoIntervals": tempo_interval_error(rhythm.get("beats", []), truth),
            "mixxxTempoIntervals": tempo_interval_error(mixxx["beats"], truth),
        }
    elif case.get("referenceBpm"):
        metrics["creatorTempo"] = {
            "rust": tempo_metrics(rhythm.get("bpm"), case["referenceBpm"], config["tempoToleranceRelative"]),
            "mixxx": tempo_metrics(mixxx["bpm"], case["referenceBpm"], config["tempoToleranceRelative"]),
        }
    return metrics


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for profile in sorted({row["profile"] for row in rows}):
        for kind in sorted({row["evidenceKind"] for row in rows}):
            selected = [row["metrics"] for row in rows if row["profile"] == profile and row["evidenceKind"] == kind]
            if not selected:
                continue
            eligible = [item for item in selected if item["rustHasTempo"] and item["mixxxHasTempo"]]
            key_values = [item["keyMatchesMixxx"] for item in selected if item["keyMatchesMixxx"] is not None]
            beat_values = [item["beatF1Mixxx"] for item in selected if item["beatF1Mixxx"] is not None]
            result = dict(profile=profile, evidenceKind=kind, caseCount=len(selected),
                          rustTempoMissing=sum(not item["rustHasTempo"] for item in selected),
                          mixxxTempoMissing=sum(not item["mixxxHasTempo"] for item in selected),
                          rustKeyMissing=sum(not item["rustHasKey"] for item in selected),
                          mixxxKeyMissing=sum(not item["mixxxHasKey"] for item in selected),
                          rustBeatsMissing=sum(not item["rustBeatCount"] for item in selected),
                          mixxxBeatsMissing=sum(not item["mixxxBeatCount"] for item in selected),
                          tempoComparableCount=len(eligible),
                          tempoExactMatches=sum(item["tempoMixxx"]["exactMatch"] for item in eligible),
                          tempoFamilyMatches=sum(item["tempoMixxx"]["familyMatch"] for item in eligible),
                          meanBeatF1Mixxx=goldens.mean_present(beat_values), beatComparableCount=len(beat_values),
                          keyMatches=sum(key_values), keyComparableCount=len(key_values))
            if kind == "synthetic-authored-ground-truth":
                result["meanRustBeatF1Authored"] = goldens.mean_present([item["authored"]["rustBeatF1"] for item in selected])
                result["meanMixxxBeatF1Authored"] = goldens.mean_present([item["authored"]["mixxxBeatF1"] for item in selected])
            summaries.append(result)
    return summaries


def capture_references(cases: list[dict], config: dict, image: str, output: pathlib.Path) -> dict:
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    # Resolve a tag once and execute only that immutable local image ID.
    image_id = inspected["Id"]
    directory = pathlib.Path(tempfile.mkdtemp(prefix="capture-", dir=output))
    job = dict(config=config, cases=[dict(name=case["name"], sha256=case["sha256"], containerPath="/repo/" + str(pathlib.Path(case["path"]).relative_to(ROOT))) for case in cases])
    (directory / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", f"{os.getuid()}:{os.getgid()}", "--cpus", "2", "--memory", "2g", "--pids-limit", "256",
        "--tmpfs", "/tmp:rw,nosuid,size=512m",
        "--mount", f"type=bind,source={ROOT},target=/repo,readonly",
        "--mount", f"type=bind,source={directory},target=/output",
        image_id, "python3", "/repo/scripts/evaluate-dj-mixxx.py", "--capture-job", "/output/job.json",
    ]
    subprocess.run(command, check=True)
    captured = json.loads((directory / "capture.json").read_text())
    captured["runtime"].update(imageId=image_id, declaredBaseImage=config["baseImage"],
                               mixxxUpstreamRevision=config["mixxxSourceRevision"], cpuLimit=2, memoryLimitBytes=2 * 1024 ** 3)
    fingerprint = hashlib.sha256(json.dumps(captured["runtime"], sort_keys=True).encode()).hexdigest()
    captured["runtime"]["fingerprint"] = "sha256:" + fingerprint
    return captured


def source_identity() -> dict:
    source_lock = goldens.sha256(ROOT / "Cargo.lock")
    expected = (ROOT / ".coding-tooling.source-lock.sha256").read_text().strip()
    if not (ROOT / ".cargo/config.toml").exists() or source_lock != expected:
        raise RuntimeError("activate the exact reviewed source graph before benchmarking")
    patches = json.loads((ROOT / ".coding-tooling.source-deps.json").read_text())["cargo"]["patches"]
    revisions = {item["rev"] for item in patches}
    if len(revisions) != 1:
        raise RuntimeError("benchmark requires one exact Foundation revision")
    for item in patches:
        checkout = ROOT / item["localPath"]
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True)
        if actual != item["rev"] or dirty:
            raise RuntimeError("benchmark Foundation checkout is not clean at its exact pin")
    library_diff = subprocess.check_output(["git", "diff", "HEAD", "--", "crates", "packages"], cwd=ROOT, text=True)
    harness = ["scripts/evaluate-dj-mixxx.py", "scripts/dj_mixxx_reference.py", "scripts/dj_benchmark_cases.py",
               "scripts/dj_mixxx_baseline.py", "scripts/evaluate-dj-goldens.py", "scripts/mixxx/Dockerfile"]
    return dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        libraryDiffSha256=hashlib.sha256(library_diff.encode()).hexdigest(), libraryDirty=bool(library_diff),
        pythonVersion=platform.python_version(), cargoLockSha256=source_lock, buildProfile="release",
        rustCompiler=subprocess.check_output(["rustc", "-vV"], text=True).strip(),
        foundationRevision=next(iter(revisions)),
        harnessSha256={path: goldens.sha256(ROOT / path) for path in harness},
        manifestSha256=goldens.sha256(MANIFEST), corpusManifestSha256=goldens.sha256(goldens.CORPUS_MANIFEST),
        baselineSha256=goldens.sha256(BASELINE) if BASELINE.exists() else None,
    )


def write_markdown(report: dict, path: pathlib.Path) -> None:
    lines = ["# Mixxx comparison benchmark", "", report["claimBoundary"], "",
             f"Audio source: `{report['audioSource']['revision']}`. Mixxx: `{report['runtime']['mixxxPackageVersion']}`.", "",
             "| Profile | Evidence | Cases | Exact BPM | BPM family | Mean beat F1 vs Mixxx | Key matches |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["aggregate"]:
        beat = item["meanBeatF1Mixxx"]
        lines.append(f"| {item['profile']} | {item['evidenceKind']} | {item['caseCount']} | {item['tempoExactMatches']}/{item['tempoComparableCount']} | {item['tempoFamilyMatches']}/{item['tempoComparableCount']} | {beat:.4f} | {item['keyMatches']}/{item['keyComparableCount']} |" if beat is not None else f"| {item['profile']} | {item['evidenceKind']} | {item['caseCount']} | — | — | unavailable | — |")
    lines += ["", "Missing outputs and repeat consistency are explicit in report.json. Authored truth is reported separately from model agreement.", "",
              "Authored drift uses the median instantaneous tempo as its scalar BPM convention. Mixxx may summarize a region differently; inspect beat F1 and interval error alongside that scalar.", "",
              "| Authored case | Profile | Rust beat F1 vs truth | Mixxx beat F1 vs truth | Rust interval MAE (BPM) | Mixxx interval MAE (BPM) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in report["comparisons"]:
        authored = row["metrics"].get("authored")
        if authored:
            lines.append(f"| {row['case']} | {row['profile']} | {authored['rustBeatF1']} | {authored['mixxxBeatF1']} | {authored['rustTempoIntervals']['meanAbsoluteErrorBpm']} | {authored['mixxxTempoIntervals']['meanAbsoluteErrorBpm']} |")
    lines += ["", f"Versioned accuracy baseline: **{report['baselineComparison']['status']}**. Model agreement is not an accuracy gate.", "",
              "| Case | Profile | Rust BPM | Mixxx BPM | Beat F1 vs Mixxx | Rust key | Mixxx key |", "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for row in report["comparisons"]:
        beat = row["metrics"]["beatF1Mixxx"]
        score = "unavailable" if beat is None else f"{beat:.4f}"
        lines.append(f"| {row['case']} | {row['profile']} | {row['rust']['rhythm'].get('bpm')} | {row['mixxx']['bpm']} | {score} | {(row['rust'].get('key') or {}).get('label')} | {row['mixxx']['key']} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="audio-analysis-mixxx:2.5.4")
    parser.add_argument("--report", type=pathlib.Path, default=ROOT / "target/dj-goldens/mixxx/report.json")
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--no-authored", action="store_true", help="focused diagnosis only; report marks incomplete coverage")
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--establish-baseline", type=pathlib.Path,
                        help="deliberately write a reviewed independent-oracle baseline from a complete run")
    parser.add_argument("--capture-job", type=pathlib.Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture_job:
        capture_job(args.capture_job)
        return 0
    config = load_config()
    corpus = goldens.load_corpus()
    if args.check_manifest:
        print(json.dumps(dict(corpusCases=len(corpus.fixtures), authoredCases=len(config["authoredCases"]), profiles=config["profiles"], repetitions=config["repetitions"]), indent=2))
        return 0
    source = source_identity()
    if not args.establish_baseline and not args.fixture and not args.no_authored and not BASELINE.exists():
        raise RuntimeError("review and establish the missing versioned accuracy baseline")
    report_path = args.report.resolve()
    output = report_path.parent
    output.mkdir(parents=True, exist_ok=True)
    # Inputs stay in the repository's ignored target cache so they can be mounted
    # read-only at one stable path in the isolated reference runtime.
    generated = ROOT / "target/dj-goldens/mixxx-authored"
    generated.mkdir(parents=True, exist_ok=True)
    cases = []
    for fixture in goldens.select_fixtures(corpus, args.fixture):
        path = goldens.download_fixture(corpus, fixture)
        cases.append(dict(name=fixture.name, path=str(path), sha256=fixture.sha256,
                          evidenceKind="checksum-pinned-corpus", category=fixture.category,
                          license=fixture.license, sourceUrl=goldens.fixture_source_url(corpus, fixture),
                          provenanceUrl=fixture.provenance_url, referenceBpm=fixture.reference_bpm,
                          referenceBpmSource=fixture.reference_bpm_source,
                          referenceKey=fixture.reference_key, referenceKeySource=fixture.reference_key_source))
    if not args.no_authored:
        for spec in config["authoredCases"]:
            case = generate_case(spec, generated)
            case["sha256"] = goldens.sha256(pathlib.Path(case["path"]))
            case["license"] = "repository-authored"
            cases.append(case)
    captured = capture_references(cases, config, args.image, output)
    indexed = {(item["case"], item["profile"], item["repetition"]): item["analysis"] for item in captured["captures"]}
    expected_count = len(cases) * len(config["profiles"]) * config["repetitions"]
    if len(indexed) != expected_count or len(captured["captures"]) != expected_count:
        raise RuntimeError("Mixxx capture is incomplete or contains duplicate records")
    rows = []
    failures = []
    for case in cases:
        print(f"Rust analysis: {case['name']}", file=sys.stderr, flush=True)
        rust = goldens.rust_analysis(pathlib.Path(case["path"]), release=True)
        rust_repeat = goldens.rust_analysis(pathlib.Path(case["path"]), release=True)
        if rust != rust_repeat:
            failures.append(f"{case['name']}: fresh Rust repeats differ")
        if goldens.sha256(pathlib.Path(case["path"])) != case["sha256"] or source_identity() != source:
            raise RuntimeError("benchmark input, harness, source revision, or dependency graph changed during analysis")
        for profile in config["profiles"]:
            first = indexed[(case["name"], profile["name"], 0)]
            repeat = indexed[(case["name"], profile["name"], 1)]
            if first != repeat:
                failures.append(f"{case['name']}/{profile['name']}: fresh Mixxx repeats differ")
            rows.append(dict(case=case["name"], profile=profile["name"], evidenceKind=case["evidenceKind"],
                             repeatsIdentical=first == repeat, rustRepeatsIdentical=rust == rust_repeat, rust=rust, mixxx=first,
                             metrics=compare(case, rust, first, config)))
    report = {
        "schemaVersion": "audio-analysis-dj-mixxx-evidence/v1",
        "claimBoundary": "Mixxx agreement is model comparison, not real-song ground truth or Mixxx parity. Authored PCM correctness is separate. No structure, waveform, downbeat, calibrated confidence, or speed claim is made.",
        "audioSource": source,
        "config": config, "runtime": captured["runtime"],
        "coverage": dict(fullCorpus=not args.fixture, authoredIncluded=not args.no_authored, corpus=goldens.coverage_summary(corpus)),
        "caseCount": len(cases), "referenceCaptureCount": len(captured["captures"]),
        "rustCaptureCount": 2 * len(cases),
        "cases": cases, "aggregate": summarize(rows), "comparisons": rows, "failures": failures,
    }
    if args.establish_baseline:
        report["baselineComparison"] = establish_baseline(report, args.establish_baseline)
    else:
        report["baselineComparison"] = compare_baseline(report, BASELINE)
        failures.extend(report["baselineComparison"]["regressions"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, report_path.with_suffix(".md"))
    print(json.dumps(dict(report=str(report_path), aggregate=report["aggregate"], failures=failures), indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
