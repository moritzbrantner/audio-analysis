#!/usr/bin/env python3
"""DJ oracle gate: compare audio-analysis with established music analyzers.

librosa and Essentia are the oracle, the way Rapier is for physics-engine and
MapLibre is for maps. Their BPM, beats, key and sections for every fixture of
the checksum-pinned real-music corpus are captured once and committed:

    python3 scripts/dj-oracle.py capture      # needs librosa==1.0.0, essentia

The gate needs neither analyzer. It runs the Rust whole-track analyzer on the
same bytes, scores it against both references and compares the failing checks
with the committed known-gap list:

    python3 scripts/dj-oracle.py gate [--base-ref origin/main]

A failing check missing from the list fails the gate (regression). A listed
gap that now passes also fails until it is removed, so the list only shrinks.
With --base-ref, gaps added relative to that revision fail too, except for
fixtures whose references are new or were recaptured. --write-gaps rewrites
the list from the current run (initialisation, or after a fix).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dj_goldens", ROOT / "scripts/evaluate-dj-goldens.py")
goldens = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goldens
SPEC.loader.exec_module(goldens)

REFERENCES = ROOT / "tests/fixtures/dj/oracle-references.v1.json"
KNOWN_GAPS = ROOT / "tests/fixtures/dj/oracle-known-gaps.v1.json"
REPORT_DIR = ROOT / "target/dj-goldens/oracle"
REFERENCES_SCHEMA = "audio-analysis-dj-oracle-references/v1"
GAPS_SCHEMA = "audio-analysis-dj-oracle-known-gaps/v1"
ORACLES = ("librosa", "essentia")
SAMPLE_RATE = 44_100

# Gate thresholds.
TEMPO_TOLERANCE = 0.035  # relative BPM error
BEAT_TOLERANCE_SECONDS = 0.07
BEAT_F1_MIN = 0.70
SECTION_TOLERANCE_SECONDS = 3.0  # MIREX structure boundary tolerance
SECTION_F_MIN = 0.50
# Section conventions for the oracles (neither has a parameter-free default).
LIBROSA_SECONDS_PER_SECTION = 30.0  # ~16 bars at 128 BPM
ESSENTIA_MIN_SECTION_SECONDS = 8.0  # matches the Rust minimum section length

# Krumhansl-Kessler key profiles, C-rooted.
KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def rounded(values: list[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def normalize_key(label: str | None) -> str | None:
    if not label:
        return None
    tonic, _, scale = label.partition(" ")
    return f"{ENHARMONIC.get(tonic, tonic)} {scale}"


# ---------------------------------------------------------------- capture


def librosa_reference(path: pathlib.Path) -> dict[str, Any]:
    import librosa
    import numpy as np

    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)

    mean = np.mean(chroma, axis=1)
    best = None
    for scale, profile in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            score = float(np.corrcoef(mean, np.roll(profile, tonic))[0, 1])
            if best is None or score > best[0]:
                best = (score, f"{PITCH_CLASSES[tonic]} {scale}")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features = np.vstack([librosa.util.normalize(chroma, axis=1), librosa.util.normalize(mfcc, axis=1)])
    if len(beat_frames) > 4:
        features = librosa.util.sync(features, beat_frames, aggregate=np.median)
        frame_times = librosa.frames_to_time(librosa.util.fix_frames(beat_frames, x_max=chroma.shape[1]), sr=sr)
    else:
        frame_times = librosa.frames_to_time(np.arange(features.shape[1]), sr=sr)
    k = max(2, min(features.shape[1], round(duration / LIBROSA_SECONDS_PER_SECTION)))
    bounds = librosa.segment.agglomerative(features, k)
    boundaries = sorted({float(frame_times[index]) for index in bounds if 0 < index < len(frame_times)})
    return {
        "bpm": round(float(np.asarray(tempo).reshape(-1)[0]), 3),
        "beats": rounded(librosa.frames_to_time(beat_frames, sr=sr)),
        "key": best[1],
        "sectionBoundaries": rounded([value for value in boundaries if 0.0 < value < duration]),
        "durationSeconds": round(duration, 3),
    }


def essentia_reference(path: pathlib.Path) -> dict[str, Any]:
    import essentia.standard as es
    import numpy as np

    audio = es.MonoLoader(filename=str(path), sampleRate=SAMPLE_RATE)()
    duration = len(audio) / SAMPLE_RATE
    bpm, ticks, _confidence, _estimates, _intervals = es.RhythmExtractor2013(method="multifeature")(audio)
    key, scale, _strength = es.KeyExtractor(sampleRate=SAMPLE_RATE)(audio)

    frame_size, hop = 2048, 1024
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc = es.MFCC(numberCoefficients=13)
    frames = [mfcc(spectrum(window(frame)))[1] for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop)]
    features = np.array(frames, dtype=np.float32).T.copy()
    frames_per_second = SAMPLE_RATE / hop
    min_length = int(ESSENTIA_MIN_SECTION_SECONDS * frames_per_second)
    segments = es.SBic(minLength=min_length, size1=3 * min_length, inc1=min_length, size2=2 * min_length, inc2=min_length // 3)(features)
    boundaries = [float(index) / frames_per_second for index in segments]
    return {
        "bpm": round(float(bpm), 3),
        "beats": rounded(list(ticks)),
        "key": normalize_key(f"{key} {scale}"),
        "sectionBoundaries": rounded([value for value in boundaries if 0.5 < value < duration - 0.5]),
        "durationSeconds": round(duration, 3),
    }


def capture(args: argparse.Namespace) -> int:
    import essentia
    import librosa

    corpus = goldens.load_corpus()
    fixtures = goldens.select_fixtures(corpus, args.fixture)
    existing = json.loads(REFERENCES.read_text())["fixtures"] if REFERENCES.exists() and args.fixture else {}
    payload = {
        "schemaVersion": REFERENCES_SCHEMA,
        "analyzers": {
            "librosa": {
                "version": librosa.__version__,
                "tempoAndBeats": "librosa.beat.beat_track",
                "key": "mean chroma_cqt correlated with Krumhansl-Kessler profiles",
                "sections": f"librosa.segment.agglomerative on beat-synced chroma+MFCC, k=round(duration/{LIBROSA_SECONDS_PER_SECTION:g}s)",
            },
            "essentia": {
                "version": essentia.__version__,
                "tempoAndBeats": "RhythmExtractor2013(method=multifeature)",
                "key": "KeyExtractor (default profile)",
                "sections": f"SBic on 13 MFCC, minLength={ESSENTIA_MIN_SECTION_SECONDS:g}s",
            },
        },
        "sampleRate": SAMPLE_RATE,
        "fixtures": existing,
    }
    for fixture in fixtures:
        path = goldens.download_fixture(corpus, fixture)
        print(f"capturing {fixture.name}", file=sys.stderr)
        payload["fixtures"][fixture.name] = {
            "sha256": fixture.sha256,
            "librosa": librosa_reference(path),
            "essentia": essentia_reference(path),
        }
    payload["fixtures"] = dict(sorted(payload["fixtures"].items()))
    text = json.dumps(payload, indent=1)
    # Keep numeric arrays (beats, boundaries) on one line each.
    text = re.sub(r"\[([-0-9.,\s]+)\]", lambda match: "[" + "".join(match.group(1).split()).replace(",", ", ") + "]", text)
    REFERENCES.write_text(text + "\n")
    print(f"wrote {REFERENCES.relative_to(ROOT)}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- scoring


def tempo_relation(actual: float | None, reference: float | None) -> str:
    if not actual or not reference:
        return "missing"
    for name, ratio in (("same", 1.0), ("half", 0.5), ("double", 2.0)):
        if abs(actual - reference * ratio) / max(actual, reference * ratio) <= TEMPO_TOLERANCE:
            return name
    return "none"


def boundary_f(actual: list[float], reference: list[float]) -> float:
    if not actual and not reference:
        return 1.0
    if not actual or not reference:
        return 0.0
    used: set[int] = set()
    hits = 0
    for value in actual:
        candidates = [
            (abs(value - ref), index) for index, ref in enumerate(reference)
            if index not in used and abs(value - ref) <= SECTION_TOLERANCE_SECONDS
        ]
        if candidates:
            used.add(min(candidates)[1])
            hits += 1
    precision, recall = hits / len(actual), hits / len(reference)
    return 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)


def rust_view(rust: dict[str, Any]) -> dict[str, Any]:
    rhythm = rust.get("rhythm", {})
    duration = float(rust.get("durationSeconds", 0.0))
    sections = rust.get("sections") or []
    return {
        "bpm": rhythm.get("bpm"),
        "beats": [float(value) for value in rhythm.get("beats", [])],
        "key": normalize_key((rust.get("key") or {}).get("label")),
        "sectionBoundaries": [
            float(section["startSeconds"]) for section in sections
            if 0.5 < float(section["startSeconds"]) < duration - 0.5
        ],
    }


def score_pair(actual: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    relation = tempo_relation(actual["bpm"], reference["bpm"])
    beat = goldens.beat_f1(actual["beats"], reference["beats"], BEAT_TOLERANCE_SECONDS)
    return {
        "tempoRelation": relation,
        "beatF1": None if beat is None else round(beat, 4),
        "keyMatch": actual["key"] is not None and actual["key"] == reference["key"],
        "sectionF": round(boundary_f(actual["sectionBoundaries"], reference["sectionBoundaries"]), 4),
    }


def failing_checks(oracle: str, score: dict[str, Any]) -> list[str]:
    checks = {
        "tempo-family": score["tempoRelation"] in {"same", "half", "double"},
        "tempo-octave": score["tempoRelation"] == "same",
        "beats": (score["beatF1"] or 0.0) >= BEAT_F1_MIN,
        "key": score["keyMatch"],
        "sections": score["sectionF"] >= SECTION_F_MIN,
    }
    return [f"{name}:{oracle}" for name, passed in checks.items() if not passed]


def creator_checks(fixture: Any, actual: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if fixture.reference_bpm is not None:
        checks["tempo-octave:creator"] = tempo_relation(actual["bpm"], fixture.reference_bpm) == "same"
    if fixture.reference_key and (fixture.reference_key_source or "").startswith("Wikimedia"):
        checks["key:creator"] = actual["key"] == normalize_key(fixture.reference_key)
    return checks


# ---------------------------------------------------------------- gate


def load_gaps(text: str) -> set[str]:
    payload = json.loads(text)
    if payload.get("schemaVersion") != GAPS_SCHEMA:
        raise RuntimeError("unsupported DJ oracle known-gap schema")
    gaps = payload.get("gaps")
    if not isinstance(gaps, list) or gaps != sorted(set(gaps)):
        raise RuntimeError("known gaps must be a sorted list without duplicates")
    return set(gaps)


def gap_verdict(
    current: set[str],
    known: set[str],
    known_in_scope: set[str],
    references: dict[str, Any],
    base: tuple[set[str], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Compare failing checks with the list; the list may only shrink against base.

    A gap absent from base is accepted only for a fixture whose references are
    new or were recaptured, since only then is it a new measurement.
    """
    added: list[str] = []
    if base is not None:
        base_gaps, base_refs = base
        added = sorted(
            gap for gap in known - base_gaps
            if base_refs.get(gap.split("/", 1)[0]) == references.get(gap.split("/", 1)[0])
        )
    return {
        "regressions": sorted(current - known_in_scope),
        "fixedButListed": sorted(known_in_scope - current),
        "addedVersusBase": added,
        "knownGapCount": len(known),
    }


def git_show(ref: str, path: pathlib.Path) -> str | None:
    completed = subprocess.run(
        ["git", "show", f"{ref}:{path.relative_to(ROOT)}"], cwd=ROOT, capture_output=True, text=True,
    )
    return completed.stdout if completed.returncode == 0 else None


def build_rust() -> pathlib.Path:
    subprocess.run(
        ["cargo", "build", "--locked", "--release", "--quiet", "-p", "moenarch-audio-analysis-rhythm", "--example", "dj_analyze"],
        cwd=ROOT, check=True,
    )
    return ROOT / "target/release/examples/dj_analyze"


def fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if value != int(value) or abs(value) < 1000 else f"{value:.0f}"
    return str(value)


def write_report(rows: list[dict[str, Any]], aggregate: dict[str, Any], verdict: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.json").write_text(json.dumps({"aggregate": aggregate, "verdict": verdict, "fixtures": rows}, indent=1) + "\n")
    lines = [
        "# DJ oracle gate",
        "",
        f"Oracles: librosa / Essentia. Tempo tolerance {TEMPO_TOLERANCE:.1%} (half/double reported, not hidden); "
        f"beat F1 ≥ {BEAT_F1_MIN} at ±{BEAT_TOLERANCE_SECONDS * 1000:.0f} ms; key exact; "
        f"section boundary F ≥ {SECTION_F_MIN} at ±{SECTION_TOLERANCE_SECONDS:g} s.",
        "",
        "| Fixture | Coverage | BPM rust / lib / ess | Tempo vs lib / ess | Beat F1 lib / ess | Key rust / lib / ess | Section F lib / ess | Oracles agree (tempo, key) | Gaps |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lib, ess = row["scores"]["librosa"], row["scores"]["essentia"]
        refs = row["references"]
        lines.append(
            f"| {row['fixture']} | {', '.join(row['coverage']) or row['category']} "
            f"| {fmt(row['rust']['bpm'])} / {fmt(refs['librosa']['bpm'])} / {fmt(refs['essentia']['bpm'])} "
            f"| {lib['tempoRelation']} / {ess['tempoRelation']} "
            f"| {fmt(lib['beatF1'])} / {fmt(ess['beatF1'])} "
            f"| {row['rust']['key'] or '–'} / {refs['librosa']['key']} / {refs['essentia']['key']} "
            f"| {fmt(lib['sectionF'])} / {fmt(ess['sectionF'])} "
            f"| {row['oracleAgreement']['tempo']}, {'yes' if row['oracleAgreement']['key'] else 'no'} "
            f"| {len(row['gaps'])} |"
        )
    lines += ["", "## Aggregate", "", "```json", json.dumps(aggregate, indent=1), "```", "", "## Verdict", ""]
    for name in ("regressions", "fixedButListed", "addedVersusBase"):
        lines.append(f"- {name}: {', '.join(verdict[name]) or 'none'}")
    lines.append(f"- known gaps: {verdict['knownGapCount']}")
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def gate(args: argparse.Namespace) -> int:
    corpus = goldens.load_corpus()
    references = json.loads(REFERENCES.read_text())
    if references.get("schemaVersion") != REFERENCES_SCHEMA:
        raise RuntimeError("unsupported DJ oracle reference schema")
    fixtures = goldens.select_fixtures(corpus, args.fixture)
    missing = [fixture.name for fixture in fixtures if fixture.name not in references["fixtures"]]
    if missing:
        raise RuntimeError(f"fixtures without oracle references (run capture): {', '.join(missing)}")
    binary = build_rust()

    rows: list[dict[str, Any]] = []
    current: set[str] = set()
    for fixture in fixtures:
        reference = references["fixtures"][fixture.name]
        if reference["sha256"] != fixture.sha256:
            raise RuntimeError(f"{fixture.name}: references were captured from different bytes; recapture")
        path = goldens.download_fixture(corpus, fixture)
        print(f"analyzing {fixture.name}", file=sys.stderr)
        completed = subprocess.run([str(binary), str(path)], cwd=ROOT, capture_output=True, text=True, check=True)
        actual = rust_view(json.loads(completed.stdout))
        scores = {oracle: score_pair(actual, reference[oracle]) for oracle in ORACLES}
        gaps = [check for oracle in ORACLES for check in failing_checks(oracle, scores[oracle])]
        creator = creator_checks(fixture, actual)
        gaps += [name for name, passed in creator.items() if not passed]
        gaps = sorted(f"{fixture.name}/{check}" for check in gaps)
        current.update(gaps)
        rows.append({
            "fixture": fixture.name,
            "category": fixture.category,
            "coverage": list(fixture.coverage),
            "rust": {"bpm": actual["bpm"], "key": actual["key"], "sectionBoundaries": rounded(actual["sectionBoundaries"])},
            "references": {
                oracle: {"bpm": reference[oracle]["bpm"], "key": reference[oracle]["key"],
                         "sectionBoundaries": reference[oracle]["sectionBoundaries"]}
                for oracle in ORACLES
            },
            "scores": scores,
            "creator": creator,
            "oracleAgreement": {
                "tempo": tempo_relation(reference["librosa"]["bpm"], reference["essentia"]["bpm"]),
                "beatF1": goldens.beat_f1(reference["librosa"]["beats"], reference["essentia"]["beats"], BEAT_TOLERANCE_SECONDS),
                "key": reference["librosa"]["key"] == reference["essentia"]["key"],
                "sectionF": round(boundary_f(reference["librosa"]["sectionBoundaries"], reference["essentia"]["sectionBoundaries"]), 4),
            },
            "gaps": gaps,
        })

    def mean(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return round(statistics.fmean(present), 4) if present else None

    aggregate: dict[str, Any] = {"fixtureCount": len(rows), "gapCount": len(current)}
    for oracle in ORACLES:
        scores = [row["scores"][oracle] for row in rows]
        aggregate[oracle] = {
            "tempoSame": sum(score["tempoRelation"] == "same" for score in scores),
            "tempoHalfOrDouble": sum(score["tempoRelation"] in {"half", "double"} for score in scores),
            "meanBeatF1": mean([score["beatF1"] for score in scores]),
            "keyMatches": sum(score["keyMatch"] for score in scores),
            "meanSectionF": mean([score["sectionF"] for score in scores]),
        }
    aggregate["oracleSelfAgreement"] = {
        "tempoSame": sum(row["oracleAgreement"]["tempo"] == "same" for row in rows),
        "keyMatches": sum(row["oracleAgreement"]["key"] for row in rows),
        "meanBeatF1": mean([row["oracleAgreement"]["beatF1"] for row in rows]),
        "meanSectionF": mean([row["oracleAgreement"]["sectionF"] for row in rows]),
    }

    selected = {fixture.name for fixture in fixtures}
    known = load_gaps(KNOWN_GAPS.read_text()) if KNOWN_GAPS.exists() else set()
    known_in_scope = {gap for gap in known if gap.split("/", 1)[0] in selected}
    if args.write_gaps:
        kept = {gap for gap in known if gap.split("/", 1)[0] not in selected}
        known = kept | current
        KNOWN_GAPS.write_text(json.dumps({"schemaVersion": GAPS_SCHEMA, "gaps": sorted(known)}, indent=1) + "\n")
        known_in_scope = current
    base = None
    if args.base_ref:
        base_gaps_text = git_show(args.base_ref, KNOWN_GAPS)
        base_refs_text = git_show(args.base_ref, REFERENCES)
        if base_gaps_text is not None and base_refs_text is not None:
            base = (load_gaps(base_gaps_text), json.loads(base_refs_text).get("fixtures", {}))
    verdict = gap_verdict(current, known, known_in_scope, references["fixtures"], base)
    added = verdict["addedVersusBase"]
    write_report(rows, aggregate, verdict)
    print((REPORT_DIR / "report.md").read_text())
    if verdict["regressions"]:
        print("FAIL: new gaps against the oracle (fix the regression; do not list it)", file=sys.stderr)
    if verdict["fixedButListed"]:
        print("FAIL: listed gaps now pass; remove them (python3 scripts/dj-oracle.py gate --write-gaps)", file=sys.stderr)
    if added:
        print(f"FAIL: known-gap list grew against {args.base_ref}; it may only shrink", file=sys.stderr)
    return 1 if verdict["regressions"] or verdict["fixedButListed"] or added else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture", help="record librosa/Essentia references")
    capture_parser.add_argument("--fixture", action="append", default=[])
    gate_parser = sub.add_parser("gate", help="score audio-analysis against the references")
    gate_parser.add_argument("--fixture", action="append", default=[])
    gate_parser.add_argument("--base-ref", help="git revision whose known-gap list may not grow")
    gate_parser.add_argument("--write-gaps", action="store_true", help="rewrite the known-gap list from this run")
    args = parser.parse_args()
    return capture(args) if args.command == "capture" else gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
