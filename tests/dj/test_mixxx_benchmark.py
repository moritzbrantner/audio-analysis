"""Offline regressions for the Mixxx evidence adapter and authored oracles."""

import importlib.util
import copy
import json
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import time
import os
import wave
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from dj_benchmark_cases import generate_case
from dj_mixxx_reference import decode_beats, read_database, ready_to_flush, desktop_key
from dj_mixxx_baseline import establish_baseline, compare_baseline

SPEC = importlib.util.spec_from_file_location("dj_mixxx_benchmark", ROOT / "scripts/evaluate-dj-mixxx.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)

# Captured from actual Mixxx 2.5.4 whole-track Queen Mary analysis of Choice.
# BPM 136, first beat at frame 228, mono/stereo frame positions are not samples.
CAPTURED_GRID = bytes.fromhex("0a09090000000000006140120308e401")


class CaptureLifecycleTests(unittest.TestCase):
    def test_cargo_timeout_closes_descendant_pipes(self):
        with tempfile.TemporaryDirectory() as temporary:
            cargo = pathlib.Path(temporary) / "cargo"
            cargo.write_text(f"#!{sys.executable}\nimport subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\ntime.sleep(30)\n")
            cargo.chmod(0o755)
            start = time.monotonic()
            with patch.dict(os.environ, {"PATH": temporary + os.pathsep + os.environ["PATH"]}):
                with self.assertRaises(subprocess.TimeoutExpired):
                    benchmark.goldens.rust_analysis(pathlib.Path("unused.wav"), timeout=0.2)
            self.assertLess(time.monotonic() - start, 5)

    def test_disappeared_dialog_never_receives_a_key(self):
        result = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"BadWindow")
        with patch("dj_mixxx_reference.subprocess.run", return_value=result) as run:
            self.assertFalse(desktop_key("stale-window", "Return", {}))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_other_focus_failures_propagate(self):
        result = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"unavailable display")
        with patch("dj_mixxx_reference.subprocess.run", return_value=result):
            with self.assertRaises(subprocess.CalledProcessError):
                desktop_key("window", "Return", {})

    def test_stalled_focus_propagates_without_sending_a_key(self):
        with patch("dj_mixxx_reference.subprocess.run", side_effect=subprocess.TimeoutExpired("xdotool", 5)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                desktop_key("window", "Return", {})
            self.assertEqual(run.call_count, 1)

    def test_container_is_stopped_on_timeout_and_cancellation(self):
        for failure in [subprocess.TimeoutExpired("docker", 1800), KeyboardInterrupt()]:
            with self.subTest(failure=type(failure)), patch.object(benchmark.subprocess, "run", side_effect=[failure, subprocess.CompletedProcess([], 0)]) as run:
                with self.assertRaises(type(failure)):
                    benchmark.run_capture(["docker", "run"], "audio-mixxx-owned")
                self.assertEqual(run.call_args_list[0].kwargs["timeout"], 1800)
                self.assertEqual(run.call_args_list[1].args[0], ["docker", "stop", "--time", "5", "audio-mixxx-owned"])
                self.assertEqual(run.call_args_list[1].kwargs["timeout"], 15)


class MixxxWireTests(unittest.TestCase):
    def test_constant_key_capture_can_finish_without_a_key_histogram(self):
        # Actual G-major authored capture. KeyUtils' singleton-key path logs
        # only the key name; waiting for a histogram hung this valid analysis.
        output = 'debug [AnalyzerThread 1 #6] AnalyzerBeats plugin detected 32 beats. Predominant BPM: Bpm(128)\n'
        output += 'debug [AnalyzerThread 1 #6] "G"\n'
        self.assertTrue(ready_to_flush(output, 1))
        self.assertFalse(ready_to_flush(output, 2))

    def test_real_mixxx_grid_uses_frames_and_extends_to_the_track_horizon(self):
        beats = decode_beats(CAPTURED_GRID, "BeatGrid-2.0", 44100, 1.0)
        self.assertEqual(len(beats), 3)
        self.assertAlmostEqual(beats[0], 228 / 44100)
        self.assertAlmostEqual(beats[1] - beats[0], 60 / 136)
        self.assertTrue(all(0 <= beat < 1 for beat in beats))

    def test_variable_map_preserves_nonuniform_positions(self):
        # BeatMap -> Beat.frame_position: 0, 100, 150, 250.
        blob = bytes.fromhex("0a0208000a0208640a030896010a0308fa01")
        self.assertEqual(decode_beats(blob, "BeatMap-1.0", 100, 3), [0, 1, 1.5, 2.5])

    def test_grid_with_negative_anchor_does_not_publish_negative_beats(self):
        # -100 in protobuf int32's sign-extended varint format.
        blob = bytes.fromhex("0a09090000000000005e40120b089cffffffffffffffff01")
        self.assertEqual(decode_beats(blob, "BeatGrid-2.0", 100, 1), [0, 0.5])

    def test_corrupt_wire_data_and_unknown_versions_fail_closed(self):
        for blob, version in [(CAPTURED_GRID[:-1], "BeatGrid-2.0"), (b"\x80", "BeatGrid-2.0"),
                              (b"\x00", "BeatGrid-2.0"), (CAPTURED_GRID, "BeatGrid-1.0"),
                              (bytes.fromhex("0a0208640a020800"), "BeatMap-1.0")]:
            with self.subTest(blob=blob, version=version), self.assertRaises(RuntimeError):
                decode_beats(blob, version, 44100, 1)

    def test_metadata_and_disabled_beats_are_not_analyzer_evidence(self):
        for extra in ["1000", "1801", "1802"]:
            blob = bytes.fromhex("0a09090000000000006140") + b"\x12" + bytes([3 + len(bytes.fromhex(extra))]) + bytes.fromhex("08e401" + extra)
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                decode_beats(blob, "BeatGrid-2.0", 44100, 1)

    def test_database_adapter_rejects_imported_keys_and_unrequested_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "mixxxdb.sqlite"
            with sqlite3.connect(path) as connection:
                connection.executescript("""
                    CREATE TABLE track_locations(id INTEGER, location TEXT);
                    CREATE TABLE library(location INTEGER, bpm REAL, duration REAL, samplerate INTEGER,
                        key_id INTEGER, beats BLOB, beats_version TEXT, beats_sub_version TEXT,
                        keys_version TEXT, keys_sub_version TEXT);
                    INSERT INTO track_locations VALUES(1, '/audio/choice.ogg');
                """)
                connection.execute("INSERT INTO library VALUES(1, 136, 1, 44100, 1, ?, 'BeatGrid-2.0', ?, 'KeyMap-1.0', ?)",
                                   (CAPTURED_GRID, "rounding=V4|vamp_plugin_id=qm-tempotracker:0", "vamp_plugin_id=qm-keydetector:2"))
            result = read_database(path, ["/audio/choice.ogg"])[0]
            self.assertEqual(result["key"], "C major")
            self.assertEqual(result["bpm"], 136)
            with self.assertRaisesRegex(RuntimeError, "requested tracks"):
                read_database(path, ["/audio/unrequested.ogg"])
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE library SET keys_sub_version = ''")
            with self.assertRaisesRegex(RuntimeError, "Queen Mary key"):
                read_database(path, ["/audio/choice.ogg"])


class MixxxBenchmarkTests(unittest.TestCase):
    def test_manifest_requires_both_profiles_and_fresh_repeats(self):
        original = benchmark.load_config()
        for change in [dict(repetitions=1), dict(profiles=original["profiles"][:1]), dict(mixxxSourceRevision="a" * 40)]:
            with tempfile.TemporaryDirectory() as temporary:
                path = pathlib.Path(temporary) / "manifest.json"
                path.write_text(json.dumps(dict(original, **change)))
                with patch.object(benchmark, "MANIFEST", path), self.assertRaises(RuntimeError):
                    benchmark.load_config()

    def test_authored_pcm_is_reproducible_and_breakdown_keeps_the_grid(self):
        spec = dict(name="authored-sparse", pattern="sparse", bpm=120, beatCount=48, startSeconds=0.5)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            case = generate_case(spec, directory)
            first = pathlib.Path(case["path"]).read_bytes()
            self.assertEqual(first, pathlib.Path(generate_case(spec, directory)["path"]).read_bytes())
            self.assertEqual(case["groundTruth"]["beats"][16:24], [8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12])
            with wave.open(case["path"]) as audio:
                self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 24000))
                audio.setpos(round(8.5 * 24000))
                self.assertEqual(audio.readframes(4 * 24000), b"\0" * (4 * 24000 * 2))
                audio.setpos(12000)
                self.assertNotEqual(audio.readframes(2400), b"\0" * 4800)

    def test_missing_beats_score_zero_against_authored_truth(self):
        self.assertEqual(benchmark.beat_score([], [0.5, 1], 0.07), 0)
        self.assertEqual(benchmark.beat_score([0.5, 1], [], 0.07), 0)
        self.assertIsNone(benchmark.beat_score([], [], 0.07))

    def test_half_time_agreement_does_not_become_exact_bpm_or_perfect_beat_f1(self):
        result = benchmark.tempo_metrics(60, 120, 0.035)
        self.assertFalse(result["exactMatch"])
        self.assertTrue(result["familyMatch"])
        self.assertAlmostEqual(benchmark.beat_score([0, 1], [0, 0.5, 1, 1.5], 0.07), 2 / 3)

    def test_missing_model_outputs_are_counted_in_aggregate(self):
        rows = []
        for name, bpm in [("observed", 120), ("missing", None)]:
            rust = dict(rhythm=dict(bpm=bpm, beats=[0, 0.5] if bpm else []), key=None)
            mixxx = dict(bpm=120, beats=[0, 0.5], key="C major")
            rows.append(dict(case=name, profile="constant", evidenceKind="checksum-pinned-corpus",
                             metrics=benchmark.compare({}, rust, mixxx, benchmark.load_config())))
        summary = benchmark.summarize(rows)[0]
        self.assertEqual(summary["caseCount"], 2)
        self.assertEqual(summary["rustTempoMissing"], 1)
        self.assertEqual(summary["tempoComparableCount"], 1)
        self.assertEqual(summary["meanBeatF1Mixxx"], 0.5)
        self.assertEqual(summary["keyComparableCount"], 0)


class AccuracyBaselineTests(unittest.TestCase):
    def report(self):
        truth = dict(bpm=120, beats=[0.5, 1.0], key="G major", tempoMap=[dict(bpm=120), dict(bpm=120)])
        rust = dict(rhythm=dict(bpm=120, beats=[0.5, 1.0]), key=dict(label="G major"))
        mixxx = dict(bpm=120, beats=[0.5, 1.0], key="G major")
        case = dict(name="authored-reference", sha256="a" * 64, groundTruth=truth)
        row = dict(case=case["name"], profile="whole-track-constant", metrics=benchmark.compare(case, rust, mixxx, benchmark.load_config()))
        source = {field: "pinned" for field in ["cargoLockSha256", "buildProfile", "rustCompiler", "harnessSha256", "manifestSha256", "corpusManifestSha256"]}
        source["revision"] = "b" * 40
        runtime = dict(mixxxBinarySha256="c" * 64, packagesSha256="d" * 64, architecture="amd64")
        return dict(coverage=dict(fullCorpus=True, authoredIncluded=True), failures=[],
                    cases=[case], comparisons=[row], audioSource=source, runtime=runtime, aggregate=[])

    def test_independent_regression_fails_and_model_agreement_does_not_become_truth(self):
        original = self.report()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "baseline.json"
            establish_baseline(original, path)
            altered = copy.deepcopy(original)
            altered["comparisons"][0]["metrics"]["beatF1Mixxx"] = 0
            self.assertEqual(compare_baseline(altered, path)["status"], "passed")
            altered["comparisons"][0]["metrics"]["authored"]["rustBeatF1"] = 0.8
            self.assertEqual(compare_baseline(altered, path)["status"], "regression")
            altered["comparisons"][0]["metrics"]["authored"]["rustBeatF1"] = 0.97
            self.assertEqual(compare_baseline(altered, path)["status"], "passed")

    def test_established_key_match_and_tempo_interval_error_are_protected(self):
        original = self.report()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "baseline.json"
            establish_baseline(original, path)
            altered = copy.deepcopy(original)
            altered["comparisons"][0]["metrics"]["authored"]["rustKeyMatch"] = False
            self.assertEqual(len(compare_baseline(altered, path)["regressions"]), 1)
            altered = copy.deepcopy(original)
            altered["comparisons"][0]["metrics"]["authored"]["rustTempoIntervals"]["meanAbsoluteErrorBpm"] = 2
            self.assertEqual(len(compare_baseline(altered, path)["regressions"]), 1)

    def test_changed_inputs_and_nonfinite_noise_cannot_hide_regressions(self):
        original = self.report()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "baseline.json"
            establish_baseline(original, path)
            changed = copy.deepcopy(original)
            changed["cases"][0]["sha256"] = "e" * 64
            with self.assertRaisesRegex(RuntimeError, "inputs differ"):
                compare_baseline(changed, path)
            baseline = json.loads(path.read_text())
            baseline["noiseThresholds"]["beatF1Absolute"] = float("nan")
            path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(RuntimeError, "finite"):
                compare_baseline(original, path)

    def test_focused_capture_neither_establishes_nor_applies_a_full_baseline(self):
        report = self.report()
        report["coverage"]["fullCorpus"] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "absent.json"
            with self.assertRaisesRegex(RuntimeError, "complete"):
                establish_baseline(report, path)
            self.assertEqual(compare_baseline(report, path)["status"], "focused-no-accuracy-gate")


if __name__ == "__main__":
    unittest.main()
