"""Offline checks for the DJ oracle gate's scoring and shrink-only gap list."""

import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("dj_oracle", ROOT / "scripts" / "dj-oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


class OracleGateTest(unittest.TestCase):
    def test_beat_window_diagnostics_separate_stable_tempo_from_wrong_phase(self):
        actual = {
            "durationSeconds": 4.0,
            "beats": [0.5, 1.0, 1.5, 2.0, 2.5],
            "beatStrengths": [0.8] * 5,
            "tempoMap": [(beat, 120.0) for beat in [0.5, 1.0, 1.5, 2.0, 2.5]],
        }
        reference = {
            "librosa": {"beats": [0.5, 1.0, 1.5, 2.0, 2.5]},
            "essentia": {"beats": [0.75, 1.25, 1.75, 2.25, 2.75]},
        }
        windows = oracle.beat_window_diagnostics(actual, reference, width=2.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["rustBeatCount"], 3)
        self.assertEqual(windows[0]["localBpmRange"], [120.0, 120.0])
        self.assertEqual(windows[0]["medianRustBeatStrength"], 0.8)
        self.assertEqual(windows[0]["references"]["librosa"]["beatF1"], 1.0)
        self.assertEqual(windows[0]["references"]["librosa"]["medianNearestBeatErrorMs"], 0.0)
        self.assertEqual(windows[0]["references"]["essentia"]["beatF1"], 0.0)
        self.assertEqual(windows[0]["references"]["essentia"]["medianNearestBeatErrorMs"], 250.0)
        with self.assertRaisesRegex(ValueError, "same length"):
            oracle.beat_window_diagnostics({**actual, "beatStrengths": []}, reference)

    def test_tempo_relation_keeps_half_and_double_explicit(self):
        self.assertEqual(oracle.tempo_relation(124.0, 124.5), "same")
        self.assertEqual(oracle.tempo_relation(62.0, 124.0), "half")
        self.assertEqual(oracle.tempo_relation(248.0, 124.0), "double")
        self.assertEqual(oracle.tempo_relation(100.0, 124.0), "none")
        self.assertEqual(oracle.tempo_relation(None, 124.0), "missing")

    def test_boundary_f_uses_tolerance_and_one_to_one_matching(self):
        self.assertEqual(oracle.boundary_f([], []), 1.0)
        self.assertEqual(oracle.boundary_f([], [30.0]), 0.0)
        self.assertEqual(oracle.boundary_f([31.0, 90.0], [30.0, 60.0]), 0.5)
        self.assertEqual(oracle.boundary_f([30.0, 31.0], [30.5]), 2 / 3)

    def test_gap_list_only_shrinks(self):
        refs = {"a": {"sha256": "1"}, "b": {"sha256": "2"}}
        verdict = oracle.gap_verdict({"a/key:librosa"}, {"a/key:librosa", "a/beats:essentia"},
                                     {"a/key:librosa", "a/beats:essentia"}, refs, None)
        self.assertEqual(verdict["fixedButListed"], ["a/beats:essentia"])
        verdict = oracle.gap_verdict({"a/key:librosa", "a/sections:librosa"}, {"a/key:librosa"},
                                     {"a/key:librosa"}, refs, None)
        self.assertEqual(verdict["regressions"], ["a/sections:librosa"])
        base = ({"a/key:librosa"}, {"a": {"sha256": "1"}})
        verdict = oracle.gap_verdict(set(), {"a/key:librosa", "a/beats:librosa", "b/key:librosa"},
                                     set(), refs, base)
        self.assertEqual(verdict["addedVersusBase"], ["a/beats:librosa"])

    def test_committed_references_cover_corpus_and_gaps_are_sorted(self):
        corpus = json.loads((ROOT / "tests/fixtures/dj/real-music-corpus.v1.json").read_text())
        references = json.loads(oracle.REFERENCES.read_text())
        self.assertEqual(references["schemaVersion"], oracle.REFERENCES_SCHEMA)
        for fixture in corpus["fixtures"]:
            entry = references["fixtures"][fixture["name"]]
            self.assertEqual(entry["sha256"], fixture["sha256"])
            for name in oracle.ORACLES:
                self.assertTrue({"bpm", "beats", "key", "sectionBoundaries"} <= entry[name].keys())
        gaps = oracle.load_gaps(oracle.KNOWN_GAPS.read_text())
        self.assertTrue(all(gap.split("/", 1)[0] in references["fixtures"] for gap in gaps))


if __name__ == "__main__":
    unittest.main()
