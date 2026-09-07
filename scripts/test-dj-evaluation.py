#!/usr/bin/env python3
"""Deterministic contract tests for DJ accuracy scoring."""

from __future__ import annotations

import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dj_evaluation import aggregate_metrics, beat_alignment, tempo_comparison


class TempoComparisonTests(unittest.TestCase):
    def test_exact_tempo(self) -> None:
        comparison = tempo_comparison(120.0, 120.0)
        self.assertTrue(comparison["equivalent"])
        self.assertEqual(comparison["referenceMultiplier"], 1.0)
        self.assertAlmostEqual(comparison["relativeError"], 0.0)

    def test_half_time_is_explicit(self) -> None:
        comparison = tempo_comparison(60.0, 120.0)
        self.assertTrue(comparison["equivalent"])
        self.assertEqual(comparison["referenceMultiplier"], 0.5)

    def test_double_time_is_explicit(self) -> None:
        comparison = tempo_comparison(180.0, 90.0)
        self.assertTrue(comparison["equivalent"])
        self.assertEqual(comparison["referenceMultiplier"], 2.0)

    def test_wrong_tempo_is_not_hidden_by_equivalence(self) -> None:
        comparison = tempo_comparison(137.0, 120.0)
        self.assertFalse(comparison["equivalent"])


class BeatAlignmentTests(unittest.TestCase):
    def test_exact_grid_scores_one(self) -> None:
        metrics = beat_alignment([0.5, 1.0, 1.5], [0.5, 1.0, 1.5])
        self.assertEqual(metrics["matchedCount"], 3)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["meanAbsoluteErrorMs"], 0.0)

    def test_missing_beat_reduces_recall(self) -> None:
        metrics = beat_alignment([0.5, 1.5], [0.5, 1.0, 1.5])
        self.assertEqual(metrics["matchedCount"], 2)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 2.0 / 3.0)
        self.assertLess(metrics["f1"], 1.0)

    def test_tolerance_is_time_based(self) -> None:
        within = beat_alignment([0.55], [0.5])
        outside = beat_alignment([0.58], [0.5])
        self.assertEqual(within["matchedCount"], 1)
        self.assertEqual(outside["matchedCount"], 0)


class AggregateMetricTests(unittest.TestCase):
    def test_aggregate_keeps_analyzer_and_pinned_key_evidence_separate(self) -> None:
        reports = [
            {
                "category": "dance",
                "reference": {
                    "dominantKey": {"value": "G major", "kind": "pinned-analyzer"}
                },
                "rust": {"key": {"label": "G major"}},
                "metrics": {
                    "librosa": {
                        "tempo": {"equivalent": True, "relativeError": 0.01},
                        "beats": {"f1": 0.8, "meanAbsoluteErrorMs": 20.0},
                    },
                    "essentia": {
                        "tempo": {"equivalent": False, "relativeError": 0.1},
                        "beats": {"f1": 0.6, "meanAbsoluteErrorMs": 30.0},
                        "key": {"exactMatch": True},
                    },
                },
            }
        ]
        aggregate = aggregate_metrics(reports)
        self.assertEqual(aggregate["fixtureCount"], 1)
        self.assertEqual(aggregate["pinnedDominantKeyAccuracy"], 1.0)
        self.assertEqual(
            aggregate["analyzers"]["librosa"]["tempoEquivalentRate"], 1.0
        )
        self.assertEqual(
            aggregate["analyzers"]["essentia"]["tempoEquivalentRate"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
