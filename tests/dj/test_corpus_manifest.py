"""Offline regressions for the real-music corpus trust boundary."""

import importlib.util
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dj_goldens", ROOT / "scripts" / "evaluate-dj-goldens.py"
)
goldens = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goldens
SPEC.loader.exec_module(goldens)


def manifest():
    return {
        "schemaVersion": "audio-analysis-dj-corpus/v1",
        "source": {"repository": "fixture/corpus", "revision": "a" * 40, "audioRoot": "audio"},
        "requiredCoverage": ["electronic-dance"],
        "fixtures": [{
            "name": "steady-track",
            "filename": "steady.ogg",
            "sha256": "a" * 64,
            "license": "CC0-1.0",
            "category": "electronic-dance",
            "coverage": ["electronic-dance"],
            "focus": ["steady-tempo"],
            "sourceUrl": "https://upload.wikimedia.org/wikipedia/commons/a/ab/steady.ogg",
            "provenanceUrl": "https://commons.wikimedia.org/w/index.php?title=File:steady.ogg&oldid=123",
            "sourceBytes": 100,
            "sourceSha1": "b" * 40,
        }],
    }


class CorpusManifestTests(unittest.TestCase):
    def load(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "corpus.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(goldens, "CORPUS_MANIFEST", path):
                return goldens.load_corpus()

    def test_pinned_external_source_is_accepted(self):
        fixture = self.load(manifest()).fixtures[0]
        self.assertEqual(fixture.source_bytes, 100)

    def test_commons_provenance_requires_an_immutable_file_revision(self):
        for url in [
            "https://commons.wikimedia.org/wiki/File:steady.ogg",
            "https://commons.wikimedia.org/w/index.php?title=File:steady.ogg&oldid=latest",
            "https://commons.wikimedia.org/w/index.php?title=File:steady.ogg&oldid=0",
            "https://commons.wikimedia.org/w/index.php?title=File:steady.ogg&oldid=123&oldid=456",
        ]:
            with self.subTest(url=url):
                payload = manifest()
                payload["fixtures"][0]["provenanceUrl"] = url
                with self.assertRaisesRegex(RuntimeError, "revision"):
                    self.load(payload)

    def test_external_byte_identity_requires_every_component(self):
        for field in ["sourceUrl", "provenanceUrl", "sourceBytes", "sourceSha1"]:
            with self.subTest(field=field):
                payload = manifest()
                del payload["fixtures"][0][field]
                with self.assertRaises(RuntimeError):
                    self.load(payload)

    def test_numeric_tempo_reference_requires_a_source_and_finite_positive_bpm(self):
        for bpm in [None, True, 0, -120, "120", float("nan"), float("inf")]:
            with self.subTest(bpm=bpm):
                payload = manifest()
                payload["fixtures"][0].update(referenceBpm=bpm, referenceBpmSource="creator")
                with self.assertRaises(RuntimeError):
                    self.load(payload)
        payload = manifest()
        payload["fixtures"][0]["referenceBpm"] = 124
        with self.assertRaises(RuntimeError):
            self.load(payload)

    def test_creator_tempo_is_preserved_as_an_independent_numeric_reference(self):
        payload = manifest()
        payload["fixtures"][0].update(referenceBpm=124, referenceBpmSource="creator revision 123")
        fixture = self.load(payload).fixtures[0]
        self.assertEqual(fixture.reference_bpm, 124)
        self.assertEqual(fixture.reference_bpm_source, "creator revision 123")

    def test_octave_comparison_distinguishes_the_known_solarity_defect(self):
        fixture_data = manifest()
        fixture_data["fixtures"][0].update(referenceBpm=124, referenceBpmSource="creator")
        fixture = self.load(fixture_data).fixtures[0]
        for actual, expected in [(124, True), (62, True), (248, True), (165.44, False)]:
            with self.subTest(actual=actual):
                result = goldens.reference_tempo_metrics(actual, 123.05, 123.95, fixture)
                self.assertEqual(result["referenceTempoMatch"], expected)
                self.assertTrue(result["librosaReferenceTempoMatch"])
                self.assertTrue(result["essentiaReferenceTempoMatch"])

    def evaluate(self, *, actual, reference, assert_tempo, analyzer_bpm):
        payload = manifest()
        payload["fixtures"][0].update(
            referenceBpm=reference,
            referenceBpmSource="creator revision 123",
            assertTempo=assert_tempo,
        )
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["evaluate-dj-goldens.py"]),
            patch.object(goldens, "load_corpus", return_value=self.load(payload)),
            patch.object(goldens, "download_fixture", return_value=pathlib.Path("local.ogg")),
            patch.object(goldens, "rust_analysis", return_value={"rhythm": {"bpm": actual, "beats": []}}),
            patch.object(goldens, "librosa_analysis", return_value={"tempo": analyzer_bpm, "beats": []}),
            patch.object(goldens, "essentia_analysis", return_value={"tempo": analyzer_bpm, "beats": [], "key": "C major"}),
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = goldens.main()
        return exit_code, json.loads(output.getvalue())

    def test_asserted_creator_tempo_failure_is_distinct_from_analyzer_disagreement(self):
        exit_code, report = self.evaluate(actual=165.44, reference=124, assert_tempo=True, analyzer_bpm=124)
        self.assertEqual(exit_code, 1)
        self.assertEqual(len(report["failures"]), 1)
        self.assertIn("creator revision 123", report["failures"][0])
        self.assertEqual(len(report["disagreements"]), 2)
        self.assertEqual(report["aggregate"]["referenceTempoFixtureCount"], 1)
        self.assertEqual(report["aggregate"]["referenceTempoMatches"], 0)

    def test_matching_creator_tempo_is_not_failed_by_an_analyzer_disagreement(self):
        exit_code, report = self.evaluate(actual=124, reference=124, assert_tempo=True, analyzer_bpm=165.44)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["failures"], [])
        self.assertEqual(len(report["disagreements"]), 2)
        self.assertEqual(report["aggregate"]["referenceTempoMatches"], 1)

    def test_non_gating_tempo_reference_records_the_known_defect(self):
        exit_code, report = self.evaluate(actual=165.44, reference=124, assert_tempo=False, analyzer_bpm=124)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["failures"], [])
        self.assertEqual(len(report["disagreements"]), 3)

    def test_download_publishes_only_verified_bytes_and_cleans_failed_acquisition(self):
        data = b"verified local audio fixture bytes"
        payload = manifest()
        payload["fixtures"][0].update(
            sha256=hashlib.sha256(data).hexdigest(),
            sourceBytes=len(data),
            sourceSha1=hashlib.sha1(data).hexdigest(),
        )
        corpus = self.load(payload)
        fixture = corpus.fixtures[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = pathlib.Path(directory)
            destination = cache / fixture.filename
            test_case = self

            class UnpublishedStream(io.BytesIO):
                def read(self, size):
                    test_case.assertFalse(destination.exists(), "unverified download became visible")
                    return super().read(size)

            with (
                patch.object(goldens, "CACHE", cache),
                patch.object(goldens.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong bytes")),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                    goldens.download_fixture(corpus, fixture)
                self.assertEqual(list(cache.iterdir()), [])
            with (
                patch.object(goldens, "CACHE", cache),
                patch.object(goldens.urllib.request, "urlopen", return_value=UnpublishedStream(data)),
                redirect_stderr(io.StringIO()),
            ):
                downloaded = goldens.download_fixture(corpus, fixture)
                self.assertEqual(downloaded.read_bytes(), data)
                self.assertEqual(list(cache.iterdir()), [downloaded])

if __name__ == "__main__":
    unittest.main()
