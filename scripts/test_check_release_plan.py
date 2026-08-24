#!/usr/bin/env python3
from __future__ import annotations

import unittest

import check_release_plan
from check_release_plan import CONSUMER_CHECKS, ISSUE, PACKAGES, contract_errors, control_binding_errors


def manifest() -> dict:
    return {
        "issue": ISSUE,
        "dependency_order": [name for name, _version in PACKAGES],
        "packages": [{"name": name, "version": version} for name, version in PACKAGES],
        "required_consumer_checks": CONSUMER_CHECKS,
        "fast_continuation": False,
        "github_releases": [],
        "source_sha": "a" * 40,
    }


class ReleaseContractTests(unittest.TestCase):
    def test_exact_contract_is_valid(self) -> None:
        self.assertEqual(contract_errors(manifest()), [])

    def test_wrong_issue_is_rejected(self) -> None:
        candidate = manifest()
        candidate["issue"] = ISSUE + 1
        self.assertTrue(any("destination issue" in error for error in contract_errors(candidate)))

    def test_broader_package_wave_is_rejected(self) -> None:
        candidate = manifest()
        candidate["packages"].append({"name": "moenarch-audio-analysis-pitch", "version": "0.1.1"})
        self.assertTrue(any("package versions" in error for error in contract_errors(candidate)))

    def test_wrong_order_is_rejected(self) -> None:
        candidate = manifest()
        candidate["dependency_order"].reverse()
        self.assertTrue(any("dependency_order" in error for error in contract_errors(candidate)))

    def test_candidate_gate_cannot_be_removed(self) -> None:
        candidate = manifest()
        candidate["required_consumer_checks"] = []
        self.assertTrue(any("consumer gate" in error for error in contract_errors(candidate)))

    def test_fast_continuation_is_rejected(self) -> None:
        candidate = manifest()
        candidate["fast_continuation"] = True
        self.assertTrue(any("fast_continuation" in error for error in contract_errors(candidate)))

    def test_manifest_only_control_binding_is_required(self) -> None:
        path = check_release_plan.ROOT / "releases/native-whisperx-audio-contract-closure.toml"
        self.assertEqual(
            control_binding_errors(
                manifest(), path, "b" * 40,
                ["releases/native-whisperx-audio-contract-closure.toml"], True,
            ),
            [],
        )
        errors = control_binding_errors(
            manifest(), path, "b" * 40,
            ["Cargo.toml", "releases/native-whisperx-audio-contract-closure.toml"], True,
        )
        self.assertTrue(any("only by the release manifest" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
