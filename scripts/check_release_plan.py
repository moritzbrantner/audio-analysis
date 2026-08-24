#!/usr/bin/env python3
"""Validate the exact six-package native-whisperx audio contract release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from publish_release import CommandEffects, ReleaseError, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "releases/native-whisperx-audio-contract-closure.toml"
ISSUE = 6
PACKAGES = [
    ("moenarch-audio-analysis-core", "0.1.1"),
    ("moenarch-audio-analysis-fourier", "0.1.1"),
    ("moenarch-audio-analysis-recognition", "0.1.1"),
    ("moenarch-audio-analysis-io", "0.1.2"),
    ("moenarch-audio-analysis-speakers", "0.1.5"),
    ("moenarch-audio-analysis-transcription", "0.1.16"),
]
CONSUMER_CHECKS = ["bash scripts/check_native_whisperx_contract_candidate.sh"]


def load_manifest(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"cannot load release manifest: {error}") from error


def cargo_metadata() -> dict:
    completed = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def contract_errors(manifest: dict) -> list[str]:
    errors: list[str] = []
    expected_names = [name for name, _version in PACKAGES]
    expected_versions = dict(PACKAGES)
    if manifest.get("issue") != ISSUE:
        errors.append(f"manifest must bind destination issue {ISSUE}")
    if manifest.get("dependency_order") != expected_names:
        errors.append("dependency_order must match the six-package contract")
    packages = manifest.get("packages", [])
    actual_versions = {
        package.get("name"): package.get("version")
        for package in packages if isinstance(package, dict)
    }
    if actual_versions != expected_versions:
        errors.append("package versions must match the six-package contract")
    if manifest.get("required_consumer_checks") != CONSUMER_CHECKS:
        errors.append("native-whisperx candidate check must be the only consumer gate")
    if manifest.get("fast_continuation") is not False:
        errors.append("fast_continuation must be false")
    if manifest.get("github_releases", []) != []:
        errors.append("this closure must not declare GitHub Releases")
    return errors


def control_binding_errors(manifest: dict, path: Path, head: str, changed: list[str], ancestor: bool) -> list[str]:
    source = manifest.get("source_sha")
    relative = path.relative_to(ROOT).as_posix()
    errors: list[str] = []
    if not isinstance(source, str) or len(source) != 40:
        errors.append("source_sha must be a full commit SHA")
    if not ancestor or source == head:
        errors.append("source_sha must be an ancestor of the control head")
    if changed != [relative]:
        errors.append("control head must differ from source_sha only by the release manifest")
    return errors


def validate(path: Path) -> tuple[dict, dict]:
    manifest = load_manifest(path)
    errors = contract_errors(manifest)
    metadata = cargo_metadata()
    try:
        validate_manifest(ROOT, manifest, metadata)
    except ReleaseError as error:
        errors.append(str(error))
    source = manifest.get("source_sha", "")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(source), head], cwd=ROOT, check=False
    ).returncode == 0
    changed = subprocess.run(
        ["git", "diff", "--name-only", str(source), head],
        cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    errors.extend(control_binding_errors(manifest, path, head, changed, ancestor))
    if errors:
        raise ReleaseError("; ".join(errors))
    return manifest, metadata


def package_release(manifest: dict, metadata: dict) -> None:
    metadata_by_name = {package["name"]: package for package in metadata["packages"]}
    patches = {
        package["name"]: str(Path(metadata_by_name[package["name"]]["manifest_path"]).parent)
        for package in manifest["packages"]
    }
    effects = CommandEffects(ROOT)
    for package in manifest["packages"]:
        effects.package(package["name"], package["version"], patches)
        print(f"PACKAGED {package['name']} {package['version']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--package-release", action="store_true")
    parser.add_argument("plan", nargs="?", type=Path, default=PLAN)
    args = parser.parse_args()
    if args.check == args.package_release:
        print("error: choose exactly one of --check or --package-release", file=sys.stderr)
        return 2
    path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    try:
        manifest, metadata = validate(path)
        if args.package_release:
            package_release(manifest, metadata)
    except (ReleaseError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"native-whisperx audio contract release plan valid: {len(PACKAGES)} packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
