#!/usr/bin/env python3
"""Behavior tests for the repository-owned release publisher."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock


SCRIPT = Path(__file__).with_name("publish_release.py")
SPEC = importlib.util.spec_from_file_location("publish_release", SCRIPT)
assert SPEC and SPEC.loader
publish_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_release)

REPOSITORY = "moritzbrantner/audio-analysis"
HEAD = "a" * 40
SOURCE = "b" * 40
REPAIR_SOURCE = "d" * 40
CHECKSUM = "c" * 64
CONFIG_COMMANDS = tomllib.loads(
    (SCRIPT.parents[1] / ".agent-loop.toml").read_text(encoding="utf-8")
)["verification"]["commands"]
ENVIRONMENT = {
    "AGENT_LOOP_REPOSITORY": REPOSITORY,
    "AGENT_LOOP_ISSUE": "7",
    "AGENT_LOOP_HEAD_SHA": HEAD,
}


def registry_record(name: str, version: str) -> dict[str, Any]:
    return {
        "crate": name,
        "num": version,
        "yanked": False,
        "checksum": CHECKSUM,
    }


class ForbiddenEffects:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        self.calls.append(name)
        raise AssertionError(f"unexpected external access: {name}")


class FakeEffects:
    def __init__(self, packages: list[dict[str, Any]]) -> None:
        self.calls: list[str] = []
        self.call_counts: dict[str, int] = {}
        self.transitions: dict[tuple[str, int], Callable[[], None]] = {}
        self.repo = REPOSITORY
        self.sha = HEAD
        self.is_clean = True
        self.manifests = ["releases/release.toml"]
        self.issue_payload = {
            "number": 7,
            "state": "OPEN",
            "url": f"https://github.com/{REPOSITORY}/issues/7",
            "labels": [{"name": "release:approved"}],
            "body": "",
        }
        self.metadata = {"packages": packages}
        self.published: set[str] = set()
        self.registry: dict[str, dict[str, Any]] = {}
        self.local_tags: dict[str, str] = {}
        self.remote_tags: dict[str, str] = {}
        self.created_tag_targets: dict[str, str] = {}
        self.releases: dict[str, dict[str, Any]] = {}
        self.fail_publish: str | None = None
        self.fail_create_tag: str | None = None
        self.fail_push_tag: str | None = None
        self.fail_create_release: str | None = None
        self.last_package_patches: dict[str, str] = {}
        self.published_registry_checksums: dict[str, str] = {}

    def _record(self, call: str) -> None:
        self.calls.append(call)
        count = self.call_counts.get(call, 0) + 1
        self.call_counts[call] = count
        transition = self.transitions.get((call, count))
        if transition is not None:
            transition()

    def transition(
        self, call: str, count: int, callback: Callable[[], None]
    ) -> None:
        self.transitions[(call, count)] = callback

    def repository(self) -> str:
        self._record("repository")
        return self.repo

    def head(self) -> str:
        self._record("head")
        return self.sha

    def clean(self) -> bool:
        self._record("clean")
        return self.is_clean

    def tracked_manifests(self) -> list[str]:
        self._record("tracked-manifests")
        return self.manifests

    def source_is_ancestor(self, source: str, head: str) -> bool:
        self._record(f"ancestor:{source}:{head}")
        return source == SOURCE and head == self.sha

    def changed_paths(self, source: str, head: str) -> list[str]:
        self._record(f"changed:{source}:{head}")
        return ["releases/release.toml"]

    def issue(self, repository: str, number: int) -> dict[str, Any]:
        self._record(f"issue:{repository}#{number}")
        return self.issue_payload

    def cargo_metadata(self) -> dict[str, Any]:
        self._record("cargo-metadata")
        return self.metadata

    def verify(self, command: str) -> None:
        self._record(f"verify:{command}")

    def registry_version(self, name: str, version: str) -> dict[str, Any] | None:
        self._record(f"registry:{name}@{version}")
        if name in self.registry:
            return self.registry[name]
        if name in self.published:
            return {
                **registry_record(name, version),
                "checksum": self.published_registry_checksums.get(name, CHECKSUM),
            }
        return None

    def package(self, name: str, version: str, patches: dict[str, str]) -> str:
        self._record(f"package:{name}@{version}")
        self.last_package_patches = dict(patches)
        return CHECKSUM

    def published_archive_checksum(self, name: str, version: str) -> str:
        self._record(f"published-archive-checksum:{name}@{version}")
        raise AssertionError("the publisher must not trust a local Cargo archive")

    def publish(self, name: str) -> None:
        self._record(f"publish:{name}")
        if name == self.fail_publish:
            raise publish_release.ReleaseError(f"simulated publish failure: {name}")
        self.published.add(name)

    def wait_for_registry(self) -> None:
        self._record("wait")

    def local_tag_target(self, tag: str) -> str | None:
        self._record(f"local-tag:{tag}")
        return self.local_tags.get(tag)

    def remote_tag_target(self, tag: str) -> str | None:
        self._record(f"remote-tag:{tag}")
        return self.remote_tags.get(tag)

    def create_tag(self, tag: str, target: str, message: str) -> None:
        self._record(f"create-tag:{tag}")
        self.created_tag_targets[tag] = target
        if tag == self.fail_create_tag:
            raise publish_release.ReleaseError(f"simulated tag creation failure: {tag}")
        self.local_tags[tag] = target

    def push_tag(self, tag: str) -> None:
        self._record(f"push-tag:{tag}")
        if tag == self.fail_push_tag:
            raise publish_release.ReleaseError(f"simulated tag push failure: {tag}")
        self.remote_tags[tag] = self.local_tags[tag]

    def release(self, repository: str, tag: str) -> dict[str, Any] | None:
        self._record(f"release:{tag}")
        return self.releases.get(tag)

    def create_release(
        self, repository: str, tag: str, title: str, notes: str
    ) -> None:
        self._record(f"create-release:{tag}")
        if tag == self.fail_create_release:
            raise publish_release.ReleaseError(f"simulated release creation failure: {tag}")
        self.releases[tag] = {
            "tagName": tag,
            "name": title,
            "body": notes,
            "isDraft": False,
            "isPrerelease": False,
        }


def package_record(
    root: Path,
    name: str,
    version: str = "1.0.0",
    dependencies: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = f"crates/{name}/Cargo.toml"
    path = root / manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[package]\nname = {json.dumps(name)}\nversion = {json.dumps(version)}\n"
    )
    manifest_package = {
        "name": name,
        "version": version,
        "owner": REPOSITORY,
        "manifest_path": manifest_path,
        "dependencies": list(dependencies),
        "tag": f"{name}-v{version}",
    }
    metadata_package = {
        "name": name,
        "version": version,
        "manifest_path": str(path.resolve()),
        "publish": None,
        "dependencies": [
            {
                "name": dependency,
                "kind": None,
                "req": f"={version}",
                "path": str((root / f"crates/{dependency}").resolve()),
                "source": None,
            }
            for dependency in dependencies
        ],
    }
    ownership = {
        "current_package_name": name,
        "manifest_path": manifest_path,
        "intended_next_release_owner": REPOSITORY,
        "publication_class": "crates.io",
        "automatic_publish_eligible": True,
    }
    return manifest_package, metadata_package, ownership


def write_fixture(
    root: Path,
    package_specs: list[tuple[str, str, tuple[str, ...]]],
    *,
    releases: bool = True,
) -> tuple[list[dict[str, Any]], FakeEffects]:
    manifest_packages: list[dict[str, Any]] = []
    metadata_packages: list[dict[str, Any]] = []
    ownership: list[dict[str, Any]] = []
    for name, version, dependencies in package_specs:
        manifest, metadata, owned = package_record(root, name, version, dependencies)
        manifest_packages.append(manifest)
        metadata_packages.append(metadata)
        ownership.append(owned)
    ownership_path = root / "docs/repository-split/package-ownership.json"
    ownership_path.parent.mkdir(parents=True, exist_ok=True)
    ownership_path.write_text(json.dumps({"packages": ownership}), encoding="utf-8")
    (root / ".agent-loop.toml").write_text(
        (SCRIPT.parents[1] / ".agent-loop.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_manifest(root / "releases/release.toml", manifest_packages, releases=releases)
    effects = FakeEffects(metadata_packages)
    authorize_manifest(root, effects)
    return manifest_packages, effects


def authorize_manifest(root: Path, effects: FakeEffects) -> None:
    digest = hashlib.sha256((root / "releases/release.toml").read_bytes()).hexdigest()
    effects.issue_payload["body"] = "\n".join(
        (
            f"Release control head SHA: {effects.sha}",
            f"Release manifest SHA-256: {digest}",
        )
    )


def write_manifest(
    path: Path,
    packages: list[dict[str, Any]],
    *,
    releases: bool = True,
    repository: str = REPOSITORY,
    issue: int = 7,
    source: str = SOURCE,
    repair_source: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "schema_version = 1",
        f"repository = {json.dumps(repository)}",
        f"issue = {issue}",
        f"source_sha = {json.dumps(source)}",
        'registry = "crates.io"',
        "dependency_order = " + json.dumps([package["name"] for package in packages]),
        "expected_tags = " + json.dumps([package["tag"] for package in packages]),
        "required_checks = " + json.dumps(CONFIG_COMMANDS),
        'required_consumer_checks = ["true"]',
    ]
    if repair_source is not None:
        lines.insert(4, f"repair_source_sha = {json.dumps(repair_source)}")
    for package in packages:
        package_lines = [
                "",
                "[[packages]]",
                f"name = {json.dumps(package['name'])}",
                f"version = {json.dumps(package['version'])}",
                f"owner = {json.dumps(package['owner'])}",
                f"manifest_path = {json.dumps(package['manifest_path'])}",
                "dependencies = " + json.dumps(package["dependencies"]),
                f"tag = {json.dumps(package['tag'])}",
            ]
        if "published_checksum" in package:
            package_lines.append(
                f"published_checksum = {json.dumps(package['published_checksum'])}"
            )
        lines.extend(package_lines)
    if releases:
        for package in packages:
            lines.extend(
                [
                    "",
                    "[[github_releases]]",
                    f"tag = {json.dumps(package['tag'])}",
                    f"title = {json.dumps('Release ' + package['name'] + ' ' + package['version'])}",
                    f"notes = {json.dumps('Verified immutable release.')}",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pin_registry_records(
    root: Path,
    packages: list[dict[str, Any]],
    effects: FakeEffects,
    *,
    releases: bool = True,
) -> None:
    for package in packages:
        record = effects.registry.get(package["name"])
        if record is not None:
            package["published_checksum"] = record["checksum"]
    write_manifest(root / "releases/release.toml", packages, releases=releases)
    authorize_manifest(root, effects)


class PublishReleaseTests(unittest.TestCase):
    def test_missing_agent_loop_binding_refuses_before_external_access(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            effects = ForbiddenEffects()
            with self.assertRaisesRegex(
                publish_release.ReleaseError,
                "AGENT_LOOP_REPOSITORY, AGENT_LOOP_ISSUE, and AGENT_LOOP_HEAD_SHA",
            ):
                publish_release.run_release(Path(temp), {}, effects)
            self.assertEqual(effects.calls, [])

    def test_malformed_and_ambiguous_checked_manifests_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            malformed = root / "releases/broken.toml"
            malformed.write_text("not = [valid", encoding="utf-8")
            effects.manifests.append("releases/broken.toml")
            with self.assertRaisesRegex(publish_release.ReleaseError, "malformed"):
                publish_release.run_release(root, ENVIRONMENT, effects)

            write_manifest(malformed, packages)
            with self.assertRaisesRegex(publish_release.ReleaseError, "multiple checked"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_repository_head_manifest_and_issue_bindings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.repo = "moritzbrantner/rust-packages"
            with self.assertRaisesRegex(publish_release.ReleaseError, "owned destination"):
                publish_release.run_release(root, ENVIRONMENT, effects)

            effects.repo = REPOSITORY
            effects.sha = "b" * 40
            with self.assertRaisesRegex(publish_release.ReleaseError, "HEAD_SHA"):
                publish_release.run_release(root, ENVIRONMENT, effects)

            effects.sha = HEAD
            write_manifest(
                root / "releases/release.toml",
                [package_record(root, "foundation-a")[0]],
                source="c" * 40,
            )
            authorize_manifest(root, effects)
            with self.assertRaisesRegex(publish_release.ReleaseError, "source_sha"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_open_destination_issue_requires_authorization_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.issue_payload["labels"] = []
            with self.assertRaisesRegex(publish_release.ReleaseError, "lacks release:approved"):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertFalse(any(call.startswith("package:") for call in effects.calls))

    def test_issue_authorizes_the_exact_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.issue_payload["body"] = "\n".join(
                (
                    f"Release control head SHA: {HEAD}",
                    "Release manifest SHA-256: " + "0" * 64,
                )
            )
            with self.assertRaisesRegex(publish_release.ReleaseError, "exact manifest digest"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_issue_authorizes_the_exact_control_head_even_for_same_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            other_head = "d" * 40
            effects.sha = other_head
            environment = {**ENVIRONMENT, "AGENT_LOOP_HEAD_SHA": other_head}

            with self.assertRaisesRegex(
                publish_release.ReleaseError, "exact release control head"
            ):
                publish_release.run_release(root, environment, effects)
            self.assertNotIn("publish:foundation-a", effects.calls)

    def test_issue_refuses_ambiguous_control_head_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.issue_payload["body"] += (
                f"\nRelease control head SHA: {'d' * 40}"
            )

            with self.assertRaisesRegex(
                publish_release.ReleaseError, "exact release control head"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertNotIn("publish:foundation-a", effects.calls)

    def test_source_commit_may_differ_only_by_the_checked_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.changed_paths = lambda source, head: [
                "crates/foundation-a/src/lib.rs",
                "releases/release.toml",
            ]
            with self.assertRaisesRegex(publish_release.ReleaseError, "more than"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_partial_wave_control_repair_preserves_immutable_release_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            packages[0]["published_checksum"] = CHECKSUM
            write_manifest(
                root / "releases/release.toml",
                packages,
                repair_source=REPAIR_SOURCE,
            )
            authorize_manifest(root, effects)
            effects.source_is_ancestor = lambda source, head: (source, head) in {
                (SOURCE, REPAIR_SOURCE),
                (REPAIR_SOURCE, HEAD),
            }
            effects.changed_paths = lambda source, head: (
                [
                    ".agent-loop.toml",
                    ".harness/invariants.md",
                    "docs/AGENT_DRIVEN_RELEASES.md",
                    "docs/RELEASE_CHECKLIST.md",
                    "releases/release.toml",
                    "scripts/check_native_whisperx_contract_candidate.sh",
                    "scripts/check_release_plan.py",
                    "scripts/publish_release.py",
                    "scripts/test_check_release_plan.py",
                    "scripts/test_publish_release.py",
                ]
                if (source, head) == (SOURCE, REPAIR_SOURCE)
                else ["releases/release.toml"]
            )
            effects.registry["foundation-a"] = registry_record(
                "foundation-a", "1.0.0"
            )
            result = publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertEqual(result["tags"][0]["status"], "created-and-pushed")
            self.assertEqual(effects.remote_tags["foundation-a-v1.0.0"], SOURCE)

    def test_partial_wave_control_repair_refuses_crate_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            write_manifest(
                root / "releases/release.toml",
                packages,
                repair_source=REPAIR_SOURCE,
            )
            authorize_manifest(root, effects)
            effects.source_is_ancestor = lambda source, head: (source, head) in {
                (SOURCE, REPAIR_SOURCE),
                (REPAIR_SOURCE, HEAD),
            }
            effects.changed_paths = lambda source, head: (
                ["crates/foundation-a/src/lib.rs", "scripts/publish_release.py"]
                if (source, head) == (SOURCE, REPAIR_SOURCE)
                else ["releases/release.toml"]
            )

            with self.assertRaisesRegex(
                publish_release.ReleaseError,
                "scripts/check_native_whisperx_contract_candidate.sh",
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_candidate_consumer_check_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            manifest = root / "releases/release.toml"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    'required_consumer_checks = ["true"]',
                    "required_consumer_checks = []",
                ),
                encoding="utf-8",
            )
            authorize_manifest(root, effects)
            with self.assertRaisesRegex(publish_release.ReleaseError, "consumer_checks"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_package_version_ownership_order_and_tag_are_validated(self) -> None:
        cases = (
            ("version", lambda packages: packages[0].update(version="9.9.9")),
            ("owner", lambda packages: packages[0].update(owner="someone/else")),
            ("dependency order", lambda packages: packages.reverse()),
            ("tag", lambda packages: packages[0].update(tag="latest")),
        )
        for expected, mutate in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                packages, effects = write_fixture(
                    root,
                    [
                        ("foundation-a", "1.0.0", ()),
                        ("foundation-b", "1.0.0", ("foundation-a",)),
                    ],
                )
                mutate(packages)
                write_manifest(root / "releases/release.toml", packages)
                authorize_manifest(root, effects)
                with self.assertRaisesRegex(publish_release.ReleaseError, expected):
                    publish_release.run_release(root, ENVIRONMENT, effects)
                self.assertFalse(any(call.startswith("publish:") for call in effects.calls))

    def test_in_wave_dependencies_require_exact_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            foundation_b = next(
                package
                for package in effects.metadata["packages"]
                if package["name"] == "foundation-b"
            )
            foundation_b["dependencies"][0]["req"] = "^1.0.0"

            with self.assertRaisesRegex(
                publish_release.ReleaseError, "exact version requirement"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertFalse(any(call.startswith("publish:") for call in effects.calls))

    def test_git_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            foundation_b = next(
                package
                for package in effects.metadata["packages"]
                if package["name"] == "foundation-b"
            )
            dependency = foundation_b["dependencies"][0]
            dependency["path"] = None
            dependency["source"] = "git+https://example.invalid/repository?branch=main"

            with self.assertRaisesRegex(publish_release.ReleaseError, "Git dependency"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_escaping_path_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            foundation_b = next(
                package
                for package in effects.metadata["packages"]
                if package["name"] == "foundation-b"
            )
            foundation_b["dependencies"][0]["path"] = str(root.parent / "foundation-a")

            with self.assertRaisesRegex(publish_release.ReleaseError, "escapes"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_package_version_must_be_explicit_in_its_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            (root / "crates/foundation-a/Cargo.toml").write_text(
                "[package]\nname = \"foundation-a\"\nversion.workspace = true\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(publish_release.ReleaseError, "explicit version"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_registry_verified_workspace_prerequisite_is_patched_for_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            write_manifest(root / "releases/release.toml", [packages[1]])
            authorize_manifest(root, effects)
            effects.registry["foundation-a"] = registry_record(
                "foundation-a", "1.0.0"
            )

            publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertIn("foundation-a", effects.last_package_patches)

    def test_registry_conflict_and_non_prefix_partial_state_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            effects.registry["foundation-a"] = {
                "crate": "foundation-a",
                "num": "1.0.0",
                "yanked": True,
            }
            with self.assertRaisesRegex(publish_release.ReleaseError, "registry conflict"):
                publish_release.run_release(root, ENVIRONMENT, effects)

            effects.registry = {
                "foundation-b": registry_record("foundation-b", "1.0.0")
            }
            with self.assertRaisesRegex(publish_release.ReleaseError, "published prefix"):
                publish_release.run_release(root, ENVIRONMENT, effects)

            effects.registry = {
                "foundation-a": {
                    **registry_record("foundation-a", "1.0.0"),
                    "checksum": "d" * 64,
                }
            }
            packages[0]["published_checksum"] = CHECKSUM
            write_manifest(root / "releases/release.toml", packages)
            authorize_manifest(root, effects)
            with self.assertRaisesRegex(publish_release.ReleaseError, "registry conflict"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_partial_resume_requires_manifest_pinned_registry_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            effects.registry["foundation-a"] = registry_record("foundation-a", "1.0.0")
            with self.assertRaisesRegex(
                publish_release.ReleaseError, "published_checksum"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertNotIn("publish:foundation-a", effects.calls)
            self.assertNotIn("publish:foundation-b", effects.calls)

    def test_successful_publish_binds_the_fresh_registry_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.published_registry_checksums["foundation-a"] = "e" * 64

            result = publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertEqual(
                result["packages"],
                [{"name": "foundation-a", "status": "published-and-verified"}],
            )
            self.assertNotIn(
                "published-archive-checksum:foundation-a@1.0.0", effects.calls
            )

    def test_successful_publish_rejects_an_invalid_fresh_registry_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.published_registry_checksums["foundation-a"] = "not-a-checksum"

            with self.assertRaisesRegex(
                publish_release.ReleaseError, "invalid registry checksum"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_partial_resume_accepts_manifest_pinned_published_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            published_checksum = "d" * 64
            packages[0]["published_checksum"] = published_checksum
            write_manifest(root / "releases/release.toml", packages)
            authorize_manifest(root, effects)
            effects.registry["foundation-a"] = {
                **registry_record("foundation-a", "1.0.0"),
                "checksum": published_checksum,
            }

            result = publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertNotIn("publish:foundation-a", effects.calls)
            self.assertIn("publish:foundation-b", effects.calls)
            self.assertEqual(result["packages"][0]["status"], "registry-verified")

    def test_complete_resume_bounds_registry_reads_per_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            for package in packages:
                package["published_checksum"] = CHECKSUM
                effects.registry[package["name"]] = registry_record(
                    package["name"], package["version"]
                )
                effects.local_tags[package["tag"]] = SOURCE
                effects.remote_tags[package["tag"]] = SOURCE
                effects.releases[package["tag"]] = {
                    "tagName": package["tag"],
                    "name": f"Release {package['name']} {package['version']}",
                    "body": "Verified immutable release.",
                    "isDraft": False,
                    "isPrerelease": False,
                }
            write_manifest(root / "releases/release.toml", packages)
            authorize_manifest(root, effects)

            publish_release.run_release(root, ENVIRONMENT, effects)

            registry_reads = sum(
                count
                for call, count in effects.call_counts.items()
                if call.startswith("registry:")
            )
            self.assertLessEqual(registry_reads, 12)

    def test_published_checksum_requires_valid_digest_and_visible_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            packages[0]["published_checksum"] = "not-a-checksum"
            write_manifest(root / "releases/release.toml", packages)
            authorize_manifest(root, effects)
            with self.assertRaisesRegex(publish_release.ReleaseError, "published_checksum"):
                publish_release.run_release(root, ENVIRONMENT, effects)

            packages[0]["published_checksum"] = "d" * 64
            write_manifest(root / "releases/release.toml", packages)
            authorize_manifest(root, effects)
            with self.assertRaisesRegex(publish_release.ReleaseError, "registry-visible"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_publish_failure_stops_before_tags_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.fail_publish = "foundation-a"
            with self.assertRaisesRegex(publish_release.ReleaseError, "simulated"):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertFalse(any(call.startswith("create-tag:") for call in effects.calls))
            self.assertFalse(any(call.startswith("push-tag:") for call in effects.calls))
            self.assertFalse(any(call.startswith("create-release:") for call in effects.calls))

    def test_checkout_mutation_during_packaging_stops_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])

            def mutate_checkout(
                name: str, version: str, patches: dict[str, str]
            ) -> str:
                effects.calls.append(f"package:{name}@{version}")
                effects.is_clean = False
                return CHECKSUM

            effects.package = mutate_checkout
            with self.assertRaisesRegex(publish_release.ReleaseError, "changed"):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertNotIn("publish:foundation-a", effects.calls)

    def test_registry_verification_precedes_tag_and_release_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            result = publish_release.run_release(root, ENVIRONMENT, effects)
            tag = "foundation-a-v1.0.0"
            publish_index = effects.calls.index("publish:foundation-a")
            verified_index = next(
                index
                for index, call in enumerate(effects.calls[publish_index + 1 :], publish_index + 1)
                if call == "registry:foundation-a@1.0.0"
            )
            create_tag_index = effects.calls.index(f"create-tag:{tag}")
            push_tag_index = effects.calls.index(f"push-tag:{tag}")
            create_release_index = effects.calls.index(f"create-release:{tag}")
            self.assertLess(publish_index, verified_index)
            self.assertLess(verified_index, create_tag_index)
            self.assertLess(create_tag_index, push_tag_index)
            self.assertLess(push_tag_index, create_release_index)
            self.assertEqual(result["githubReleases"][0]["status"], "created-and-verified")

    def test_every_release_effect_has_fresh_issue_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            publish_release.run_release(root, ENVIRONMENT, effects)
            mutations = (
                "publish:foundation-a",
                "create-tag:foundation-a-v1.0.0",
                "push-tag:foundation-a-v1.0.0",
                "create-release:foundation-a-v1.0.0",
            )
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    index = effects.calls.index(mutation)
                    self.assertEqual(effects.calls[index - 1], f"issue:{REPOSITORY}#7")

    def test_authorization_revoked_after_packaging_stops_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.transition(
                "package:foundation-a@1.0.0",
                1,
                lambda: effects.issue_payload.update(labels=[]),
            )
            with self.assertRaisesRegex(publish_release.ReleaseError, "lacks release:approved"):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertFalse(any(call.startswith("publish:") for call in effects.calls))
            self.assertFalse(any(call.startswith("create-tag:") for call in effects.calls))
            self.assertFalse(any(call.startswith("push-tag:") for call in effects.calls))

    def test_control_head_authorization_changed_after_packaging_stops_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            digest = hashlib.sha256(
                (root / "releases/release.toml").read_bytes()
            ).hexdigest()
            effects.transition(
                "package:foundation-a@1.0.0",
                1,
                lambda: effects.issue_payload.update(
                    body="\n".join(
                        (
                            f"Release control head SHA: {'d' * 40}",
                            f"Release manifest SHA-256: {digest}",
                        )
                    )
                ),
            )

            with self.assertRaisesRegex(
                publish_release.ReleaseError, "exact release control head"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertNotIn("publish:foundation-a", effects.calls)

    def test_authorization_revoked_after_first_visibility_stops_next_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(
                root,
                [
                    ("foundation-a", "1.0.0", ()),
                    ("foundation-b", "1.0.0", ("foundation-a",)),
                ],
            )
            registry_call = "registry:foundation-a@1.0.0"

            def revoke_on_first_visibility() -> None:
                effects.transition(
                    registry_call,
                    effects.call_counts.get(registry_call, 0) + 1,
                    lambda: effects.issue_payload.update(labels=[]),
                )

            effects.transition("publish:foundation-a", 1, revoke_on_first_visibility)
            with self.assertRaisesRegex(publish_release.ReleaseError, "lacks release:approved"):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertIn("publish:foundation-a", effects.calls)
            self.assertNotIn("publish:foundation-b", effects.calls)
            self.assertFalse(any(call.startswith("create-tag:") for call in effects.calls))
            self.assertFalse(any(call.startswith("push-tag:") for call in effects.calls))

    def test_fresh_registry_yank_stops_before_tagging(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.transition(
                "registry:foundation-a@1.0.0",
                3,
                lambda: effects.registry.update(
                    {
                        "foundation-a": {
                            **registry_record("foundation-a", "1.0.0"),
                            "yanked": True,
                        }
                    }
                ),
            )
            with self.assertRaisesRegex(publish_release.ReleaseError, "registry conflict"):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertNotIn("publish:foundation-a", effects.calls)
            self.assertNotIn("create-tag:foundation-a-v1.0.0", effects.calls)

    def test_registry_version_appearing_before_publish_requires_checksum_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.transition(
                "registry:foundation-a@1.0.0",
                3,
                lambda: effects.registry.update(
                    {"foundation-a": registry_record("foundation-a", "1.0.0")}
                ),
            )
            with self.assertRaisesRegex(
                publish_release.ReleaseError, "pin its published_checksum"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertNotIn("publish:foundation-a", effects.calls)

    def test_registry_appearing_after_ambiguous_publish_requires_checksum_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            issue_call = f"issue:{REPOSITORY}#7"
            effects.fail_publish = "foundation-a"
            effects.transition(
                issue_call,
                4,
                lambda: effects.registry.update(
                    {"foundation-a": registry_record("foundation-a", "1.0.0")}
                ),
            )
            with self.assertRaisesRegex(
                publish_release.ReleaseError, "ambiguous publish failure"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertEqual(effects.call_counts["publish:foundation-a"], 1)

    def test_failed_publish_with_registry_visibility_requires_checksum_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            effects.fail_publish = "foundation-a"
            effects.transition(
                "publish:foundation-a",
                1,
                lambda: effects.registry.update(
                    {"foundation-a": registry_record("foundation-a", "1.0.0")}
                ),
            )
            with self.assertRaisesRegex(
                publish_release.ReleaseError, "ambiguous publish failure"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertEqual(effects.call_counts["publish:foundation-a"], 1)

    def test_conflicting_fresh_tag_stops_before_tag_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            effects.transition(
                f"remote-tag:{tag}",
                2,
                lambda: effects.remote_tags.update({tag: "d" * 40}),
            )
            with self.assertRaisesRegex(publish_release.ReleaseError, "immutable tag conflict"):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertNotIn(f"create-tag:{tag}", effects.calls)
            self.assertNotIn(f"push-tag:{tag}", effects.calls)
            self.assertFalse(any(call.startswith("create-release:") for call in effects.calls))

    def test_tag_at_control_head_is_an_immutable_source_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            effects.registry["foundation-a"] = registry_record(
                "foundation-a", "1.0.0"
            )
            effects.local_tags[tag] = HEAD
            effects.remote_tags[tag] = HEAD

            with self.assertRaisesRegex(
                publish_release.ReleaseError, "immutable tag conflict"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertNotIn(f"create-tag:{tag}", effects.calls)
            self.assertNotIn(f"push-tag:{tag}", effects.calls)
            self.assertNotIn(f"create-release:{tag}", effects.calls)

    def test_new_tag_is_created_at_source_sha_not_control_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"

            publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertEqual(effects.created_tag_targets[tag], SOURCE)
            self.assertEqual(effects.local_tags[tag], SOURCE)
            self.assertEqual(effects.remote_tags[tag], SOURCE)

    def test_existing_source_tag_reconciles_when_control_head_differs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            effects.registry["foundation-a"] = registry_record(
                "foundation-a", "1.0.0"
            )
            effects.local_tags[tag] = SOURCE
            effects.remote_tags[tag] = SOURCE
            pin_registry_records(root, packages, effects)

            result = publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertEqual(result["tags"], [{"tag": tag, "status": "existing"}])
            self.assertEqual(
                result["githubReleases"],
                [{"tag": tag, "status": "created-and-verified"}],
            )
            self.assertNotIn(f"create-tag:{tag}", effects.calls)
            self.assertNotIn(f"push-tag:{tag}", effects.calls)

    def test_exact_tag_appearing_during_final_authority_reconciles_failed_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            issue_call = f"issue:{REPOSITORY}#7"
            effects.registry["foundation-a"] = registry_record("foundation-a", "1.0.0")
            pin_registry_records(root, packages, effects)
            effects.fail_create_tag = tag
            effects.transition(
                issue_call,
                3,
                lambda: effects.remote_tags.update({tag: SOURCE}),
            )
            publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertEqual(effects.call_counts[f"create-tag:{tag}"], 1)
            self.assertNotIn(f"push-tag:{tag}", effects.calls)

    def test_failed_tag_creation_reconciles_concurrent_exact_tag_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            effects.fail_create_tag = tag
            effects.transition(
                f"create-tag:{tag}",
                1,
                lambda: effects.remote_tags.update({tag: SOURCE}),
            )
            publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertEqual(effects.call_counts[f"create-tag:{tag}"], 1)
            self.assertNotIn(f"push-tag:{tag}", effects.calls)

    def test_exact_remote_tag_during_final_authority_reconciles_failed_push_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(
                root, [("foundation-a", "1.0.0", ())], releases=False
            )
            tag = "foundation-a-v1.0.0"
            issue_call = f"issue:{REPOSITORY}#7"
            effects.registry["foundation-a"] = registry_record("foundation-a", "1.0.0")
            pin_registry_records(root, packages, effects, releases=False)
            effects.local_tags[tag] = SOURCE
            effects.fail_push_tag = tag
            effects.transition(
                issue_call,
                3,
                lambda: effects.remote_tags.update({tag: SOURCE}),
            )
            publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertEqual(effects.call_counts[f"push-tag:{tag}"], 1)
            self.assertFalse(any(call.startswith("create-release:") for call in effects.calls))

    def test_authority_revoked_after_tag_creation_stops_before_push_and_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            effects.transition(
                f"create-tag:{tag}",
                1,
                lambda: effects.issue_payload.update(labels=[]),
            )

            with self.assertRaisesRegex(
                publish_release.ReleaseError, "lacks release:approved"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertEqual(effects.local_tags[tag], SOURCE)
            self.assertNotIn(f"push-tag:{tag}", effects.calls)
            self.assertNotIn(f"create-release:{tag}", effects.calls)

    def test_remote_tag_moving_to_control_head_stops_release_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            effects.transition(
                f"release:{tag}",
                2,
                lambda: effects.remote_tags.update({tag: HEAD}),
            )

            with self.assertRaisesRegex(
                publish_release.ReleaseError, "remote tag conflict"
            ):
                publish_release.run_release(root, ENVIRONMENT, effects)

            self.assertNotIn(f"create-release:{tag}", effects.calls)

    def test_conflicting_fresh_release_stops_before_release_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            effects.transition(
                f"release:{tag}",
                2,
                lambda: effects.releases.update(
                    {
                        tag: {
                            "tagName": tag,
                            "name": "Conflicting release",
                            "body": "Not reviewed",
                            "isDraft": False,
                            "isPrerelease": False,
                        }
                    }
                ),
            )
            with self.assertRaisesRegex(publish_release.ReleaseError, "Release conflict"):
                publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertNotIn(f"create-release:{tag}", effects.calls)

    def test_exact_release_during_final_authority_reconciles_failed_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            issue_call = f"issue:{REPOSITORY}#7"
            exact = {
                "tagName": tag,
                "name": "Release foundation-a 1.0.0",
                "body": "Verified immutable release.",
                "isDraft": False,
                "isPrerelease": False,
            }
            effects.registry["foundation-a"] = registry_record("foundation-a", "1.0.0")
            pin_registry_records(root, packages, effects)
            effects.local_tags[tag] = SOURCE
            effects.remote_tags[tag] = SOURCE
            effects.fail_create_release = tag
            effects.transition(
                issue_call,
                3,
                lambda: effects.releases.update({tag: exact}),
            )
            publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertEqual(effects.call_counts[f"create-release:{tag}"], 1)

    def test_failed_release_creation_reconciles_concurrent_exact_release_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(root, [("foundation-a", "1.0.0", ())])
            tag = "foundation-a-v1.0.0"
            effects.fail_create_release = tag
            effects.transition(
                f"create-release:{tag}",
                1,
                lambda: effects.releases.update(
                    {
                        tag: {
                            "tagName": tag,
                            "name": "Release foundation-a 1.0.0",
                            "body": "Verified immutable release.",
                            "isDraft": False,
                            "isPrerelease": False,
                        }
                    }
                ),
            )
            publish_release.run_release(root, ENVIRONMENT, effects)
            self.assertEqual(effects.call_counts[f"create-release:{tag}"], 1)

    def test_undeclared_existing_github_release_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, effects = write_fixture(
                root, [("foundation-a", "1.0.0", ())], releases=False
            )
            tag = "foundation-a-v1.0.0"
            effects.registry["foundation-a"] = registry_record("foundation-a", "1.0.0")
            effects.local_tags[tag] = SOURCE
            effects.remote_tags[tag] = SOURCE
            effects.releases[tag] = {
                "tagName": tag,
                "name": "Unreviewed",
                "body": "Unreviewed",
                "isDraft": False,
                "isPrerelease": False,
            }
            with self.assertRaisesRegex(publish_release.ReleaseError, "undeclared"):
                publish_release.run_release(root, ENVIRONMENT, effects)

    def test_command_effects_target_source_sha_and_verify_tag_for_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            completed = publish_release.subprocess.CompletedProcess([], 0, "", "")
            with mock.patch.object(
                publish_release.subprocess, "run", return_value=completed
            ) as run:
                effects = publish_release.CommandEffects(root)
                effects.create_tag(
                    "foundation-a-v1.0.0", SOURCE, "Release foundation-a 1.0.0"
                )
                effects.create_release(
                    REPOSITORY,
                    "foundation-a-v1.0.0",
                    "Release foundation-a 1.0.0",
                    "Verified immutable release.",
                )

            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn(
                [
                    "git",
                    "tag",
                    "--annotate",
                    "foundation-a-v1.0.0",
                    SOURCE,
                    "--message",
                    "Release foundation-a 1.0.0",
                ],
                commands,
            )
            release_command = next(
                command
                for command in commands
                if command[:3] == ["gh", "release", "create"]
            )
            self.assertIn("--verify-tag", release_command)

    def test_registry_query_honors_rate_limit_retry_after(self) -> None:
        response = io.BytesIO(
            json.dumps(
                {"version": registry_record("foundation-a", "1.0.0")}
            ).encode()
        )
        rate_limited = publish_release.urllib.error.HTTPError(
            "https://crates.io/api/v1/crates/foundation-a/1.0.0",
            429,
            "rate limited",
            {"Retry-After": "3"},
            None,
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(
                publish_release.urllib.request,
                "urlopen",
                side_effect=[rate_limited, response],
            ),
            mock.patch.object(publish_release.time, "sleep") as sleep,
        ):
            record = publish_release.CommandEffects(Path(temp)).registry_version(
                "foundation-a", "1.0.0"
            )

        self.assertEqual(record, registry_record("foundation-a", "1.0.0"))
        sleep.assert_called_once_with(3.0)

    def test_registry_query_uses_bounded_backoff_without_retry_after(self) -> None:
        rate_limited = publish_release.urllib.error.HTTPError(
            "https://crates.io/api/v1/crates/foundation-a/1.0.0",
            429,
            "rate limited",
            {},
            None,
        )
        response = io.BytesIO(
            json.dumps(
                {"version": registry_record("foundation-a", "1.0.0")}
            ).encode()
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(
                publish_release.urllib.request,
                "urlopen",
                side_effect=[rate_limited, response],
            ),
            mock.patch.object(publish_release.time, "sleep") as sleep,
        ):
            publish_release.CommandEffects(Path(temp)).registry_version(
                "foundation-a", "1.0.0"
            )

        sleep.assert_called_once_with(1)

    def test_cargo_effects_pin_package_and_publish_to_crates_io(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with tempfile.TemporaryDirectory() as external:
                external_target = Path(external)
                archive = external_target / "package/foundation-a-1.0.0.crate"
                archive.parent.mkdir(parents=True)
                archive.write_bytes(b"archive")
                completed = publish_release.subprocess.CompletedProcess([], 0, "", "")
                with (
                    mock.patch.dict(
                        publish_release.os.environ,
                        {"CARGO_TARGET_DIR": str(external_target)},
                    ),
                    mock.patch.object(
                        publish_release.subprocess, "run", return_value=completed
                    ) as run,
                ):
                    effects = publish_release.CommandEffects(root)
                    effects.package(
                        "foundation-a",
                        "1.0.0",
                        {"foundation-a": str(root / "foundation-a")},
                    )
                    effects.publish("foundation-a")
            commands = [call.args[0] for call in run.call_args_list]
            package_command = next(command for command in commands if "--config" in command)
            self.assertEqual(package_command[:7], [
                "cargo",
                "package",
                "-p",
                "foundation-a",
                "--locked",
                "--registry",
                "crates-io",
            ])
            self.assertEqual(package_command[7], "--config")
            self.assertIn(
                [
                    "cargo",
                    "publish",
                    "-p",
                    "foundation-a",
                    "--locked",
                    "--registry",
                    "crates-io",
                ],
                commands,
            )

    def test_relative_cargo_target_directory_is_resolved_from_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "custom-target/package/foundation-a-1.0.0.crate"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"archive")
            completed = publish_release.subprocess.CompletedProcess([], 0, "", "")
            with (
                mock.patch.dict(
                    publish_release.os.environ,
                    {"CARGO_TARGET_DIR": "custom-target"},
                ),
                mock.patch.object(
                    publish_release.subprocess, "run", return_value=completed
                ),
            ):
                checksum = publish_release.CommandEffects(root).package(
                    "foundation-a",
                    "1.0.0",
                    {"foundation-a": str(root / "foundation-a")},
                )
            self.assertEqual(checksum, hashlib.sha256(b"archive").hexdigest())

    def test_cargo_packages_downstream_with_unpublished_local_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["foundation-a", "foundation-b"]\nresolver = "2"\n',
                encoding="utf-8",
            )
            for name in ("foundation-a", "foundation-b"):
                crate = root / name
                (crate / "src").mkdir(parents=True)
                dependencies = (
                    '\n[dependencies]\nfoundation-a = { version = "1.0.0", path = "../foundation-a" }\n'
                    if name == "foundation-b"
                    else ""
                )
                (crate / "Cargo.toml").write_text(
                    f'[package]\nname = "{name}"\nversion = "1.0.0"\nedition = "2021"\nlicense = "MIT"\n'
                    + dependencies,
                    encoding="utf-8",
                )
                (crate / "src/lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
            subprocess.run(
                ["cargo", "generate-lockfile"], cwd=root, check=True, capture_output=True
            )
            with mock.patch.dict(
                publish_release.os.environ, {"CARGO_NET_OFFLINE": "true"}
            ):
                checksum = publish_release.CommandEffects(root).package(
                    "foundation-b",
                    "1.0.0",
                    {
                        "foundation-a": str(root / "foundation-a"),
                        "foundation-b": str(root / "foundation-b"),
                    },
                )
            self.assertRegex(checksum, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
