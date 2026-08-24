#!/usr/bin/env python3
"""Publish the one checked release manifest bound to this Agent Loop invocation.

All validation completes before the first publication, tag, or GitHub Release
side effect. The public interface is the no-argument CLI configured in
``.agent-loop.toml``; ``run_release`` accepts an effects adapter so tests can
replace only network and process boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Protocol


EXPECTED_REPOSITORY = "moritzbrantner/audio-analysis"
AUTHORIZATION_LABEL = "release:approved"
REGISTRY = "crates.io"
FAST_CONTINUATION_ENV = "AUDIO_RELEASE_FAST_CONTINUATION"
SAFEGUARDS_ONLY_ENV = "AUDIO_RELEASE_SAFEGUARDS_ONLY"
OWNERSHIP_PATH = Path("docs/repository-split/package-ownership.json")
ROOT_FIELDS = {
    "schema_version",
    "repository",
    "issue",
    "source_sha",
    "repair_source_sha",
    "registry",
    "dependency_order",
    "expected_tags",
    "packages",
    "github_releases",
    "required_checks",
    "required_consumer_checks",
    "fast_continuation",
}
CONTROL_REPAIR_SCRIPT_PATHS = {
    ".agent-loop.toml",
    ".harness/invariants.md",
    "docs/AGENT_DRIVEN_RELEASES.md",
    "docs/RELEASE_CHECKLIST.md",
    "scripts/check_native_whisperx_contract_candidate.sh",
    "scripts/check_release_plan.py",
    "scripts/publish_release.py",
    "scripts/test_check_release_plan.py",
    "scripts/test_publish_release.py",
}
REGISTRY_QUERY_ATTEMPTS = 8
REGISTRY_QUERY_MAX_BACKOFF_SECONDS = 60.0
PACKAGE_FIELDS = {
    "name",
    "version",
    "owner",
    "manifest_path",
    "dependencies",
    "tag",
    "published_checksum",
}
RELEASE_FIELDS = {"tag", "title", "notes"}


class ReleaseError(RuntimeError):
    """A fail-closed release validation or external-operation failure."""


def _effect_failure(action: str, error: Exception) -> ReleaseError:
    return ReleaseError(f"{action} failed without exact concurrent state: {error}")


class Effects(Protocol):
    def repository(self) -> str: ...
    def head(self) -> str: ...
    def clean(self) -> bool: ...
    def tracked_manifests(self) -> list[str]: ...
    def source_is_ancestor(self, source: str, head: str) -> bool: ...
    def changed_paths(self, source: str, head: str) -> list[str]: ...
    def issue(self, repository: str, number: int) -> dict[str, Any]: ...
    def cargo_metadata(self) -> dict[str, Any]: ...
    def registry_version(self, name: str, version: str) -> dict[str, Any] | None: ...
    def verify(self, command: str) -> None: ...
    def package(self, name: str, version: str, patches: Mapping[str, str]) -> str: ...
    def publish(self, name: str) -> None: ...
    def wait_for_registry(self) -> None: ...
    def local_tag_target(self, tag: str) -> str | None: ...
    def remote_tag_target(self, tag: str) -> str | None: ...
    def create_tag(self, tag: str, target: str, message: str) -> None: ...
    def push_tag(self, tag: str) -> None: ...
    def release(self, repository: str, tag: str) -> dict[str, Any] | None: ...
    def create_release(
        self, repository: str, tag: str, title: str, notes: str
    ) -> None: ...


class CommandEffects:
    """Production adapter for GitHub, Cargo, git, and crates.io."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _run(
        self, args: list[str], *, capture: bool = True, allow_failure: bool = False
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            args,
            cwd=self.root,
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
        if completed.returncode and not allow_failure:
            detail = (completed.stderr or "").strip()
            suffix = f": {detail}" if detail else ""
            raise ReleaseError(f"command failed ({' '.join(args)}){suffix}")
        return completed

    def repository(self) -> str:
        remote = self._run(["git", "config", "--get", "remote.origin.url"]).stdout.strip()
        if remote.startswith("git@github.com:"):
            remote = remote.removeprefix("git@github.com:")
        elif "github.com/" in remote:
            remote = remote.split("github.com/", 1)[1]
        return remote.removesuffix(".git").strip("/")

    def head(self) -> str:
        return self._run(["git", "rev-parse", "HEAD"]).stdout.strip()

    def clean(self) -> bool:
        return not self._run(["git", "status", "--porcelain"]).stdout.strip()

    def source_is_ancestor(self, source: str, head: str) -> bool:
        return self._run(
            ["git", "merge-base", "--is-ancestor", source, head],
            allow_failure=True,
        ).returncode == 0

    def changed_paths(self, source: str, head: str) -> list[str]:
        output = self._run(["git", "diff", "--name-only", source, head]).stdout
        return sorted(line.strip() for line in output.splitlines() if line.strip())

    def tracked_manifests(self) -> list[str]:
        output = self._run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "releases"]
        ).stdout
        return sorted(
            line.strip()
            for line in output.splitlines()
            if line.strip().startswith("releases/") and line.strip().endswith(".toml")
        )

    def issue(self, repository: str, number: int) -> dict[str, Any]:
        output = self._run(
            [
                "gh",
                "issue",
                "view",
                str(number),
                "--repo",
                repository,
                "--json",
                "number,state,url,labels,body",
            ]
        ).stdout
        try:
            return json.loads(output)
        except json.JSONDecodeError as error:
            raise ReleaseError("GitHub issue response was not valid JSON") from error

    def cargo_metadata(self) -> dict[str, Any]:
        output = self._run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"]
        ).stdout
        try:
            return json.loads(output)
        except json.JSONDecodeError as error:
            raise ReleaseError("Cargo metadata response was not valid JSON") from error

    def registry_version(self, name: str, version: str) -> dict[str, Any] | None:
        encoded_name = urllib.parse.quote(name, safe="")
        encoded_version = urllib.parse.quote(version, safe="")
        request = urllib.request.Request(
            f"https://crates.io/api/v1/crates/{encoded_name}/{encoded_version}",
            headers={"User-Agent": "moenarch-nlp-stack-release-control/1"},
        )
        for attempt in range(REGISTRY_QUERY_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                if error.code != 429 or attempt == REGISTRY_QUERY_ATTEMPTS - 1:
                    raise ReleaseError(
                        f"crates.io query failed with HTTP {error.code}"
                    ) from error
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 0.0
                except ValueError:
                    delay = 0.0
                if delay <= 0:
                    delay = min(2**attempt, REGISTRY_QUERY_MAX_BACKOFF_SECONDS)
                time.sleep(min(delay, REGISTRY_QUERY_MAX_BACKOFF_SECONDS))
            except (OSError, ValueError) as error:
                raise ReleaseError(f"crates.io query failed: {error}") from error
        record = payload.get("version")
        if not isinstance(record, dict):
            raise ReleaseError("crates.io returned an invalid version record")
        return record

    def verify(self, command: str) -> None:
        self._run(["bash", "-lc", command], capture=False)

    def package(self, name: str, version: str, patches: Mapping[str, str]) -> str:
        lines = ["[patch.crates-io]"]
        lines.extend(
            f'{json.dumps(package)} = {{ path = {json.dumps(path)} }}'
            for package, path in sorted(patches.items())
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml") as config:
            config.write("\n".join(lines) + "\n")
            config.flush()
            self._run(
                [
                    "cargo",
                    "package",
                    "-p",
                    name,
                    "--locked",
                    "--registry",
                    "crates-io",
                    "--config",
                    config.name,
                ],
                capture=False,
            )
        configured_target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
        target = (
            configured_target
            if configured_target.is_absolute()
            else self.root / configured_target
        )
        archive = target / "package" / f"{name}-{version}.crate"
        try:
            return hashlib.sha256(archive.read_bytes()).hexdigest()
        except OSError as error:
            raise ReleaseError(f"cannot checksum packaged archive {archive}: {error}") from error

    def publish(self, name: str) -> None:
        self._run(
            ["cargo", "publish", "-p", name, "--locked", "--registry", "crates-io"],
            capture=False,
        )

    def wait_for_registry(self) -> None:
        time.sleep(5)

    def local_tag_target(self, tag: str) -> str | None:
        completed = self._run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{}}"],
            allow_failure=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    def remote_tag_target(self, tag: str) -> str | None:
        output = self._run(
            [
                "git",
                "ls-remote",
                "--tags",
                "origin",
                f"refs/tags/{tag}",
                f"refs/tags/{tag}^{{}}",
            ]
        ).stdout
        records = {
            ref: sha for sha, ref in (line.split("\t", 1) for line in output.splitlines())
        }
        return records.get(f"refs/tags/{tag}^{{}}") or records.get(f"refs/tags/{tag}")

    def create_tag(self, tag: str, target: str, message: str) -> None:
        self._run(["git", "tag", "--annotate", tag, target, "--message", message])

    def push_tag(self, tag: str) -> None:
        self._run(["git", "push", "origin", f"refs/tags/{tag}"])

    def release(self, repository: str, tag: str) -> dict[str, Any] | None:
        completed = self._run(
            [
                "gh",
                "release",
                "view",
                tag,
                "--repo",
                repository,
                "--json",
                "tagName,name,body,isDraft,isPrerelease",
            ],
            allow_failure=True,
        )
        if completed.returncode:
            detail = (completed.stderr or "").lower()
            if "not found" in detail or "404" in detail:
                return None
            raise ReleaseError(f"could not inspect GitHub Release for {tag}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ReleaseError("GitHub Release response was not valid JSON") from error

    def create_release(
        self, repository: str, tag: str, title: str, notes: str
    ) -> None:
        self._run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                repository,
                "--verify-tag",
                "--title",
                title,
                "--notes",
                notes,
            ]
        )


def _string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseError(f"{description} must be a non-empty string")
    return value.strip()


def _inside(root: Path, relative: str, description: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ReleaseError(f"{description} escapes the repository") from error
    return candidate


def _unknown_fields(document: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ReleaseError(f"{where} has unknown field(s): {', '.join(unknown)}")


def _load_candidates(root: Path, paths: list[str]) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    if not paths:
        raise ReleaseError("no checked releases/*.toml manifests exist")
    for relative in paths:
        path = _inside(root, relative, "release manifest path")
        if not path.is_file():
            raise ReleaseError(f"checked release manifest is missing: {relative}")
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise ReleaseError(f"malformed release manifest {relative}: {error}") from error
        if not isinstance(document, dict):
            raise ReleaseError(f"release manifest {relative} is not a TOML table")
        candidates.append((relative, document))
    return candidates


def _select_manifest(
    candidates: list[tuple[str, dict[str, Any]]],
    repository: str,
    issue: int,
) -> tuple[str, dict[str, Any]]:
    matches = [
        item
        for item in candidates
        if item[1].get("repository") == repository
        and item[1].get("issue") == issue
    ]
    if not matches:
        raise ReleaseError("no checked release manifest matches repository and issue")
    if len(matches) != 1:
        raise ReleaseError("multiple checked release manifests match repository and issue")
    return matches[0]


def _validate_issue(
    issue: dict[str, Any],
    repository: str,
    number: int,
    head: str,
    manifest_sha256: str,
    *,
    require_approval: bool = True,
) -> None:
    expected_url = f"https://github.com/{repository}/issues/{number}"
    labels = {
        label.get("name") for label in issue.get("labels", []) if isinstance(label, dict)
    }
    if issue.get("number") != number or issue.get("url") != expected_url:
        raise ReleaseError("GitHub issue does not match the destination-local authorization")
    if issue.get("state") != "OPEN":
        raise ReleaseError("destination-local release issue must be open")
    if require_approval and AUTHORIZATION_LABEL not in labels:
        raise ReleaseError(f"destination-local release issue lacks {AUTHORIZATION_LABEL}")
    body_lines = str(issue.get("body") or "").splitlines()
    head_authorization = f"Release control head SHA: {head}"
    head_authorizations = [
        line for line in body_lines if line.startswith("Release control head SHA: ")
    ]
    if head_authorizations != [head_authorization]:
        raise ReleaseError(
            "destination-local issue does not authorize the exact release control head"
        )
    manifest_authorization = f"Release manifest SHA-256: {manifest_sha256}"
    manifest_authorizations = [
        line for line in body_lines if line.startswith("Release manifest SHA-256: ")
    ]
    if manifest_authorizations != [manifest_authorization]:
        raise ReleaseError("destination-local issue does not authorize the exact manifest digest")


def _validate_registry_record(
    record: dict[str, Any], name: str, version: str, checksum: str | None = None
) -> None:
    record_name = record.get("crate") or record.get("name")
    if (
        record_name != name
        or record.get("num") != version
        or record.get("yanked") is not False
        or (checksum is not None and record.get("checksum") != checksum)
    ):
        raise ReleaseError(f"registry conflict for {name} {version}")


def _registry_checksum(record: dict[str, Any], name: str, version: str) -> str:
    """Return the immutable checksum from one valid crates.io version record."""

    _validate_registry_record(record, name, version)
    checksum = record.get("checksum")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ReleaseError(f"invalid registry checksum for {name} {version}")
    return checksum


def _expected_checksum(package: Mapping[str, Any], candidate_checksum: str) -> str:
    pinned = package.get("published_checksum")
    return pinned if isinstance(pinned, str) else candidate_checksum


def validate_manifest(
    root: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    ownership_document: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    _unknown_fields(manifest, ROOT_FIELDS, "release manifest")
    if manifest.get("schema_version") != 1:
        raise ReleaseError("release manifest schema_version must be 1")
    if manifest.get("repository") != EXPECTED_REPOSITORY:
        raise ReleaseError(f"release manifest repository must be {EXPECTED_REPOSITORY}")
    issue = manifest.get("issue")
    if not isinstance(issue, int) or isinstance(issue, bool) or issue < 1:
        raise ReleaseError("release manifest issue must be a positive integer")
    if manifest.get("registry") != REGISTRY:
        raise ReleaseError(f"release manifest registry must be {REGISTRY}")
    source_sha = manifest.get("source_sha")
    if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ReleaseError("release manifest source_sha must be a full lowercase commit SHA")
    repair_source_sha = manifest.get("repair_source_sha")
    if repair_source_sha is not None and (
        not isinstance(repair_source_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", repair_source_sha) is None
        or repair_source_sha == source_sha
    ):
        raise ReleaseError(
            "release manifest repair_source_sha must be a distinct full lowercase commit SHA"
        )
    config = tomllib.loads((root / ".agent-loop.toml").read_text(encoding="utf-8"))
    configured_checks = config.get("verification", {}).get("commands")
    if manifest.get("required_checks") != configured_checks:
        raise ReleaseError("required_checks must exactly match .agent-loop.toml")
    consumer_checks = manifest.get("required_consumer_checks")
    if not isinstance(consumer_checks, list) or not consumer_checks or any(
        not isinstance(command, str) or not command.strip() for command in consumer_checks
    ):
        raise ReleaseError("required_consumer_checks must be a string array")
    fast_continuation = manifest.get("fast_continuation", False)
    if not isinstance(fast_continuation, bool):
        raise ReleaseError("fast_continuation must be a boolean")

    raw_packages = manifest.get("packages")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ReleaseError("release manifest packages must be a non-empty array")
    packages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_packages):
        if not isinstance(raw, dict):
            raise ReleaseError(f"packages[{index}] must be a table")
        _unknown_fields(raw, PACKAGE_FIELDS, f"packages[{index}]")
        package = dict(raw)
        for field in ("name", "version", "owner", "manifest_path", "tag"):
            package[field] = _string(package.get(field), f"packages[{index}].{field}")
        if package["owner"] != EXPECTED_REPOSITORY:
            raise ReleaseError(f"{package['name']}: owner must be {EXPECTED_REPOSITORY}")
        if package["tag"] != f"{package['name']}-v{package['version']}":
            raise ReleaseError(f"{package['name']}: tag must be package-name-vversion")
        dependencies = package.get("dependencies")
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) or not item for item in dependencies
        ):
            raise ReleaseError(f"{package['name']}: dependencies must be a string array")
        package["dependencies"] = dependencies
        published_checksum = package.get("published_checksum")
        if published_checksum is not None and (
            not isinstance(published_checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", published_checksum) is None
        ):
            raise ReleaseError(
                f"{package['name']}: published_checksum must be a lowercase SHA-256 digest"
            )
        packages.append(package)

    names = [package["name"] for package in packages]
    tags = [package["tag"] for package in packages]
    if len(names) != len(set(names)) or len(tags) != len(set(tags)):
        raise ReleaseError("release manifest contains duplicate package names or tags")
    if manifest.get("dependency_order") != names:
        raise ReleaseError("dependency_order must exactly match packages array order")
    if manifest.get("expected_tags") != tags:
        raise ReleaseError("expected_tags must exactly match package tags")

    if ownership_document is None:
        ownership_file = root / OWNERSHIP_PATH
        try:
            ownership_document = json.loads(ownership_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReleaseError(f"cannot load package ownership: {error}") from error
    owned = {
        record.get("current_package_name"): record
        for record in ownership_document.get("packages", [])
        if isinstance(record, dict)
    }
    cargo_packages = {
        package.get("name"): package
        for package in metadata.get("packages", [])
        if isinstance(package, dict)
    }
    positions = {name: index for index, name in enumerate(names)}
    selected = set(names)
    prerequisite_names: set[str] = set()
    for package in packages:
        name = package["name"]
        ownership = owned.get(name)
        cargo = cargo_packages.get(name)
        if ownership is None or cargo is None:
            raise ReleaseError(f"{name}: package is not owned by this Cargo workspace")
        if (
            ownership.get("intended_next_release_owner") != EXPECTED_REPOSITORY
            or ownership.get("manifest_path") != package["manifest_path"]
            or ownership.get("publication_class") != REGISTRY
            or ownership.get("automatic_publish_eligible") is not True
        ):
            raise ReleaseError(f"{name}: package ownership does not authorize this release")
        actual_manifest = Path(_string(cargo.get("manifest_path"), f"{name} Cargo manifest"))
        expected_manifest = _inside(root, package["manifest_path"], f"{name} manifest_path")
        if actual_manifest.resolve() != expected_manifest or not expected_manifest.is_file():
            raise ReleaseError(f"{name}: manifest_path does not match Cargo metadata")
        try:
            package_manifest = tomllib.loads(
                expected_manifest.read_text(encoding="utf-8")
            ).get("package", {})
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise ReleaseError(f"{name}: cannot read Cargo manifest: {error}") from error
        if package_manifest.get("version") != package["version"]:
            raise ReleaseError(
                f"{name}: Cargo manifest must declare its explicit version "
                f"{package['version']}"
            )
        if cargo.get("version") != package["version"]:
            raise ReleaseError(f"{name}: version does not match Cargo metadata")
        publish = cargo.get("publish")
        if publish == [] or (isinstance(publish, list) and "crates-io" not in publish):
            raise ReleaseError(f"{name}: Cargo manifest does not permit crates.io publication")
        normal_dependencies = [
            dependency
            for dependency in cargo.get("dependencies", [])
            if isinstance(dependency, dict)
            and dependency.get("kind") != "dev"
        ]
        for dependency in normal_dependencies:
            dependency_name = dependency.get("name")
            source = dependency.get("source")
            if isinstance(source, str) and source.startswith("git+"):
                raise ReleaseError(
                    f"{name}: Git dependency {dependency_name} is not publishable"
                )
            dependency_path = dependency.get("path")
            if dependency_path is None:
                continue
            resolved_dependency = Path(dependency_path).resolve()
            try:
                resolved_dependency.relative_to(root.resolve())
            except ValueError as error:
                raise ReleaseError(
                    f"{name}: path dependency {dependency_name} escapes the repository"
                ) from error
            dependency_package = cargo_packages.get(dependency_name)
            if dependency_package is None:
                raise ReleaseError(
                    f"{name}: path dependency {dependency_name} is not an owned workspace package"
                )
            dependency_manifest = Path(
                _string(
                    dependency_package.get("manifest_path"),
                    f"{dependency_name} Cargo manifest",
                )
            )
            if resolved_dependency != dependency_manifest.resolve().parent:
                raise ReleaseError(
                    f"{name}: path dependency {dependency_name} does not match Cargo metadata"
                )
            dependency_version = dependency_package.get("version")
            if dependency.get("req") != f"={dependency_version}":
                raise ReleaseError(
                    f"{name}: {dependency_name} must use the exact version requirement "
                    f"={dependency_version}"
                )

        workspace_dependencies = [
            dependency
            for dependency in normal_dependencies
            if dependency.get("name") in cargo_packages
        ]
        actual_dependencies = {
            dependency.get("name") for dependency in workspace_dependencies
        }
        if set(package["dependencies"]) != actual_dependencies:
            raise ReleaseError(f"{name}: dependencies do not match Cargo metadata")
        for dependency in package["dependencies"]:
            if dependency not in selected:
                prerequisite_names.add(dependency)
            elif positions[dependency] >= positions[name]:
                raise ReleaseError(f"wrong dependency order: {dependency} must precede {name}")

    raw_releases = manifest.get("github_releases", [])
    if not isinstance(raw_releases, list):
        raise ReleaseError("github_releases must be an array")
    releases: list[dict[str, str]] = []
    release_tags: set[str] = set()
    for index, raw in enumerate(raw_releases):
        if not isinstance(raw, dict):
            raise ReleaseError(f"github_releases[{index}] must be a table")
        _unknown_fields(raw, RELEASE_FIELDS, f"github_releases[{index}]")
        release = {
            field: _string(raw.get(field), f"github_releases[{index}].{field}")
            for field in ("tag", "title", "notes")
        }
        if release["tag"] not in tags or release["tag"] in release_tags:
            raise ReleaseError("GitHub Releases must reference unique manifest-declared tags")
        release_tags.add(release["tag"])
        releases.append(release)
    prerequisites = [
        {
            "name": name,
            "version": _string(cargo_packages[name].get("version"), f"{name} version"),
            "manifest_path": _string(
                cargo_packages[name].get("manifest_path"), f"{name} Cargo manifest"
            ),
        }
        for name in sorted(prerequisite_names)
    ]
    return packages, releases, prerequisites


def _revalidate_exact_checkout(
    root: Path,
    effects: Effects,
    repository: str,
    head: str,
    control_source_sha: str,
    manifest_path: str,
    manifest_digest: str,
) -> None:
    if effects.repository() != repository or effects.head() != head or not effects.clean():
        raise ReleaseError("publication checkout changed after validation")
    if manifest_path not in effects.tracked_manifests():
        raise ReleaseError("selected release manifest is no longer checked at the exact head")
    if effects.changed_paths(control_source_sha, head) != [manifest_path]:
        raise ReleaseError(
            "exact head no longer differs from its control source only by its manifest"
        )
    if hashlib.sha256((root / manifest_path).read_bytes()).hexdigest() != manifest_digest:
        raise ReleaseError("selected release manifest changed after authorization")


def _revalidate_authority(
    root: Path,
    effects: Effects,
    repository: str,
    issue_number: int,
    head: str,
    control_source_sha: str,
    manifest_path: str,
    manifest_digest: str,
    *,
    require_approval: bool = True,
) -> None:
    """Refresh checkout and destination-local issue authority before an effect."""

    _revalidate_exact_checkout(
        root,
        effects,
        repository,
        head,
        control_source_sha,
        manifest_path,
        manifest_digest,
    )
    _validate_issue(
        effects.issue(repository, issue_number),
        repository,
        issue_number,
        head,
        manifest_digest,
        require_approval=require_approval,
    )


def _registry_prefix(
    effects: Effects,
    packages: list[dict[str, Any]],
    checksums: list[str],
) -> list[bool]:
    """Read and validate one fresh dependency-ordered registry snapshot."""

    present: list[bool] = []
    for package, checksum in zip(packages, checksums):
        record = effects.registry_version(package["name"], package["version"])
        if record is not None:
            _validate_registry_record(
                record,
                package["name"],
                package["version"],
                _expected_checksum(package, checksum) if checksum else None,
            )
        present.append(record is not None)
    first_absent = next(
        (index for index, is_present in enumerate(present) if not is_present),
        len(present),
    )
    if any(present[first_absent:]):
        raise ReleaseError("registry state is not a published prefix in dependency order")
    return present


def _registry_package(
    effects: Effects,
    package: dict[str, Any],
    checksum: str,
) -> None:
    """Freshly verify the one immutable registry record guarding an effect."""

    record = effects.registry_version(package["name"], package["version"])
    if record is None:
        raise ReleaseError(
            f"registry version missing for {package['name']} {package['version']}"
        )
    _validate_registry_record(
        record,
        package["name"],
        package["version"],
        _expected_checksum(package, checksum),
    )


def _tag_state(
    effects: Effects,
    packages: list[dict[str, Any]],
    checksums: list[str],
    package_index: int,
    source_sha: str,
    *,
    verify_registry: bool = True,
) -> tuple[str | None, str | None]:
    package = packages[package_index]
    if verify_registry:
        _registry_package(effects, package, checksums[package_index])
    tag = package["tag"]
    local = effects.local_tag_target(tag)
    remote = effects.remote_tag_target(tag)
    for target in (local, remote):
        if target is not None and target != source_sha:
            raise ReleaseError(f"immutable tag conflict for {tag}")
    return local, remote


def _validate_existing_release(
    existing: dict[str, Any], release: dict[str, str]
) -> None:
    if (
        existing.get("tagName") != release["tag"]
        or existing.get("name") != release["title"]
        or existing.get("body") != release["notes"]
        or existing.get("isDraft") is not False
        or existing.get("isPrerelease") is not False
    ):
        raise ReleaseError(f"GitHub Release conflict for {release['tag']}")


def _release_state(
    effects: Effects,
    repository: str,
    packages: list[dict[str, Any]],
    checksums: list[str],
    package_index: int,
    source_sha: str,
    release: dict[str, str] | None,
    *,
    verify_registry: bool = True,
) -> dict[str, Any] | None:
    package = packages[package_index]
    if verify_registry:
        _registry_package(effects, package, checksums[package_index])
    tag = package["tag"]
    if effects.remote_tag_target(tag) != source_sha:
        raise ReleaseError(f"remote tag conflict for GitHub Release {tag}")
    existing = effects.release(repository, tag)
    if existing is not None:
        if release is None:
            raise ReleaseError(f"undeclared GitHub Release exists for {tag}")
        _validate_existing_release(existing, release)
    return existing


def run_release(
    root: Path, environment: Mapping[str, str], effects: Effects
) -> dict[str, Any]:
    root = root.resolve()
    required = ("AGENT_LOOP_REPOSITORY", "AGENT_LOOP_ISSUE", "AGENT_LOOP_HEAD_SHA")
    if any(not environment.get(name, "").strip() for name in required):
        raise ReleaseError(
            "AGENT_LOOP_REPOSITORY, AGENT_LOOP_ISSUE, and "
            "AGENT_LOOP_HEAD_SHA are required"
        )
    repository = environment["AGENT_LOOP_REPOSITORY"].strip()
    head = environment["AGENT_LOOP_HEAD_SHA"].strip()
    try:
        issue_number = int(environment["AGENT_LOOP_ISSUE"])
    except ValueError as error:
        raise ReleaseError("AGENT_LOOP_ISSUE must be a positive integer") from error
    if issue_number < 1:
        raise ReleaseError("AGENT_LOOP_ISSUE must be a positive integer")
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ReleaseError("AGENT_LOOP_HEAD_SHA must be a full lowercase commit SHA")
    if repository != EXPECTED_REPOSITORY or effects.repository() != repository:
        raise ReleaseError("publication repository is not the owned destination repository")
    if effects.head() != head:
        raise ReleaseError("publication checkout does not match AGENT_LOOP_HEAD_SHA")
    if not effects.clean():
        raise ReleaseError("publication checkout must be clean")

    candidates = _load_candidates(root, effects.tracked_manifests())
    manifest_path, manifest = _select_manifest(candidates, repository, issue_number)
    _unknown_fields(manifest, ROOT_FIELDS, "release manifest")
    source_sha = _string(manifest.get("source_sha"), "release manifest source_sha")
    repair_source = manifest.get("repair_source_sha")
    if repair_source is None:
        control_source_sha = source_sha
        if not effects.source_is_ancestor(source_sha, head):
            raise ReleaseError(
                "release manifest source_sha is not an ancestor of the exact head"
            )
    else:
        control_source_sha = _string(
            repair_source, "release manifest repair_source_sha"
        )
        if (
            re.fullmatch(r"[0-9a-f]{40}", control_source_sha) is None
            or control_source_sha == source_sha
            or not effects.source_is_ancestor(source_sha, control_source_sha)
            or not effects.source_is_ancestor(control_source_sha, head)
        ):
            raise ReleaseError(
                "release manifest control repair is not bound between source_sha and exact head"
            )
        repair_paths = set(effects.changed_paths(source_sha, control_source_sha))
        expected_repair_paths = CONTROL_REPAIR_SCRIPT_PATHS | {manifest_path}
        if repair_paths != expected_repair_paths:
            expected = ", ".join(sorted(expected_repair_paths))
            raise ReleaseError(
                "release manifest control repair changed paths outside the fixed repair "
                f"surface; expected exactly: {expected}"
            )
    if effects.changed_paths(control_source_sha, head) != [manifest_path]:
        raise ReleaseError(
            "exact head differs from its control source by more than its release manifest"
        )
    manifest_digest = hashlib.sha256((root / manifest_path).read_bytes()).hexdigest()
    safeguards_flag = environment.get(SAFEGUARDS_ONLY_ENV, "").strip()
    if safeguards_flag not in ("", "0", "1"):
        raise ReleaseError(f"{SAFEGUARDS_ONLY_ENV} must be 1 when enabled")
    safeguards_only = safeguards_flag == "1"
    issue = effects.issue(repository, issue_number)
    _validate_issue(
        issue,
        repository,
        issue_number,
        head,
        manifest_digest,
        require_approval=not safeguards_only,
    )
    packages, releases, prerequisites = validate_manifest(
        root, manifest, effects.cargo_metadata()
    )
    fast_flag = environment.get(FAST_CONTINUATION_ENV, "").strip()
    if fast_flag not in ("", "0", "1"):
        raise ReleaseError(f"{FAST_CONTINUATION_ENV} must be 1 when enabled")
    fast_continuation = manifest.get("fast_continuation") is True
    if fast_continuation != (fast_flag == "1"):
        raise ReleaseError(
            "fast continuation requires both manifest fast_continuation = true and "
            f"{FAST_CONTINUATION_ENV}=1"
        )
    if safeguards_only and not fast_continuation:
        raise ReleaseError("safeguards-only audit requires fast continuation")

    for prerequisite in prerequisites:
        record = effects.registry_version(prerequisite["name"], prerequisite["version"])
        if record is None:
            raise ReleaseError(
                f"release prerequisite is not registry-visible: "
                f"{prerequisite['name']} {prerequisite['version']}"
            )
        _validate_registry_record(
            record,
            prerequisite["name"],
            prerequisite["version"],
        )

    registry_records: list[dict[str, Any] | None] = []
    local_tags: dict[str, str | None] = {}
    remote_tags: dict[str, str | None] = {}
    for package in packages:
        record = effects.registry_version(package["name"], package["version"])
        if record is not None:
            _validate_registry_record(record, package["name"], package["version"])
        registry_records.append(record)
        local_tags[package["tag"]] = effects.local_tag_target(package["tag"])
        remote_tags[package["tag"]] = effects.remote_tag_target(package["tag"])
        for target in (local_tags[package["tag"]], remote_tags[package["tag"]]):
            if target is not None and target != source_sha:
                raise ReleaseError(f"immutable tag conflict for {package['tag']}")
        if record is None and (
            local_tags[package["tag"]] is not None
            or remote_tags[package["tag"]] is not None
        ):
            raise ReleaseError(f"tag exists before registry version for {package['name']}")

    registry_present = [record is not None for record in registry_records]
    for package, record in zip(packages, registry_records):
        if package.get("published_checksum") is not None and record is None:
            raise ReleaseError(
                f"{package['name']}: published_checksum requires a registry-visible version"
            )
    first_absent = next(
        (index for index, present in enumerate(registry_present) if not present),
        len(packages),
    )
    if any(registry_present[first_absent:]):
        raise ReleaseError("registry state is not a published prefix in dependency order")

    releases_by_tag = {release["tag"]: release for release in releases}
    for index, package in enumerate(packages):
        tag = package["tag"]
        existing = effects.release(repository, tag)
        release = releases_by_tag.get(tag)
        if existing is not None and release is None:
            raise ReleaseError(f"undeclared GitHub Release exists for {tag}")
        if release is None:
            continue
        if existing is not None:
            if registry_records[index] is None:
                raise ReleaseError(f"GitHub Release exists before registry version for {tag}")
            if remote_tags[tag] is None:
                raise ReleaseError(f"GitHub Release exists without its manifest tag: {tag}")
            _validate_existing_release(existing, release)

    if not fast_continuation:
        for command in manifest["required_consumer_checks"]:
            effects.verify(command)

    # The default policy packages the complete candidate closure before a side
    # effect. Explicit fast continuation retains exact authority and registry
    # safeguards but packages only each next-absent crate immediately before
    # its publish attempt.
    patches = {
        package["name"]: str(
            _inside(root, package["manifest_path"], "package manifest").parent
        )
        for package in packages
    }
    patches.update(
        {
            prerequisite["name"]: str(Path(prerequisite["manifest_path"]).parent)
            for prerequisite in prerequisites
        }
    )
    if not fast_continuation:
        for package in packages:
            effects.package(package["name"], package["version"], patches)
    _revalidate_exact_checkout(
        root,
        effects,
        repository,
        head,
        control_source_sha,
        manifest_path,
        manifest_digest,
    )
    checksums: list[str] = ["" for _ in packages]
    for index, (package, record) in enumerate(zip(packages, registry_records)):
        if record is None:
            continue
        pinned = package.get("published_checksum")
        if not isinstance(pinned, str):
            raise ReleaseError(
                f"{package['name']}: registry-visible version requires "
                "published_checksum in the release manifest"
            )
        checksums[index] = pinned
    present = _registry_prefix(effects, packages, checksums)

    if safeguards_only:
        next_index = next(
            (index for index, is_present in enumerate(present) if not is_present),
            len(packages),
        )
        next_package: dict[str, str] | None = None
        if next_index < len(packages):
            package = packages[next_index]
            _revalidate_authority(
                root,
                effects,
                repository,
                issue_number,
                head,
                control_source_sha,
                manifest_path,
                manifest_digest,
                require_approval=False,
            )
            if effects.registry_version(package["name"], package["version"]) is not None:
                raise ReleaseError(
                    f"{package['name']}: registry version appeared before safeguards package"
                )
            effects.package(package["name"], package["version"], patches)
            _revalidate_authority(
                root,
                effects,
                repository,
                issue_number,
                head,
                control_source_sha,
                manifest_path,
                manifest_digest,
                require_approval=False,
            )
            if effects.registry_version(package["name"], package["version"]) is not None:
                raise ReleaseError(
                    f"{package['name']}: registry version appeared during safeguards package"
                )
            next_package = {"name": package["name"], "version": package["version"]}
        return {
            "schemaVersion": 1,
            "repository": repository,
            "issue": issue_number,
            "head": head,
            "manifest": manifest_path,
            "mode": "safeguards-only",
            "registryPrefixLength": next_index,
            "nextPackage": next_package,
            "approvalRequiredForPublication": True,
        }

    package_results: list[dict[str, str]] = []
    for index, package in enumerate(packages):
        status = "registry-verified"
        if not present[index]:
            _revalidate_authority(
                root,
                effects,
                repository,
                issue_number,
                head,
                control_source_sha,
                manifest_path,
                manifest_digest,
            )
            record = effects.registry_version(package["name"], package["version"])
            if record is not None:
                _validate_registry_record(
                    record, package["name"], package["version"]
                )
                raise ReleaseError(
                    f"{package['name']}: registry version appeared before publish; "
                    "pin its published_checksum in the release manifest and resume"
                )
            if fast_continuation:
                effects.package(package["name"], package["version"], patches)
                _revalidate_authority(
                    root,
                    effects,
                    repository,
                    issue_number,
                    head,
                    control_source_sha,
                    manifest_path,
                    manifest_digest,
                )
                record = effects.registry_version(package["name"], package["version"])
                if record is not None:
                    _validate_registry_record(record, package["name"], package["version"])
                    raise ReleaseError(
                        f"{package['name']}: registry version appeared during packaging; "
                        "pin its published_checksum in the release manifest and resume"
                    )
            _revalidate_authority(
                root,
                effects,
                repository,
                issue_number,
                head,
                control_source_sha,
                manifest_path,
                manifest_digest,
            )
            try:
                effects.publish(package["name"])
            except Exception as error:
                _revalidate_authority(
                    root,
                    effects,
                    repository,
                    issue_number,
                    head,
                    control_source_sha,
                    manifest_path,
                    manifest_digest,
                )
                record = effects.registry_version(
                    package["name"], package["version"]
                )
                if record is not None:
                    _validate_registry_record(
                        record, package["name"], package["version"]
                    )
                    raise ReleaseError(
                        f"{package['name']}: registry version appeared after an "
                        "ambiguous publish failure; pin its published_checksum in "
                        "the release manifest and resume"
                    ) from error
                raise _effect_failure(f"publish {package['name']}", error) from error
            record = None
            for _ in range(12):
                record = effects.registry_version(package["name"], package["version"])
                if record is not None:
                    checksums[index] = _registry_checksum(
                        record, package["name"], package["version"]
                    )
                    break
                effects.wait_for_registry()
            if record is None:
                raise ReleaseError(
                    f"published {package['name']} {package['version']} is not visible on crates.io"
                )
            status = "published-and-verified"
            present[index] = True
        package_results.append({"name": package["name"], "status": status})

    if not all(present):
        present = _registry_prefix(effects, packages, checksums)
    if not all(present):
        raise ReleaseError("all registry versions must be verified before tags")
    tag_results: list[dict[str, str]] = []
    for index, package in enumerate(packages):
        tag = package["tag"]
        status = "existing"
        local, remote = _tag_state(
            effects,
            packages,
            checksums,
            index,
            source_sha,
            verify_registry=False,
        )
        if remote is not None:
            tag_results.append({"tag": tag, "status": status})
            continue
        if local is None:
            _revalidate_authority(
                root,
                effects,
                repository,
                issue_number,
                head,
                control_source_sha,
                manifest_path,
                manifest_digest,
            )
            local, remote = _tag_state(effects, packages, checksums, index, source_sha)
            if remote is not None:
                tag_results.append({"tag": tag, "status": status})
                continue
            if local is None:
                _revalidate_authority(
                    root,
                    effects,
                    repository,
                    issue_number,
                    head,
                    control_source_sha,
                    manifest_path,
                    manifest_digest,
                )
                try:
                    effects.create_tag(
                        tag,
                        source_sha,
                        f"Release {package['name']} {package['version']}",
                    )
                except Exception as error:
                    _revalidate_authority(
                        root,
                        effects,
                        repository,
                        issue_number,
                        head,
                        control_source_sha,
                        manifest_path,
                        manifest_digest,
                    )
                    local, remote = _tag_state(
                        effects, packages, checksums, index, source_sha
                    )
                    if local != source_sha and remote != source_sha:
                        raise _effect_failure(f"create tag {tag}", error) from error
                else:
                    status = "created"
            local, remote = _tag_state(effects, packages, checksums, index, source_sha)
            if remote is None and local != source_sha:
                raise ReleaseError(f"local tag verification failed for {tag}")
        if remote is None:
            _revalidate_authority(
                root,
                effects,
                repository,
                issue_number,
                head,
                control_source_sha,
                manifest_path,
                manifest_digest,
            )
            local, remote = _tag_state(effects, packages, checksums, index, source_sha)
            if remote is None:
                _revalidate_authority(
                    root,
                    effects,
                    repository,
                    issue_number,
                    head,
                    control_source_sha,
                    manifest_path,
                    manifest_digest,
                )
                try:
                    effects.push_tag(tag)
                except Exception as error:
                    _revalidate_authority(
                        root,
                        effects,
                        repository,
                        issue_number,
                        head,
                        control_source_sha,
                        manifest_path,
                        manifest_digest,
                    )
                    _, remote = _tag_state(effects, packages, checksums, index, source_sha)
                    if remote != source_sha:
                        raise _effect_failure(f"push tag {tag}", error) from error
                else:
                    status = "created-and-pushed" if status == "created" else "pushed"
        _, remote = _tag_state(effects, packages, checksums, index, source_sha)
        if remote != source_sha:
            raise ReleaseError(f"remote tag verification failed for {tag}")
        tag_results.append({"tag": tag, "status": status})

    if not all(_registry_prefix(effects, packages, checksums)):
        raise ReleaseError("all registry versions must be verified before GitHub Releases")
    release_results: list[dict[str, str]] = []
    for index, package in enumerate(packages):
        release = releases_by_tag.get(package["tag"])
        existing = _release_state(
            effects,
            repository,
            packages,
            checksums,
            index,
            source_sha,
            release,
            verify_registry=False,
        )
        if release is None:
            continue
        status = "existing"
        if existing is None:
            _revalidate_authority(
                root,
                effects,
                repository,
                issue_number,
                head,
                control_source_sha,
                manifest_path,
                manifest_digest,
            )
            existing = _release_state(
                effects,
                repository,
                packages,
                checksums,
                index,
                source_sha,
                release,
            )
            if existing is not None:
                release_results.append({"tag": release["tag"], "status": status})
                continue
            _revalidate_authority(
                root,
                effects,
                repository,
                issue_number,
                head,
                control_source_sha,
                manifest_path,
                manifest_digest,
            )
            try:
                effects.create_release(
                    repository, release["tag"], release["title"], release["notes"]
                )
            except Exception as error:
                _revalidate_authority(
                    root,
                    effects,
                    repository,
                    issue_number,
                    head,
                    control_source_sha,
                    manifest_path,
                    manifest_digest,
                )
                existing = _release_state(
                    effects,
                    repository,
                    packages,
                    checksums,
                    index,
                    source_sha,
                    release,
                )
                if existing is None:
                    raise _effect_failure(
                        f"create GitHub Release {release['tag']}", error
                    ) from error
                release_results.append({"tag": release["tag"], "status": status})
                continue
            created = _release_state(
                effects,
                repository,
                packages,
                checksums,
                index,
                source_sha,
                release,
            )
            if created is None:
                raise ReleaseError(f"GitHub Release verification failed for {release['tag']}")
            status = "created-and-verified"
        release_results.append({"tag": release["tag"], "status": status})

    return {
        "schemaVersion": 1,
        "repository": repository,
        "issue": issue_number,
        "head": head,
        "manifest": manifest_path,
        "packages": package_results,
        "tags": tag_results,
        "githubReleases": release_results,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-safeguards", action="store_true")
    args = parser.parse_args()
    effects = CommandEffects(root)
    environment = dict(os.environ)
    if args.check_safeguards:
        environment.update(
            {
                "AGENT_LOOP_REPOSITORY": EXPECTED_REPOSITORY,
                "AGENT_LOOP_ISSUE": "2",
                "AGENT_LOOP_HEAD_SHA": effects.head(),
                FAST_CONTINUATION_ENV: "1",
                SAFEGUARDS_ONLY_ENV: "1",
            }
        )
    try:
        payload = run_release(root, environment, effects)
    except ReleaseError as error:
        print(f"release refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
