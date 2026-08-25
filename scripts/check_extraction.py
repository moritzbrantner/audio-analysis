#!/usr/bin/env python3
"""Validate the restructuring-only audio extraction contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OWNERSHIP = ROOT / "docs/repository-split/package-ownership.json"
ADAPTATIONS = ROOT / "docs/repository-split/copy-adaptations.json"
IDENTITY = ROOT / "docs/repository-split/byte-identity.json"
EXPECTED_DIGEST = "b3e231c734b8615c524b012971458ea1370997c20bc2c57ced41934e6af317fc"
EXPECTED_EXTERNAL = {
    "audio-contracts": ("moenarch-audio-contracts", "=0.1.0"),
    "data-inversion-core": ("moenarch-data-inversion-core", "=0.1.1"),
    "jobs-core": ("moenarch-jobs-core", "=0.1.2"),
    "math-signal-core": ("moenarch-math-signal-core", "=0.1.1"),
    "media-core": ("moenarch-media-core", "=0.1.0"),
    "model-runtime": ("moenarch-model-runtime", "=0.1.1"),
    "runtime-core": ("moenarch-runtime-core", "=0.2.1"),
    "runtime-onnx": ("moenarch-runtime-onnx", "=0.1.1"),
    "tensor-data": ("moenarch-tensor-data", "=0.1.1"),
    "text-model-runtime": ("moenarch-text-model-runtime", "=0.1.1"),
    "text-transcripts": ("moenarch-text-transcripts", "=0.1.3"),
}
FORBIDDEN = (
    "video-analysis-ffmpeg",
    "video_analysis_ffmpeg",
    "video-analysis-ingest",
    "video_analysis_ingest",
    "prototypes/web/video-analysis-web",
    "setup:colmap-video",
    "@moritzbrantner/video-analysis-ui",
    "../video-analysis-ui",
)
GENERATED_PARTS = {"target", "node_modules", "pkg", "dist", "coverage"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def package_tree(record: dict) -> str:
    return str(Path(record["manifest_path"]).parent)


def tree_digest(tree: Path) -> tuple[int, str]:
    records = []
    for path in sorted(tree.rglob("*"), key=lambda candidate: candidate.relative_to(tree).as_posix()):
        if not path.is_file() or any(part in GENERATED_PARTS for part in path.relative_to(tree).parts):
            continue
        records.append(
            {
                "path": path.relative_to(tree).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return len(records), hashlib.sha256(encoded).hexdigest()


def dependency_tables(manifest: dict):
    for name in ("dependencies", "dev-dependencies", "build-dependencies"):
        yield manifest.get(name, {})
    for target in manifest.get("target", {}).values():
        if isinstance(target, dict):
            for name in ("dependencies", "dev-dependencies", "build-dependencies"):
                yield target.get(name, {})


def main() -> int:
    errors: list[str] = []
    ownership = load_json(OWNERSHIP)
    packages = sorted(ownership.get("packages", []), key=lambda record: record.get("id", ""))
    encoded = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != EXPECTED_DIGEST or ownership.get("canonical_digest") != EXPECTED_DIGEST:
        errors.append(f"ownership digest differs: {digest}")

    cargo_records = [record for record in packages if record.get("ecosystem") == "cargo"]
    bun_records = [record for record in packages if record.get("ecosystem") == "bun"]
    if len(cargo_records) != 53 or len(bun_records) != 26:
        errors.append(f"expected 53 Cargo and 26 Bun records, got {len(cargo_records)} and {len(bun_records)}")
    for record in packages:
        if not (ROOT / record["manifest_path"]).is_file():
            errors.append(f"missing package manifest: {record['manifest_path']}")
    if (ROOT / "crates/audio/audio-contracts").exists():
        errors.append("foundation-owned crates/audio/audio-contracts must not be copied")

    actual_cargo = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "crates").rglob("Cargo.toml"))
    expected_cargo = sorted(record["manifest_path"] for record in cargo_records)
    if actual_cargo != expected_cargo:
        errors.append("Cargo manifest set differs from the 53 reviewed records")
    actual_bun = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "packages").glob("*/package.json")
        if path.parent.name != "audio-app-ui"
    )
    expected_bun = sorted(record["manifest_path"] for record in bun_records)
    if actual_bun != expected_bun:
        errors.append("Bun manifest set differs from the 26 reviewed records")

    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    workspace_dependencies = root_manifest["workspace"]["dependencies"]
    for alias, (package, version) in EXPECTED_EXTERNAL.items():
        spec = workspace_dependencies.get(alias)
        if not isinstance(spec, dict) or spec.get("package") != package or spec.get("version") != version:
            errors.append(f"external dependency {alias} must be exact {package} {version}")
        elif "path" in spec or "git" in spec:
            errors.append(f"external dependency {alias} must be registry-only")

    cargo_manifest_set = set(expected_cargo)
    for manifest_path in [ROOT / "Cargo.toml", *(ROOT / path for path in expected_cargo)]:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        tables = dependency_tables(manifest)
        if manifest_path == ROOT / "Cargo.toml":
            tables = [workspace_dependencies]
        for table in tables:
            for name, spec in table.items():
                if not isinstance(spec, dict):
                    continue
                if "git" in spec:
                    errors.append(f"moving Git dependency {name} in {manifest_path.relative_to(ROOT)}")
                if "path" not in spec:
                    continue
                target = (manifest_path.parent / spec["path"] / "Cargo.toml").resolve()
                try:
                    relative = target.relative_to(ROOT.resolve()).as_posix()
                except ValueError:
                    errors.append(f"cross-repository path dependency {name} in {manifest_path.relative_to(ROOT)}")
                    continue
                if relative not in cargo_manifest_set:
                    errors.append(f"path dependency {name} targets non-owned manifest {relative}")

    local_bun_names = {record["current_package_name"] for record in bun_records} | {"@moritzbrantner/audio-app-ui"}
    for manifest_path in [ROOT / record["manifest_path"] for record in bun_records]:
        manifest = load_json(manifest_path)
        for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            for name, spec in manifest.get(section, {}).items():
                if isinstance(spec, str) and (spec.startswith(("file:", "link:", "git+")) or "github.com" in spec):
                    errors.append(f"external Bun path/Git dependency {name} in {manifest_path.relative_to(ROOT)}")
                if isinstance(spec, str) and spec.startswith("workspace:") and name not in local_bun_names:
                    errors.append(f"workspace dependency {name} is not destination-owned")

    scan_paths = [ROOT / "Cargo.toml", ROOT / "crates", ROOT / "packages"]
    for base in scan_paths:
        paths = [base] if base.is_file() else (path for path in base.rglob("*") if path.is_file())
        for path in paths:
            if any(part in GENERATED_PARTS for part in path.relative_to(ROOT).parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for forbidden in FORBIDDEN:
                if forbidden in text:
                    errors.append(f"forbidden visual edge {forbidden} in {path.relative_to(ROOT)}")

    adaptations = set(load_json(ADAPTATIONS)["adapted_package_trees"])
    expected_trees = {package_tree(record) for record in packages}
    if not adaptations <= expected_trees:
        errors.append("adaptation inventory contains a non-source package tree")

    # byte-identity.json is extraction provenance: once a copied tree is
    # intentionally adapted in the destination, its original identity record
    # may remain as historical evidence. Every still-unadapted tree must keep
    # an identity record and must continue to match it byte-for-byte.
    identity = load_json(IDENTITY)
    identity_trees = set(identity.get("trees", {}))
    unadapted_trees = expected_trees - adaptations
    if identity_trees - expected_trees:
        errors.append("byte-identity inventory contains a non-source package tree")
    missing_identity = unadapted_trees - identity_trees
    if missing_identity:
        errors.append(
            "byte-identity inventory is missing unadapted trees: "
            + ", ".join(sorted(missing_identity))
        )
    for tree in sorted(unadapted_trees):
        expected = identity["trees"][tree]
        count, current = tree_digest(ROOT / tree)
        if count != expected["file_count"] or current != expected["digest"]:
            errors.append(f"unadapted tree differs from source: {tree}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "audio extraction structure valid: "
        f"{len(cargo_records)} Cargo, {len(bun_records)} Bun, "
        f"{len(unadapted_trees)} byte-identical trees"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
