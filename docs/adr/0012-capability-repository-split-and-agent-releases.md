# ADR 0012: Capability Repository Split And Agent-Driven Releases

> Destination adaptation: this copy is authoritative only for the `nlp-stack`
> boundary. It retains the original ecosystem decision as historical rationale.
> Issue `rust-packages#113` performs clean-copy bootstrap only and authorizes no
> publication, source removal, consumer migration, tag, or release.

## Status

Accepted.

This ADR supersedes only the guidance in
[ADR 0011](0011-hybrid-geo-extraction-and-namespace.md) and the
[post-pilot playbook](../POST_PILOT_EXTRACTION_PLAYBOOK.md) that rejects
capability repositories for the primary media families. The geo clean-copy,
provenance, namespace, signpost, and source-removal lessons remain in force.

## Context

The 347-crate Rust workspace and its 173 Bun package surfaces impose a broad
verification and context cost on focused changes. Release ownership is unclear,
and applications consume a mixture of registry, path, Git, file, and shim
dependencies. A split is justified by maintainability, focused agent loops,
independent release cadence, explicit public-contract ownership, and smaller
verification surfaces—not checkout or build-cache size.

## Decision

Split by capability:

- `moritzbrantner/moenarch-foundation` owns domain-neutral runtime, job,
  progress, cancellation, diagnostic, artifact, model-lifecycle, media/time,
  data, math, tensor, graph, geometry, signal, and vector contracts.
- `moritzbrantner/nlp-stack` owns text, lexical/linguistic analysis,
  classification, embeddings, indexing, retrieval, QA, generation, and purified
  transcript documents.
- `moritzbrantner/audio-analysis` owns audio IO/analysis, recognition,
  separation, transcription execution, synthesis, MIDI, TTS, and native audio
  adapters.
- `moritzbrantner/visual-analysis` owns image/vision and non-spatial video
  contracts and implementations.
- `moritzbrantner/spatial-analysis` owns 3D, animation, posture, SFM, MVS,
  reconstruction, radiance fields, and Gaussian splatting.
- `moritzbrantner/rust-packages` becomes the compatibility facade, integration
  suite, incubator, migration-signpost home, cross-domain prototype home, and
  temporary ComfyUI owner.

The allowed production graph is:

```text
foundation
  ↑
  ├── nlp
  ├── audio ──→ narrowly scoped nlp contracts
  └── visual ─→ narrowly scoped nlp contracts
       ↑
       └── spatial

rust-packages compatibility/integration ─→ every released capability repository
```

Foundation depends on no target repository. NLP depends only on foundation.
Audio and visual may depend on foundation and narrow NLP contracts. Spatial may
depend on foundation and visual. Reverse edges and cycles are forbidden. The
machine ownership source, exact reviewed baseline, and checker under
`docs/repository-split/` and `scripts/check_repository_boundaries.py` enforce
this direction while the current monolith is neutralized.

### Neutral contracts and cycle breaking

No media-family source extraction begins until issue
[#108](https://github.com/moritzbrantner/rust-packages/issues/108) establishes
the domain-neutral media/time crate. The provisional `moenarch-media-core` name
must be checked against Cargo packages, crates.io, npm, and repositories. It may
own timebases, timestamps, time ranges, generic media/source metadata, neutral
events, and neutral source/stream traits. It must not own scenes, frames,
buffers, text documents, detections, keypoints, 2D geometry, or model-runtime
behavior.

Issue [#112](https://github.com/moritzbrantner/rust-packages/issues/112)
purifies `text-transcripts` around transcript/segment/timing/speaker and
SRT/WebVTT/Whisper JSON semantics. Audio decoding, transcription/VAD execution,
speaker models, FFmpeg, and downloads stay out. Audio transcription produces
those contracts.

Generic probing, finite-source selection, container metadata, and audio-track
decoding must not force audio applications through visual-analysis. Generic
source metadata belongs in foundation, visual frames in visual-analysis, and
audio sample preparation in audio-analysis. A narrow neutral IO/FFmpeg adapter
is permitted only where implementation behavior is genuinely shared.

### Names, semver, adapters, and provenance

Package names remain stable where practical. Repository movement alone is not a
breaking change. Patch releases cover compatible metadata or re-exports;
additive stable APIs use minor bumps; stable breaking APIs use major bumps at
1.x and minor bumps at 0.x. Renames require a new package plus a deliberately
versioned old-name deprecation release. One repository is the sole release
owner at every step.

Each extraction is a clean copy from an exact source commit. The destination
records every copied path, licenses, notices, attribution, and relevant history
notes and starts a focused history. History rewriting, force pushing, repository
deletion, and source removal before release proof are excluded.

Focused CLI, server, WASM, npm, and app adapters remain during initial
extraction. A repository-level registry may be piloted additively. Removing a
focused adapter requires usage evidence, migration notes, deployment analysis,
and a separate semver/release decision.

### Consumer and release gates

Before publication, destination code must pass independent clean-checkout
builds, its repository checks, `cargo package`, package-surface/operation-ID
parity, provenance review, and candidate consumer checks using temporary,
uncommitted patches. After publication, agents verify the exact registry
version, resolve it in a clean consumer without patches, run the narrow consumer
check, and create a repository-scoped update PR. Manifest inspection is never
reported as a passing consumer check.

Each publication wave is authorized by this ADR, its exact GitHub release issue,
and a reviewed machine release manifest. Agents may choose documented semver
bumps, open/merge ordinary release PRs when gates permit, publish, verify, tag,
create GitHub Releases, and open consumer PRs without another confirmation.
They may not use administrator bypasses, publish an unspecified package, or
publish a version absent from the authorization.

Agents may publish locally through Cargo's already-configured credential.
GitHub Actions/OIDC trusted publishing is an optional alternative, not a
prerequisite. Before either path, the validator fetches the exact live release
issue and binds its structured authorization to the repository, issue URL,
immutable source/base SHAs, required checks, and exact package versions. The
publisher packages and publishes topologically, verifies each registry version,
resumes idempotently from the first unpublished crate, and tags only
registry-verified versions. Credential values are never inspected, printed,
copied, placed in arguments, or logged.

If a wave partially publishes, already published versions remain immutable and
are neither republished nor automatically yanked. Record the partial state, fix
the remaining package/workflow in a follow-up commit, and resume at the first
unpublished version. Downstream constraints wait for the required closure.

### Source-removal and repository-creation gates

Source leaves `rust-packages` only after the destination is independently green,
release ownership is active, required crates are verified on the registry,
consumer migration is possible, compatibility signposts exist, the facade can
consume released crates, and rollback is documented. Source removal,
deprecation releases, and consumer migration are separate PRs unless an issue
proves the family tiny.

An agent may create a named target repository only when its issue specifies the
exact `moritzbrantner` repository and visibility and authenticated permissions
are sufficient. Unspecified visibility defaults private. Existing repository
visibility does not change under this ADR.

## Rollback

Before registry publication, close the extraction/release PR and retain active
ownership in `rust-packages`. After publication, published artifacts are not
deleted or yanked automatically: keep the last known-good source and
compatibility facade, stop further source removal and consumer updates, record
the exact released state, and ship a forward-compatible repair through a new
authorized release. A reverse migration of release ownership requires its own
ADR and issue.

## Consequences

Release cadence and verification become capability-scoped, while neutralization
and registry-first consumer proof add deliberate sequencing. Existing 49
forbidden edges are visible exceptions with individual owners and phases; no
wildcard exemption permits new coupling.
