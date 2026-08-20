# Project invariants

## INV-001 — Reviewed ownership is exact

- Requirement: Exactly 53 Cargo and 26 Bun source records match the canonical reviewed digest.
- Forbidden behavior: extra, omitted, renamed, or source-relocated package records.
- Authority/source: repo:docs/repository-split/package-ownership.json
- Required evidence: contract, static

## INV-002 — External capability seams are registry-only

- Requirement: Destination-owned dependencies remain local; foundation and NLP dependencies use exact registry coordinates.
- Forbidden behavior: sibling paths, Git dependencies, visual ingest/FFmpeg edges, or the broad UI workspace dependency.
- Authority/source: repo:CONTEXT.md
- Required evidence: contract, static

## INV-003 — Extraction remains reversible

- Requirement: Unadapted package trees remain byte-identical to the exact source and adapted trees are enumerated.
- Forbidden behavior: source removal, publication, undocumented copied-tree mutation, or generated output committed as source.
- Authority/source: repo:docs/PROVENANCE.md
- Required evidence: contract, static
