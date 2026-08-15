# Audio-analysis invariants

## INV-001 — Focused audio adapters preserve their library contract

- Authority/source: issue:#115
- Affected surfaces: crates/audio/**, crates/bindings/**, packages/**
- Compatibility promise: Existing crate names, serialized shapes, and focused CLI/server/WASM/app adapters remain until a separately authorized semver decision.
- Required evidence: contract, behavioral, integration
- Sensitivity: required
- Risk dimensions: security=not-applicable:bootstrap-has-no-auth-surface; recovery=covered:INV-001; persistence=not-applicable:no-persistent-store; concurrency=covered:INV-001; migration=covered:INV-001; partial-failure=covered:INV-001; operational=covered:INV-001
