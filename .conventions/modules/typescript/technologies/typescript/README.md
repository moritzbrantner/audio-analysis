# TypeScript conventions

## TS-002 — Model invalid states out of the type system

- Prefer types, especially discriminated unions, that make invalid combinations unrepresentable.

## TS-003 — Prefer type over interface

- Use type aliases instead of interfaces for TypeScript models and declarations.
- If interface-specific semantics such as declaration merging are genuinely required, use an explicit, narrowly scoped lint suppression rather than weakening the shared rule.
- Deterministic lint configuration: [`./TS-003.oxlint.json`](./TS-003.oxlint.json).

## TS-004 — Use strict compiler options that expose missing-state mistakes

- Enable `strict`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `noImplicitOverride`, and `noFallthroughCasesInSwitch` by default.
- Enable unused-local and unused-parameter diagnostics for implementation code; intentional callback/adapter parameters should use the language/tooling's explicit convention rather than weakening the project globally.
- Repositories may document a narrow compatibility exception for generated code or an incompatible external tool, but the normal state is strict.

## TS-005 — Handle closed variants exhaustively

- Switches over enums and discriminated unions owned by the application should be exhaustive so adding a new variant produces a deterministic failure until callers handle it.
- External protocol boundaries may use an explicit unknown/fallback branch when forward compatibility requires it.

## TS-006 — Do not leave promises unobserved

- Await or return promises by default.
- Detached asynchronous work must flow through an explicit abstraction that owns lifetime, cancellation, error handling, and observability; a bare `void promise` is not the normal escape hatch.
