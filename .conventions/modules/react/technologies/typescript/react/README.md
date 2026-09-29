# React conventions

## REACT-001 — Colocate components and directly related artifacts

- Keep a component and its focused tests, styles, hooks, and types in their smallest shared directory.

## REACT-002 — Keep React state local by default

- Own state in the smallest subtree that needs it; widen only for real shared ownership.

## REACT-003 — Put important navigational state in URL query parameters

- Put durable, shareable view state in query parameters; keep ephemeral and sensitive state out of URLs.

## REACT-004 — Use effects for external synchronization

- Use effects for systems outside React, not derived values or ordinary control flow.

## REACT-005 — Prefer composition over highly configurable mega-components

- Prefer focused composition over unrelated flags and modes.

## REACT-006 — Keep component boundaries structurally clear

- Names, directories, props, and immediate dependencies must make a component's purpose locally understandable.
- Split unrelated concerns at ownership boundaries; coordinator components are valid when coordination is their purpose.

## REACT-007 — Reuse shared UI before creating local primitives

- Inspect and reuse the established UI package before creating local primitives.
- Keep workflows and stateful page composition in the app; keep reusable state-light primitives and patterns shared.

## Child scopes

- [`nextjs/`](nextjs/)
- [`moritzbrantner-ui/`](moritzbrantner-ui/)
- [`tanstack-query/`](tanstack-query/)
- [`react-hook-form/`](react-hook-form/)
- [`zustand/`](zustand/)
- [`testing-library/`](testing-library/)

These are sibling specializations of React and may be composed together when a project uses several of them.
