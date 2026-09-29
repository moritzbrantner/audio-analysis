# Triage labels

The canonical issue labels are `bug`, `enhancement`, `needs-triage`,
`needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`, and `prd`.

Agent Loop state uses `agent-loop:active`, `agent-loop:blocked`, and
`agent-loop:ready-to-merge`. Closed issues and merged pull requests are the
completion record. A future publication label must be treated as
security-sensitive and scoped to an exact destination-local release issue.

## Queue-ready issue metadata

`ready-for-agent` authorizes the sequential GitHub queue only when the issue
body starts with agent-loop metadata. `scope` is mandatory: it declares the
paths or behavioral surfaces that one implementation run may own. Do not infer
or widen this scope from the issue prose. `blocked_by` is optional and keeps an
issue out of the queue until each referenced issue is closed.

```yaml
---
scope:
  - crates/audio/audio-analysis-transcription/**
blocked_by:
  - 123
---
```

The queue considers eligible issues in ascending issue-number order, creates
and checks one pull request at a time, squash-merges it, and then selects the
next eligible issue. Use the **Agent-ready implementation** issue template so
the required metadata is present before applying `ready-for-agent`.
