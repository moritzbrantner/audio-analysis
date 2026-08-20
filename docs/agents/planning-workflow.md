# Planning workflow

Substantial work starts as a GitHub PRD issue with acceptance criteria and
out-of-scope boundaries. Implementation slices use canonical YAML frontmatter:

```yaml
---
parent: 123
blocked_by: []
scope:
  - crates/example/**
---
```

Parallel slices require resolved blockers and disjoint concrete scopes. Tiny
changes may be implemented directly when a maintainer explicitly requests it.
