# Issue tracker: GitHub

GitHub Issues in `moritzbrantner/nlp-stack` are the durable work queue. PRDs are
parent issues; implementation slices carry canonical `parent`, `blocked_by`,
and `scope` YAML frontmatter.

Release authorization must come from an open issue in this same repository. An
issue in `rust-packages` may authorize extraction work but cannot authorize
publication from this checkout.
