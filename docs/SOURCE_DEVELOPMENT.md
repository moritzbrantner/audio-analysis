# Audio source development

The audio capability repository is developed as source first and released separately.

Consumer-driven work should begin from a concrete Native WhisperX or application requirement. Change the smallest existing audio package surface that owns the behavior, then let the consumer pin the exact resulting source revision through its source-development declaration. Do not publish intermediate crate versions solely so the consumer can compile.

Package versions remain stable during normal source changes when compatibility permits. A later release task determines the minimal publication closure, performs version bumps, validates packages, publishes in dependency order, and proves registry-only consumer resolution.

This separation means an unreleased source graph may be valid development evidence. It does not weaken release requirements: registry publication and clean registry-only consumer checks still occur before distribution.

The current package count is not a target. Avoid adding independently versioned crates unless independent consumers or hard dependency boundaries justify them. Repeated co-change across packages is evidence for later consolidation rather than a reason to add more release surfaces.
