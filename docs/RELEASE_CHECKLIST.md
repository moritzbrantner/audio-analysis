# Native-whisperx audio contract release checklist

1. Review the six package versions and exact dependency order in issue #6.
2. Verify the source commit, followed by a control commit that changes only the
   checked release manifest.
3. Run every command declared in `.agent-loop.toml`.
4. Run the native-whisperx PR #230 candidate gate from the manifest.
5. Confirm every declared crate version and tag is absent.
6. Record the exact control head and manifest SHA-256 in issue #6.
7. Apply `release:approved` only after independent review.
8. Invoke the repository publisher from the exact clean control head.
9. Verify the registry prefix, immutable source tags, and consumer lockfile.
10. Remove approval and reconcile native-whisperx #230 and rust-packages #116.

This preparation PR performs none of steps 6–10.
