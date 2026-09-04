# Audio Inspector site

The static GitHub Pages site composes existing `audio-analysis` package surfaces instead of creating a new capability owner.

## Browser analysis

The public site decodes audio with the browser and runs the existing core, Fourier, pitch, and rhythm Rust packages through their WASM adapters. Whole-file signal statistics are computed directly over decoded PCM, while package-surface analyses use explicitly reported bounded windows to respect their existing request limits.

The two built-in examples are synthesized deterministically in the browser: a 440 Hz reference tone and a 120 BPM click track. No external media is required.

## Backend boundary

On GitHub Pages, backend access is disabled even when a `backend` query parameter is supplied. On localhost, the inspector probes `http://127.0.0.1:3000` by default; another base URL can be selected with `?backend=<base-url>`.

The current site only discovers and reports the local transcription capability. It does not upload the selected audio or trigger heavy model execution automatically. A later enrichment slice can add explicit user-triggered backend operations without changing the browser-local default.

## Build

Run:

```text
bash scripts/build-pages.sh
```

The script validates the site, builds the four WASM adapters, and assembles `_site/` with relative asset paths suitable for the repository Pages URL.

## Verification

Run the fast deterministic browser-contract tests without compiling WASM:

```text
bun run test:pages
```

Run the Pages artifact gate, including the WASM build and deployable-artifact checks:

```text
bun run check:pages
```

Run the real-browser integration smoke test after `_site/` exists:

```text
bun run test:pages:e2e
```

The E2E runner serves the built artifact locally, installs the pinned Playwright 1.62.1 toolchain in an isolated temporary directory, and launches headless Chromium. It verifies the generated 440 Hz example through WebAudio decoding, real core/Fourier/pitch WASM execution, rendered report output, and JSON export. It then runs the generated 120 BPM click track through the real rhythm WASM path and verifies the resulting beat/tempo report.

`test:pages:e2e` is also exposed as the repository `test:e2e:smoke` capability. It stays separate from the cheaper `check:pages` gate so local and agent validation can progress from deterministic unit/contract checks to the heavier browser integration tier.

The GitHub Pages workflow runs both gates before it uploads or deploys `_site/`.
