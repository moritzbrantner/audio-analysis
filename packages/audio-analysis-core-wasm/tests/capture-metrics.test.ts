import { existsSync, readFileSync } from "node:fs";
import { expect, test } from "bun:test";

// The same fixtures the Rust library test runs, so TypeScript and Rust agree on identical audio.
const fixtures = JSON.parse(
  readFileSync(
    new URL(
      "../../../crates/audio/audio-analysis-core/tests/fixtures/capture-metrics.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as {
  tolerance: number;
  cases: Array<{
    name: string;
    sampleRate: number;
    channels: number;
    samples: number[];
    options?: Record<string, number>;
    expected: Record<string, number>;
  }>;
};
// The WASM build (`bun run build`) is not part of ordinary package tests.
const built = existsSync(new URL("../pkg/moenarch_audio_analysis_core_wasm.js", import.meta.url));

test.skipIf(!built)("captureMetrics reproduces the shared Rust fixtures", async () => {
  const { captureMetrics } = await import("../index.js");
  for (const fixture of fixtures.cases) {
    const metrics = (await captureMetrics(
      Float32Array.from(fixture.samples),
      fixture.sampleRate,
      fixture.channels,
      fixture.options,
    )) as Record<string, number>;
    for (const [key, expected] of Object.entries(fixture.expected)) {
      expect(Math.abs(metrics[key]! - expected), `${fixture.name}: ${key}`).toBeLessThanOrEqual(
        fixtures.tolerance,
      );
    }
  }
});

test.skipIf(!built)("captureMetrics rejects unknown options and invalid audio", async () => {
  const { captureMetrics } = await import("../index.js");
  await expect(captureMetrics([0], 16_000, 1, { clipLvl: 1 } as never)).rejects.toThrow();
  await expect(captureMetrics([0, 0, 0], 16_000, 2)).rejects.toThrow();
  await expect(captureMetrics([Number.NaN], 16_000, 1)).rejects.toThrow();
});

test("exports captureMetrics", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.captureMetrics).toBe("function");
});
