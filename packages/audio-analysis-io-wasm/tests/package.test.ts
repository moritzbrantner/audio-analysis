import { afterEach, expect, test } from "bun:test";

const originalAudioContext = globalThis.AudioContext;
const originalOfflineAudioContext = globalThis.OfflineAudioContext;

afterEach(() => {
  globalThis.AudioContext = originalAudioContext;
  globalThis.OfflineAudioContext = originalOfflineAudioContext;
});

test("audio-analysis-io-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
  expect(typeof entry.browserAudioDecodeCapabilities).toBe("function");
  expect(typeof entry.supportsBrowserAudioDecode).toBe("function");
  expect(typeof entry.decodeBrowserAudioBlob).toBe("function");
});

test("browser audio decode reports one validated mono PCM snapshot", async () => {
  const entry = await import("../index.js");
  const calls = {
    arrayBuffer: 0,
    decode: 0,
    render: 0,
    close: 0,
  };

  class FakeAudioContext {
    state = "running";

    async decodeAudioData(_bytes: ArrayBuffer) {
      calls.decode += 1;
      return {
        duration: 0.00025,
        sampleRate: 48_000,
        numberOfChannels: 2,
      };
    }

    async close() {
      calls.close += 1;
      this.state = "closed";
    }
  }

  class FakeOfflineAudioContext {
    destination = {};
    length: number;
    sampleRate: number;

    constructor(channels: number, length: number, sampleRate: number) {
      expect(channels).toBe(1);
      this.length = length;
      this.sampleRate = sampleRate;
    }

    createBufferSource() {
      return {
        buffer: null,
        connect: (destination: unknown) => {
          expect(destination).toBe(this.destination);
        },
        start: (when: number) => {
          expect(when).toBe(0);
        },
      };
    }

    async startRendering() {
      calls.render += 1;
      const samples = Float32Array.from(
        { length: this.length },
        (_, index) => index / Math.max(1, this.sampleRate),
      );
      return {
        getChannelData: (channel: number) => {
          expect(channel).toBe(0);
          return samples;
        },
      };
    }
  }

  globalThis.AudioContext = FakeAudioContext as unknown as typeof AudioContext;
  globalThis.OfflineAudioContext =
    FakeOfflineAudioContext as unknown as typeof OfflineAudioContext;

  const source = {
    async arrayBuffer() {
      calls.arrayBuffer += 1;
      return new ArrayBuffer(16);
    },
  };

  const decoded = await entry.decodeBrowserAudioBlob(source, { sampleRateHz: 16_000 });

  expect(decoded.sampleRateHz).toBe(16_000);
  expect(decoded.channels).toBe(1);
  expect(decoded.sourceSampleRateHz).toBe(48_000);
  expect(decoded.sourceChannels).toBe(2);
  expect(decoded.samples).toBeInstanceOf(Float32Array);
  expect(decoded.samples.length).toBe(4);
  expect(decoded.durationSeconds).toBe(4 / 16_000);
  expect(calls).toEqual({
    arrayBuffer: 1,
    decode: 1,
    render: 1,
    close: 1,
  });
});

test("browser audio decode fails closed without browser audio primitives", async () => {
  const entry = await import("../index.js");
  globalThis.AudioContext = undefined as unknown as typeof AudioContext;
  globalThis.OfflineAudioContext = undefined as unknown as typeof OfflineAudioContext;

  expect(entry.supportsBrowserAudioDecode()).toBe(false);
  await expect(
    entry.decodeBrowserAudioBlob({
      async arrayBuffer() {
        return new ArrayBuffer(4);
      },
    }),
  ).rejects.toThrow("AudioContext");
});
