import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const appSource = readFileSync(new URL("../../site/app.js", import.meta.url), "utf8");

function makeCanvasContext() {
  return {
    beginPath() {},
    clearRect() {},
    fillRect() {},
    fillText() {},
    lineTo() {},
    moveTo() {},
    stroke() {},
    set fillStyle(_value) {},
    set font(_value) {},
    set lineWidth(_value) {},
    set strokeStyle(_value) {},
  };
}

function makeElement() {
  return {
    hidden: false,
    textContent: "",
    value: "",
    files: [],
    className: "",
    src: "",
    classList: {
      add() {},
      remove() {},
    },
    addEventListener() {},
    append() {},
    click() {},
    getAttribute() {
      return null;
    },
    getBoundingClientRect() {
      return { width: 640, height: 240 };
    },
    getContext() {
      return makeCanvasContext();
    },
    load() {},
    pause() {},
    remove() {},
    removeAttribute() {},
    replaceChildren() {},
    setAttribute() {},
  };
}

function loadApp({ hostname = "example.github.io", search = "", fetchImpl } = {}) {
  const nodes = new Map();
  const body = makeElement();
  const document = {
    body,
    createElement() {
      return makeElement();
    },
    querySelector(selector) {
      if (!nodes.has(selector)) nodes.set(selector, makeElement());
      return nodes.get(selector);
    },
    querySelectorAll() {
      return [];
    },
  };
  const window = {
    devicePixelRatio: 1,
    location: { hostname, search },
    addEventListener() {},
    scrollTo() {},
  };
  const fetchCalls = [];
  const fetch = async (...args) => {
    fetchCalls.push(args);
    if (fetchImpl) return fetchImpl(...args);
    throw new Error("Unexpected fetch");
  };
  const context = vm.createContext({
    AbortController,
    Array,
    ArrayBuffer,
    Blob,
    DataView,
    Date,
    Error,
    File,
    Float32Array,
    JSON,
    Math,
    Number,
    Object,
    Promise,
    String,
    URL: {
      createObjectURL() {
        return "blob:test";
      },
      revokeObjectURL() {},
    },
    URLSearchParams,
    clearTimeout,
    console,
    document,
    fetch,
    requestAnimationFrame(callback) {
      callback();
    },
    setTimeout,
    window,
  });

  vm.runInContext(
    `${appSource}\n;globalThis.__audioInspectorTestApi = { analyzerDefinitions, backendInitialization, buildFindings, chooseFftSize, createExampleFile, encodeMonoPcm16Wav, initializeBackendBoundary, isGitHubPages, monoCenterWindow, resampleLinear, scanAudioBuffer, state, synthesizeClicks, synthesizeTone };`,
    context,
    { filename: "site/app.js" },
  );

  return {
    api: context.__audioInspectorTestApi,
    fetchCalls,
    window,
  };
}

function fakeAudioBuffer(channels, sampleRate = 4) {
  const data = channels.map((channel) => Float32Array.from(channel));
  return {
    length: data[0].length,
    sampleRate,
    numberOfChannels: data.length,
    duration: data[0].length / sampleRate,
    getChannelData(index) {
      return data[index];
    },
  };
}

describe("Audio Inspector signal analysis", () => {
  test("scans the complete decoded buffer and reports clipping and stereo metrics", () => {
    const { api } = loadApp();
    const buffer = fakeAudioBuffer([
      [0, 1, -1, 0.5],
      [0, 1, -1, 0.5],
    ]);

    const result = api.scanAudioBuffer(buffer);

    expect(result.coverage.kind).toBe("exact-whole-file");
    expect(result.coverage.frameStride).toBe(1);
    expect(result.peak).toBe(1);
    expect(result.rms).toBeCloseTo(0.75, 6);
    expect(result.clippedSampleCount).toBe(4);
    expect(result.clippedSamplePercent).toBeCloseTo(50, 6);
    expect(result.nearSilentSamplePercent).toBeCloseTo(25, 6);
    expect(Array.from(result.firstClippedTimesSeconds)).toEqual([0.25, 0.5]);
    expect(result.stereoCorrelation).toBeCloseTo(1, 6);
    expect(result.stereoBalanceDb).toBeCloseTo(0, 6);
  });

  test("uses deterministic centered windows and FFT sizing", () => {
    const { api } = loadApp();
    const buffer = fakeAudioBuffer([[0, 1, 2, 3, 4, 5]], 6);

    const window = api.monoCenterWindow(buffer, 4);

    expect(window.startSample).toBe(1);
    expect(Array.from(window.samples)).toEqual([1, 2, 3, 4]);
    expect(api.chooseFftSize(128)).toBe(256);
    expect(api.chooseFftSize(1024)).toBe(1024);
    expect(api.chooseFftSize(5000)).toBe(2048);
  });

  test("resamples deterministically without changing equal-rate inputs", () => {
    const { api } = loadApp();
    const samples = [0, 1, 0, -1];

    expect(api.resampleLinear(samples, 4, 4)).toBe(samples);
    const upsampled = api.resampleLinear(samples, 4, 8);
    expect(upsampled).toHaveLength(8);
    expect(upsampled[0]).toBeCloseTo(0, 6);
    expect(upsampled[1]).toBeCloseTo(0.5, 6);
    expect(upsampled[2]).toBeCloseTo(1, 6);
  });

  test("turns analyzer evidence into bounded headline findings", () => {
    const { api } = loadApp();
    const findings = api.buildFindings(
      {
        clippedSampleCount: 2,
        clippedSamplePercent: 1.25,
        nearSilentSamplePercent: 60,
        stereoCorrelation: -0.4,
      },
      { dominantFrequencyHz: 440, centroidHz: 1200 },
      { frequencyHz: 440, confidence: 0.9, noteName: "A4" },
      { key: "A major", confidence: 0.8 },
      { bpm: 120, confidence: 0.75 },
    );

    expect(findings).toHaveLength(6);
    expect(findings.map((finding) => finding.title)).toEqual([
      "Possible clipping detected",
      "Dominant frequency near 440.0 Hz",
      "Pitch estimate: A4",
      "Estimated musical key: A major",
      "Estimated tempo: 120.0 BPM",
      "Large near-silent sample share",
    ]);
  });
});

describe("Audio Inspector deterministic examples", () => {
  test("generates the reference tone and click track deterministically", () => {
    const { api } = loadApp();
    const tone = api.synthesizeTone(8000, 3, 440);
    const clicks = api.synthesizeClicks(8000, 8, 120);

    expect(tone).toHaveLength(24_000);
    expect(clicks).toHaveLength(64_000);
    expect(Math.max(...tone)).toBeLessThanOrEqual(0.45);
    expect(Math.min(...tone)).toBeGreaterThanOrEqual(-0.45);
    expect(Array.from(api.synthesizeTone(8000, 3, 440))).toEqual(Array.from(tone));
  });

  test("encodes examples as valid mono PCM16 WAV files", async () => {
    const { api } = loadApp();
    const wav = api.encodeMonoPcm16Wav(Float32Array.from([0, 1, -1]), 8000);
    const bytes = new Uint8Array(await wav.arrayBuffer());
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const ascii = (offset, length) => String.fromCharCode(...bytes.slice(offset, offset + length));

    expect(ascii(0, 4)).toBe("RIFF");
    expect(ascii(8, 4)).toBe("WAVE");
    expect(ascii(36, 4)).toBe("data");
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(8000);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(6);
    expect(bytes.byteLength).toBe(50);
  });

  test("creates named browser example files", () => {
    const { api } = loadApp();
    const file = api.createExampleFile("tone", "440 Hz reference tone");

    expect(file.name).toBe("440-hz-reference-tone.wav");
    expect(file.type).toBe("audio/wav");
    expect(file.size).toBe(48_044);
  });
});

describe("Audio Inspector runtime boundary", () => {
  test("GitHub Pages ignores an explicit backend and never probes it", async () => {
    const { api, fetchCalls } = loadApp({
      hostname: "moritzbrantner.github.io",
      search: "?backend=https://example.invalid",
    });

    await api.backendInitialization;

    expect(fetchCalls).toHaveLength(0);
    expect(api.isGitHubPages()).toBe(true);
    expect(api.state.backend).toEqual({
      baseUrl: null,
      available: false,
      automaticUpload: false,
      reason: "github-pages-browser-only",
    });
  });

  test("a local deployment may discover a backend without enabling automatic upload", async () => {
    const { api, fetchCalls } = loadApp({
      hostname: "localhost",
      fetchImpl: async () => ({
        ok: true,
        async json() {
          return {
            library: "audio-analysis-transcription",
            version: "1.2.3",
            operations: [{ id: "audio.transcription.transcribe" }],
          };
        },
      }),
    });

    await api.backendInitialization;

    expect(fetchCalls).toHaveLength(1);
    expect(fetchCalls[0][0]).toBe("http://127.0.0.1:3000/api/rust/packages/audio-analysis-transcription/api/package");
    expect(api.state.backend.available).toBe(true);
    expect(api.state.backend.automaticUpload).toBe(false);
    expect(api.state.backend.transcription).toEqual({
      library: "audio-analysis-transcription",
      version: "1.2.3",
      operations: ["audio.transcription.transcribe"],
    });
  });
});
