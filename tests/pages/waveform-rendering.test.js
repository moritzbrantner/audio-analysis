import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const appSource = readFileSync(new URL("../../site/app.js", import.meta.url), "utf8");

function makeCanvasContext() {
  const moveToCalls = [];
  const transforms = [];
  return {
    moveToCalls,
    transforms,
    reset() {
      moveToCalls.length = 0;
      transforms.length = 0;
    },
    beginPath() {},
    clearRect() {},
    fillRect() {},
    fillText() {},
    lineTo() {},
    moveTo(x, y) {
      moveToCalls.push([x, y]);
    },
    setTransform(...args) {
      transforms.push(args);
    },
    stroke() {},
    set fillStyle(_value) {},
    set font(_value) {},
    set lineWidth(_value) {},
    set strokeStyle(_value) {},
  };
}

function makeElement() {
  const attributes = new Map();
  return {
    hidden: false,
    textContent: "",
    value: "",
    files: [],
    className: "",
    src: "",
    disabled: false,
    open: false,
    currentTime: 0,
    paused: true,
    classList: {
      add() {},
      remove() {},
    },
    addEventListener() {},
    append() {},
    click() {},
    focus() {},
    getAttribute(name) {
      return attributes.get(name) ?? null;
    },
    getBoundingClientRect() {
      return { left: 0, width: 640, height: 240 };
    },
    getContext() {
      return makeCanvasContext();
    },
    load() {},
    pause() {},
    remove() {},
    removeAttribute(name) {
      attributes.delete(name);
    },
    replaceChildren() {},
    scrollIntoView() {},
    setAttribute(name, value) {
      attributes.set(name, String(value));
    },
  };
}

function loadApp() {
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
    location: { hostname: "example.github.io", search: "" },
    addEventListener() {},
    scrollTo() {},
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
    cancelAnimationFrame() {},
    clearTimeout,
    console,
    document,
    fetch: async () => {
      throw new Error("Unexpected fetch");
    },
    requestAnimationFrame() {
      return 1;
    },
    setTimeout,
    window,
  });

  vm.runInContext(
    `${appSource}\n;globalThis.__waveformRenderingTestApi = { drawWaveform, prepareWaveformCanvas };`,
    context,
    { filename: "site/app.js" },
  );

  return { api: context.__waveformRenderingTestApi, window };
}

function makeCanvas(context) {
  const canvas = makeElement();
  canvas.width = 300;
  canvas.height = 150;
  canvas.getContext = () => context;
  return canvas;
}

function fakeAudioBuffer() {
  const sampleRate = 6_400;
  const samples = Float32Array.from({ length: sampleRate }, (_, index) =>
    0.75 * Math.sin((2 * Math.PI * 440 * index) / sampleRate),
  );
  return {
    length: samples.length,
    sampleRate,
    numberOfChannels: 1,
    duration: 1,
    getChannelData() {
      return samples;
    },
  };
}

describe("Audio Inspector waveform rendering", () => {
  test("keeps waveform sampling in CSS pixels when the backing-store scale changes", () => {
    const { api, window } = loadApp();
    const context = makeCanvasContext();
    const canvas = makeCanvas(context);
    const buffer = fakeAudioBuffer();

    api.drawWaveform(canvas, buffer);
    const logicalMoveCount = context.moveToCalls.length;

    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(240);
    expect(logicalMoveCount).toBe(642);

    context.reset();
    window.devicePixelRatio = 2;
    api.drawWaveform(canvas, buffer);

    expect(canvas.width).toBe(1_280);
    expect(canvas.height).toBe(480);
    expect(context.moveToCalls.length).toBe(logicalMoveCount);
    expect(context.transforms.at(-1)).toEqual([2, 0, 0, 2, 0, 0]);
  });
});
