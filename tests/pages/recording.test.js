import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const recordingSource = readFileSync(new URL("../../site/recording.js", import.meta.url), "utf8");

function makeElement() {
  const attributes = new Map();
  const listeners = new Map();
  return {
    disabled: false,
    files: [],
    hidden: false,
    textContent: "",
    addEventListener(type, listener) {
      const existing = listeners.get(type) ?? [];
      existing.push(listener);
      listeners.set(type, existing);
    },
    dispatchEvent(event) {
      for (const listener of listeners.get(event.type) ?? []) listener(event);
      return true;
    },
    getAttribute(name) {
      return attributes.get(name) ?? null;
    },
    setAttribute(name, value) {
      attributes.set(name, String(value));
    },
  };
}

function loadRecordingSurface({ mediaRecorder = true } = {}) {
  const selectors = new Map([
    ["#record-audio", makeElement()],
    ["#recording-status", makeElement()],
    ["#file-input", makeElement()],
    ["#choose-file", makeElement()],
    ["#input-panel", makeElement()],
    ["#input-error", makeElement()],
  ]);
  const exampleButton = makeElement();
  const track = { stopped: false, stop() { this.stopped = true; } };
  const stream = { getTracks: () => [track] };
  let recorder;
  let requestedConstraints;

  class FakeMediaRecorder {
    static isTypeSupported(type) {
      return type === "audio/webm;codecs=opus";
    }

    constructor(inputStream, options = {}) {
      expect(inputStream).toBe(stream);
      this.listeners = new Map();
      this.mimeType = options.mimeType ?? "audio/webm";
      this.state = "inactive";
      recorder = this;
    }

    addEventListener(type, listener) {
      const existing = this.listeners.get(type) ?? [];
      existing.push(listener);
      this.listeners.set(type, existing);
    }

    emit(type, event = {}) {
      for (const listener of this.listeners.get(type) ?? []) listener(event);
    }

    start() {
      this.state = "recording";
    }

    stop() {
      this.state = "inactive";
      this.emit("dataavailable", {
        data: new Blob(["recorded voice"], { type: this.mimeType }),
      });
      this.emit("stop");
    }
  }

  class FakeDataTransfer {
    constructor() {
      this.files = [];
      this.items = {
        add: (file) => {
          this.files = [file];
        },
      };
    }
  }

  class FakeEvent {
    constructor(type) {
      this.type = type;
    }
  }

  class FakeMutationObserver {
    observe() {}
  }

  const context = vm.createContext({
    Blob,
    DataTransfer: FakeDataTransfer,
    Date,
    Event: FakeEvent,
    File,
    MediaRecorder: mediaRecorder ? FakeMediaRecorder : undefined,
    MutationObserver: FakeMutationObserver,
    console,
    document: {
      querySelector(selector) {
        return selectors.get(selector) ?? null;
      },
      querySelectorAll(selector) {
        return selector === "[data-example]" ? [exampleButton] : [];
      },
    },
    navigator: {
      mediaDevices: {
        async getUserMedia(constraints) {
          requestedConstraints = constraints;
          return stream;
        },
      },
    },
  });

  vm.runInContext(recordingSource, context, { filename: "site/recording.js" });
  return {
    exampleButton,
    recorder: () => recorder,
    requestedConstraints: () => requestedConstraints,
    selectors,
    track,
  };
}

test("recorded microphone audio is forwarded through the existing file change path", async () => {
  const surface = loadRecordingSurface();
  const recordButton = surface.selectors.get("#record-audio");
  const fileInput = surface.selectors.get("#file-input");
  let forwardedFile;
  fileInput.addEventListener("change", () => {
    forwardedFile = fileInput.files[0];
  });

  recordButton.dispatchEvent({ type: "click" });
  await Promise.resolve();
  await Promise.resolve();

  expect(surface.requestedConstraints()?.audio).toBe(true);
  expect(surface.recorder().state).toBe("recording");
  expect(recordButton.textContent).toBe("Stop recording");
  expect(fileInput.disabled).toBe(true);
  expect(surface.exampleButton.disabled).toBe(true);

  recordButton.dispatchEvent({ type: "click" });

  expect(surface.track.stopped).toBe(true);
  expect(forwardedFile).toBeInstanceOf(File);
  expect(forwardedFile.type).toBe("audio/webm;codecs=opus");
  expect(forwardedFile.name).toEndWith(".webm");
  expect(fileInput.disabled).toBe(false);
  expect(surface.selectors.get("#recording-status").textContent).toContain("Analyzing it locally");
});

test("unsupported browsers keep file analysis available", () => {
  const surface = loadRecordingSurface({ mediaRecorder: false });
  expect(surface.selectors.get("#record-audio").disabled).toBe(true);
  expect(surface.selectors.get("#choose-file").disabled).toBe(false);
  expect(surface.selectors.get("#recording-status").textContent).toContain("not supported");
});
