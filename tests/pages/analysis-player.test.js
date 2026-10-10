import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import {
  formatTransportTime,
  isPlaybackToggleKey,
  normalizePlaybackRate,
  setupAnalysisPlayer,
  transportTimeText,
} from "../../site/analysis-player.js";

class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.attributes = new Map();
    this.children = [];
    this.listeners = new Map();
    this.textContent = "";
    this.value = "";
    this.disabled = false;
    this.parent = null;
  }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  append(...nodes) {
    for (const node of nodes) {
      node.parent = this;
      this.children.push(node);
    }
  }
  after(node) {
    this.parent.children.splice(this.parent.children.indexOf(this) + 1, 0, node);
    node.parent = this.parent;
  }
  remove() {
    this.parent.children.splice(this.parent.children.indexOf(this), 1);
  }
  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(handler);
  }
  removeEventListener(type, handler) {
    this.listeners.get(type)?.delete(handler);
  }
  dispatch(type) {
    for (const handler of this.listeners.get(type) ?? []) handler({ type });
  }
  listenerCount() {
    return [...this.listeners.values()].reduce((sum, set) => sum + set.size, 0);
  }
}

function fakeMedia() {
  const media = new FakeNode("audio");
  Object.assign(media, {
    controls: true,
    paused: true,
    ended: false,
    currentTime: 0,
    duration: Number.NaN,
    volume: 1,
    muted: false,
    playbackRate: 1,
    defaultPlaybackRate: 1,
    currentSrc: "",
    async play() {
      this.paused = false;
      this.dispatch("play");
    },
    pause() {
      this.paused = true;
      this.dispatch("pause");
    },
  });
  media.setAttribute("controls", "");
  const panel = new FakeNode("section");
  panel.append(media);
  return { media, panel };
}

const doc = { createElement: (tag) => new FakeNode(tag) };

function controls(transport) {
  const [play, time, mute, volume, rateLabel] = transport.children;
  return { play, time, mute, volume, rate: rateLabel.children[0] };
}

describe("analysis player transport", () => {
  test("formats transport time with an unknown duration placeholder", () => {
    expect(formatTransportTime(0)).toBe("0:00");
    expect(formatTransportTime(75.9)).toBe("1:15");
    expect(formatTransportTime(3725)).toBe("1:02:05");
    expect(formatTransportTime(Number.NaN)).toBe("0:00");
    expect(transportTimeText(12, Number.NaN)).toBe("0:12 / --:--");
    expect(transportTimeText(12, 180)).toBe("0:12 / 3:00");
    expect(normalizePlaybackRate("1.5")).toBe(1.5);
    expect(normalizePlaybackRate(3)).toBe(1);
  });

  test("replaces native controls with one transport driven by the media element", async () => {
    const { media, panel } = fakeMedia();
    const player = setupAnalysisPlayer(media, doc);

    expect(media.getAttribute("controls")).toBeNull();
    expect(media.controls).toBe(false);
    expect(panel.children[1]).toBe(player.element);
    expect(player.element.getAttribute("aria-label")).toBe("Playback");
    const { play, time, mute, volume, rate } = controls(player.element);
    expect(play.disabled).toBe(true);
    expect(rate.children.map((option) => option.value)).toEqual(["0.5", "0.75", "1", "1.25", "1.5", "2"]);

    media.currentSrc = "blob:track";
    media.duration = 180;
    media.dispatch("loadedmetadata");
    expect(play.disabled).toBe(false);
    expect(time.textContent).toBe("0:00 / 3:00");

    play.dispatch("click");
    await Promise.resolve();
    expect(media.paused).toBe(false);
    expect(play.getAttribute("aria-label")).toBe("Pause");
    expect(play.getAttribute("aria-pressed")).toBeNull();

    // A waveform seek writes the same media time the transport shows.
    media.currentTime = 42;
    media.dispatch("seeked");
    expect(time.textContent).toBe("0:42 / 3:00");

    mute.dispatch("click");
    expect(media.muted).toBe(true);
    expect(mute.getAttribute("aria-label")).toBe("Unmute");
    expect(volume.value).toBe("0");
    mute.dispatch("click");
    expect(media.muted).toBe(false);
    expect(volume.value).toBe("1");

    volume.value = "0.4";
    volume.dispatch("input");
    expect(media.volume).toBe(0.4);
    volume.value = "0";
    volume.dispatch("input");
    expect(media.muted).toBe(true);
    mute.dispatch("click");
    expect(media.muted).toBe(false);
    expect(media.volume).toBe(1);

    rate.value = "1.25";
    rate.dispatch("change");
    expect(media.playbackRate).toBe(1.25);
    expect(media.defaultPlaybackRate).toBe(1.25);

    play.dispatch("click");
    await Promise.resolve();
    expect(media.paused).toBe(true);
    expect(play.getAttribute("aria-label")).toBe("Play");
  });

  test("hides the volume slider where media volume cannot be set", () => {
    const { media } = fakeMedia();
    Object.defineProperty(media, "volume", { get: () => 1, set: () => {} });
    const player = setupAnalysisPlayer(media, doc);
    expect(controls(player.element).volume.hidden).toBe(true);
    const { media: desktop } = fakeMedia();
    expect(controls(setupAnalysisPlayer(desktop, doc).element).volume.hidden).toBe(false);
    expect(desktop.volume).toBe(1);
  });

  test("survives a refused play() and detaches cleanly", async () => {
    const { media, panel } = fakeMedia();
    media.currentSrc = "blob:track";
    media.play = async () => {
      throw new Error("NotAllowedError");
    };
    const player = setupAnalysisPlayer(media, doc);
    await player.togglePlayback();
    expect(controls(player.element).play.getAttribute("aria-label")).toBe("Play");

    player.dispose();
    expect(panel.children).toEqual([media]);
    expect(media.listenerCount()).toBe(0);
  });

  test("toggles playback only for unmodified Space and K", () => {
    expect(isPlaybackToggleKey({ key: " " })).toBe(true);
    expect(isPlaybackToggleKey({ key: "k" })).toBe(true);
    expect(isPlaybackToggleKey({ key: "K" })).toBe(true);
    expect(isPlaybackToggleKey({ key: " ", ctrlKey: true })).toBe(false);
    expect(isPlaybackToggleKey({ key: " ", shiftKey: true })).toBe(false);
    expect(isPlaybackToggleKey({ key: "K", shiftKey: true })).toBe(false);
    expect(isPlaybackToggleKey({ key: "ArrowRight" })).toBe(false);
  });

  test("both inspector pages use the integrated transport instead of native controls", () => {
    for (const page of ["index.html", "song-analysis.html"]) {
      const html = readFileSync(new URL(`../../site/${page}`, import.meta.url), "utf8");
      expect(html).not.toMatch(/<audio[^>]*\scontrols/);
      expect(html).toContain('src="./analysis-player.js"');
    }
  });
});
