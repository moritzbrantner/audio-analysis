// Overlay placement through the zoomed waveform viewport (#140): beat, boundary and section layers
// are laid out over the whole file and shifted/scaled so that the visible window fills the stage.
import { describe, expect, test } from "bun:test";
import { viewportPlacement, visibleSegmentPadding } from "../../site/waveform-overlay.js";
import { timeToX } from "../../site/waveform-viewport.js";

const percent = (value) => Number(value.replace("%", ""));

describe("waveform overlay viewport placement", () => {
  test("the whole file or a missing viewport fills the stage", () => {
    expect(viewportPlacement(null)).toEqual({ left: "0%", width: "100%" });
    expect(viewportPlacement({ startSeconds: 0, endSeconds: 72, durationSeconds: 72 })).toEqual({ left: "0%", width: "100%" });
  });

  test("a time placed at its whole-file percentage lands on the viewport's pixel", () => {
    const viewport = { startSeconds: 20, endSeconds: 36, durationSeconds: 72 };
    const placement = viewportPlacement(viewport);
    const stageWidth = 1280;
    const trackLeft = (percent(placement.left) / 100) * stageWidth;
    const trackWidth = (percent(placement.width) / 100) * stageWidth;
    for (const time of [0, 20, 27.5, 36, 72]) {
      expect(trackLeft + (time / 72) * trackWidth).toBeCloseTo(timeToX(viewport, time, stageWidth), 6);
    }
  });

  test("section padding hides only the out-of-view part of a section", () => {
    const viewport = { startSeconds: 30, endSeconds: 31.5, durationSeconds: 72 };
    expect(visibleSegmentPadding(viewport, 24, 48)).toEqual({
      left: `calc(${(6 / 72) * 100}% + 4px)`,
      right: `calc(${(16.5 / 72) * 100}% + 4px)`,
    });
    expect(visibleSegmentPadding(viewport, 30.5, 31)).toEqual({ left: "", right: "" });
    expect(visibleSegmentPadding(null, 0, 10)).toEqual({ left: "", right: "" });
  });
});
