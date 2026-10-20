import { expect, test } from "bun:test";
import { gunzipSync } from "fflate";
import { COMMUNITY_PROFILE } from "../community-profile.js";

test("distributed ONNX templates zero every trained tensor span", () => {
  for (const name of ["segmentation.onnx", "embedding.onnx"] as const) {
    const profile = COMMUNITY_PROFILE.models[name];
    const template = gunzipSync(
      Uint8Array.from(atob(profile.templateBase64), (char) => char.charCodeAt(0)),
    );
    expect(template.length).toBe(profile.bytes);
    expect(profile.recipe.parts.length).toBe(name === "segmentation.onnx" ? 35 : 74);
    let previousEnd = 0;
    for (const span of profile.recipe.parts) {
      expect(span.targetStart).toBeGreaterThanOrEqual(previousEnd);
      expect(span.targetEnd - span.targetStart).toBe(span.end - span.start);
      expect(template.subarray(span.targetStart, span.targetEnd).some((value) => value !== 0)).toBe(
        false,
      );
      previousEnd = span.targetEnd;
    }
  }
});
