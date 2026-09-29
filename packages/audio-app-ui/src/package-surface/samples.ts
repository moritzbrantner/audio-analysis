import type { FileInputDefinition, FileInputSample } from "./types";

export function builtInVideoSamples(): FileInputSample[] {
  return [
    {
      id: "test-pattern",
      label: "Test Pattern",
      url: publicAssetUrl("/samples/video/test-pattern.webm"),
      description: "Generated 2s test pattern clip.",
    },
    {
      id: "color-bars",
      label: "Color Bars",
      url: publicAssetUrl("/samples/video/color-bars.webm"),
      description: "Generated 2s SMPTE color bars clip.",
    },
    {
      id: "moving-box",
      label: "Moving Box",
      url: publicAssetUrl("/samples/video/moving-box.webm"),
      description: "Generated 2s moving rectangle clip.",
    },
  ];
}

export function builtInVideoFileInput(): FileInputDefinition {
  return {
    id: "video",
    label: "Video input",
    accept: "video/*",
    targetPath: ["videoDataUrl"],
    samples: builtInVideoSamples(),
  };
}

export function publicAssetUrl(path: string): string {
  const meta = import.meta as unknown as { env?: { BASE_URL?: string } };
  const base = meta.env?.BASE_URL ?? "/";
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  return `${normalizedBase}${path.replace(/^\/+/, "")}`;
}
