import type { FileInputDefinition, FileInputSample } from "./types";

export const COLMAP_TEST_VIDEO_PATH = "prototypes/web/video-analysis-web/public/samples/video/test-video.mp4";
export const COLMAP_TEST_VIDEO_URL = "/samples/video/test-video.mp4";
export const COLMAP_TEST_VIDEO_OUTPUT_DIR = ".external-test-tools/colmap-runs/test-video";
export const COLMAP_TEST_VIDEO_SETUP_COMMAND = "bun run setup:colmap-video";

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
    colmapTestVideoSample(),
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

export function colmapTestVideoSample(): FileInputSample {
  return {
    id: "colmap-test-video",
    label: "COLMAP Test Video",
    url: publicAssetUrl(COLMAP_TEST_VIDEO_URL),
    description: "Optional downloaded video for native COLMAP reconstruction.",
    missingHint: `Create it with ${COLMAP_TEST_VIDEO_SETUP_COMMAND}.`,
    patches: [
      { targetPath: ["videoPath"], value: COLMAP_TEST_VIDEO_PATH },
      { targetPath: ["videoUrl"], value: COLMAP_TEST_VIDEO_URL },
      { targetPath: ["outputDir"], value: COLMAP_TEST_VIDEO_OUTPUT_DIR },
    ],
  };
}

export function publicAssetUrl(path: string): string {
  const meta = import.meta as unknown as { env?: { BASE_URL?: string } };
  const base = meta.env?.BASE_URL ?? "/";
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  return `${normalizedBase}${path.replace(/^\/+/, "")}`;
}
