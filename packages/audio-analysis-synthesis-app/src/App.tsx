import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-synthesis-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-synthesis",
  title: "Audio Analysis Synthesis",
  description: "Deterministic audio synthesis from analysis events for video-analysis.",
  domain: "audio",
  defaultOperation: "audio.synthesis.tone",
  featuredOperations: [
    "audio.synthesis.tone",
    "audio.synthesis.timeline",
    "audio.synthesis.fromEvents",
    "audio.synthesis.clickTrack",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: [
        "audio.synthesis.tone",
        "audio.synthesis.timeline",
        "audio.synthesis.fromEvents",
        "audio.synthesis.clickTrack",
      ],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe"],
    },
  ],
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/audio-analysis-synthesis",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
