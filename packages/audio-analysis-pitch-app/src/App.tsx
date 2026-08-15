import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-pitch-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-pitch",
  title: "Audio Analysis Pitch",
  description: "Autocorrelation pitch detection for video-analysis audio pipelines.",
  domain: "audio",
  defaultOperation: "audio.pitch.estimate",
  featuredOperations: [
    "audio.pitch.estimate",
    "audio.pitch.track",
    "audio.pitch.chroma",
    "audio.pitch.noteName",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["audio.pitch.estimate", "audio.pitch.track", "audio.pitch.chroma"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "audio.pitch.noteName"],
    },
  ],
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/audio-analysis-pitch",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
