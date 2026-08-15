import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-rhythm-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-rhythm",
  title: "Audio Analysis Rhythm",
  description: "Onset and tempo analysis for video-analysis audio pipelines.",
  domain: "audio",
  defaultOperation: "audio.rhythm.onsets",
  featuredOperations: ["audio.rhythm.onsets", "audio.rhythm.tempo", "audio.rhythm.beatGrid", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["audio.rhythm.onsets", "audio.rhythm.tempo", "audio.rhythm.beatGrid"],
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
    scopedRoute: "/api/rust/packages/audio-analysis-rhythm",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
