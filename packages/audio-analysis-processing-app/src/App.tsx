import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-processing-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-processing",
  title: "Audio Analysis Processing",
  description: "Realtime-safe audio transforms and processed sources for video-analysis.",
  domain: "audio",
  defaultOperation: "audio.processing.apply",
  featuredOperations: [
    "audio.processing.apply",
    "audio.processing.offlineEdit",
    "audio.processing.mixdown",
    "audio.processing.loudness",
    "audio.processing.preset",
    "audio.processing.energy",
    "audio.processing.effectsCatalog",
    "audio.processing.chainSummary",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: [
        "audio.processing.apply",
        "audio.processing.offlineEdit",
        "audio.processing.mixdown",
        "audio.processing.loudness",
        "audio.processing.preset",
        "audio.processing.energy",
      ],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "audio.processing.effectsCatalog", "audio.processing.chainSummary"],
    },
  ],
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/audio-analysis-processing",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
