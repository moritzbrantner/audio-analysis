import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-recognition-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-recognition",
  title: "Audio Analysis Recognition",
  description: "Deterministic audio embeddings and similarity search for video-analysis.",
  domain: "audio",
  defaultOperation: "audio.recognition.embed",
  featuredOperations: [
    "audio.recognition.embed",
    "audio.recognition.compare",
    "audio.recognition.search",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: [
        "audio.recognition.embed",
        "audio.recognition.compare",
        "audio.recognition.search",
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
    scopedRoute: "/api/rust/packages/audio-analysis-recognition",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
