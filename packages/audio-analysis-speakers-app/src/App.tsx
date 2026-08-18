import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-speakers-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-speakers",
  title: "Audio Analysis Speakers",
  description: "Speaker embeddings, enrollment, identification, VAD, and diarization APIs for video-analysis.",
  domain: "audio",
  defaultOperation: "audio.speakers.vad",
  featuredOperations: [
    "audio.speakers.vad",
    "audio.speakers.diarize",
    "audio.speakers.embed",
    "audio.speakers.identify",
    "audio.speakers.assignTranscript",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: [
        "audio.speakers.vad",
        "audio.speakers.diarize",
        "audio.speakers.embed",
        "audio.speakers.identify",
        "audio.speakers.assignTranscript",
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
    scopedRoute: "/api/rust/packages/audio-analysis-speakers",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
