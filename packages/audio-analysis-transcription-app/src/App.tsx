import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-transcription-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-transcription",
  title: "Audio Analysis Transcription",
  description: "Rust-native transcription orchestration for audio and video.",
  domain: "audio",
  defaultOperation: "audio.transcription.transcribe",
  featuredOperations: [
    "audio.transcription.transcribe",
    "audio.transcription.importWhisperX",
    "audio.transcription.providers",
    "audio.transcription.plan",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: [
        "audio.transcription.transcribe",
        "audio.transcription.importWhisperX",
      ],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect provider plans, model setup, VAD, and alignment diagnostics.",
      operations: [
        "describe",
        "audio.transcription.providers",
        "audio.transcription.plan",
        "audio.transcription.modelPlan",
        "audio.transcription.vadPlan",
        "audio.transcription.alignmentPlan",
        "audio.transcription.alignmentBundlePlan",
        "audio.transcription.decodePlan",
        "audio.transcription.diarizationPlan",
      ],
    },
  ],
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/audio-analysis-transcription",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
