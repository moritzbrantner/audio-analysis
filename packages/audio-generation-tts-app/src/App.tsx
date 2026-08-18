import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-generation-tts-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-generation-tts",
  title: "Audio Generation TTS",
  description: "Generic and speaker-conditioned TTS contracts, validation, and setup diagnostics.",
  domain: "audio",
  defaultOperation: "audio.tts.synthesize",
  featuredOperations: [
    "audio.tts.synthesize",
    "audio.tts.plan",
    "audio.tts.models",
    "audio.tts.referencePromptPlan",
    "audio.tts.debug.f5Mel",
    "audio.tts.debug.vocosVocoder",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["audio.tts.synthesize"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: [
        "describe",
        "audio.tts.plan",
        "audio.tts.models",
        "audio.tts.referencePromptPlan",
        "audio.tts.debug.f5Mel",
        "audio.tts.debug.vocosVocoder",
      ],
    },
  ],
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/audio-generation-tts",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
