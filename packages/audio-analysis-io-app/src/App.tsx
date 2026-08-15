import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-io-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-io",
  title: "Audio Analysis IO",
  description: "Audio input helpers and FFmpeg-backed source conveniences for video-analysis.",
  domain: "audio",
  defaultOperation: "audio.io.waveformBatchSummary",
  featuredOperations: [
    "audio.io.waveformBatchSummary",
    "audio.io.wavSummary",
    "audio.io.inputPlan",
    "audio.io.probePlan",
    "audio.io.decodePlan",
    "audio.io.ffmpegFilterPlan",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["audio.io.waveformBatchSummary", "audio.io.wavSummary"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: [
        "describe",
        "audio.io.inputPlan",
        "audio.io.probePlan",
        "audio.io.decodePlan",
        "audio.io.editPlan",
        "audio.io.splitPlan",
        "audio.io.joinPlan",
        "audio.io.ffmpegFilterPlan",
      ],
    },
  ],
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/audio-analysis-io",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
