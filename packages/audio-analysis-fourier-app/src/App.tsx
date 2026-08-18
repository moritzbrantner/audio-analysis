import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-fourier-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-fourier",
  title: "Audio Analysis Fourier",
  description: "FFT, STFT, and spectral audio analysis for video-analysis.",
  domain: "audio",
  defaultOperation: "audio.fourier.spectrum",
  featuredOperations: ["audio.fourier.spectrum", "audio.fourier.spectrogram", "audio.fourier.features", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["audio.fourier.spectrum", "audio.fourier.spectrogram", "audio.fourier.features"],
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
    scopedRoute: "/api/rust/packages/audio-analysis-fourier",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
