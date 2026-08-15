import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-analysis-core-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-analysis-core",
  title: "Audio Analysis Core",
  description: "Shared audio frame conversion, windowing, and streaming helpers for video-analysis.",
  domain: "audio",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/audio-analysis-core",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
