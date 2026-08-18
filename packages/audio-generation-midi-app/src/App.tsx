import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/video-analysis-ui/package-surface";
import * as wasm from "@moritzbrantner/audio-generation-midi-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "audio-generation-midi",
  title: "Audio Generation Midi",
  description: "MIDI-like note sequencing, Standard MIDI export, and audio rendering helpers for video-analysis.",
  domain: "audio",
  defaultOperation: "audio.midi.render",
  featuredOperations: [
    "audio.midi.render",
    "audio.midi.fromPitchTrack",
    "audio.midi.encode",
    "audio.midi.note",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["audio.midi.encode", "audio.midi.render", "audio.midi.fromPitchTrack"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "audio.midi.note"],
    },
  ],
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/audio-generation-midi",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
