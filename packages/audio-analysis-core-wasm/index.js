let wasmModulePromise;

export async function init() {
  const wasmEntry = "./pkg/moenarch_audio_analysis_core_wasm.js";
  wasmModulePromise ??= import(/* @vite-ignore */ wasmEntry).then(async (module) => {
    if (typeof module.default === "function") {
      await module.default();
    }
    return module;
  });
  return wasmModulePromise;
}

export async function packageSurface() {
  const module = await init();
  return module.packageSurface();
}

export async function runOperation(request) {
  const module = await init();
  return module.runOperation(request);
}

export async function captureMetrics(samples, sampleRate, channels = 1, options) {
  const module = await init();
  const pcm = samples instanceof Float32Array ? samples : Float32Array.from(samples);
  return module.captureMetrics(pcm, sampleRate, channels, options);
}
