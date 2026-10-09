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
  // wasm-bindgen wraps out-of-range numbers into u32/u16 instead of rejecting them.
  requireInteger("sampleRate", sampleRate, 1, 0xffff_ffff);
  requireInteger("channels", channels, 1, 0xffff);
  const module = await init();
  const pcm = samples instanceof Float32Array ? samples : Float32Array.from(samples);
  return module.captureMetrics(pcm, sampleRate, channels, options);
}

function requireInteger(name, value, min, max) {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer from ${min} to ${max}`);
  }
}
