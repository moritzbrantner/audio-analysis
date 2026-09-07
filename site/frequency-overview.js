const DEFAULT_BIN_COUNT = 1600;
const MIN_BIN_COUNT = 64;
const MAX_BIN_COUNT = 4096;
const LOW_CUTOFF_HZ = 220;
const MID_CUTOFF_HZ = 2000;

export function buildFrequencyOverview(samples, sampleRate, requestedBins = DEFAULT_BIN_COUNT) {
  if (!samples || typeof samples.length !== "number" || samples.length === 0) {
    return { sampleRate, sampleCount: 0, bins: [], maxEnergy: 0 };
  }
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("frequency overview sampleRate must be finite and positive");
  }

  const binCount = Math.max(
    1,
    Math.min(
      samples.length,
      Math.max(MIN_BIN_COUNT, Math.min(MAX_BIN_COUNT, Math.round(requestedBins || DEFAULT_BIN_COUNT))),
    ),
  );
  const bins = Array.from({ length: binCount }, () => ({
    min: 1,
    max: -1,
    lowSquare: 0,
    midSquare: 0,
    highSquare: 0,
    count: 0,
  }));

  const lowAlpha = onePoleAlpha(LOW_CUTOFF_HZ, sampleRate);
  const midAlpha = onePoleAlpha(Math.min(MID_CUTOFF_HZ, sampleRate * 0.45), sampleRate);
  let lowPass = 0;
  let midPass = 0;

  for (let index = 0; index < samples.length; index += 1) {
    const sample = finiteSample(samples[index]);
    lowPass += lowAlpha * (sample - lowPass);
    midPass += midAlpha * (sample - midPass);

    const low = lowPass;
    const mid = midPass - lowPass;
    const high = sample - midPass;
    const binIndex = Math.min(binCount - 1, Math.floor((index * binCount) / samples.length));
    const bin = bins[binIndex];
    bin.min = Math.min(bin.min, sample);
    bin.max = Math.max(bin.max, sample);
    bin.lowSquare += low * low;
    bin.midSquare += mid * mid;
    bin.highSquare += high * high;
    bin.count += 1;
  }

  let maxEnergy = 0;
  const normalized = bins.map((bin) => {
    const count = Math.max(1, bin.count);
    const low = Math.sqrt(bin.lowSquare / count);
    const mid = Math.sqrt(bin.midSquare / count);
    const high = Math.sqrt(bin.highSquare / count);
    const energy = Math.sqrt(low * low + mid * mid + high * high);
    maxEnergy = Math.max(maxEnergy, energy);
    return {
      min: bin.count ? bin.min : 0,
      max: bin.count ? bin.max : 0,
      low,
      mid,
      high,
      energy,
    };
  });

  const safeMax = Math.max(maxEnergy, Number.EPSILON);
  for (const bin of normalized) {
    bin.intensity = clamp01(Math.sqrt(bin.energy / safeMax));
    const bandTotal = bin.low + bin.mid + bin.high;
    if (bandTotal > Number.EPSILON) {
      bin.lowShare = bin.low / bandTotal;
      bin.midShare = bin.mid / bandTotal;
      bin.highShare = bin.high / bandTotal;
    } else {
      bin.lowShare = 0;
      bin.midShare = 0;
      bin.highShare = 0;
    }
  }

  return {
    schemaVersion: "audio-analysis-frequency-overview/v1",
    sampleRate,
    sampleCount: samples.length,
    lowCutoffHz: LOW_CUTOFF_HZ,
    midCutoffHz: Math.min(MID_CUTOFF_HZ, sampleRate * 0.45),
    maxEnergy,
    bins: normalized,
  };
}

export function drawFrequencyOverview(context, width, height, overview) {
  const bins = Array.isArray(overview?.bins) ? overview.bins : [];
  if (!context || bins.length === 0 || width <= 0 || height <= 0) return;

  const middle = height * 0.58;
  const amplitude = height * 0.28;
  context.save();
  context.lineWidth = 1;

  for (let x = 0; x < width; x += 1) {
    const binIndex = Math.min(bins.length - 1, Math.floor((x * bins.length) / width));
    const bin = bins[binIndex];
    const intensity = clamp01(bin.intensity ?? 0);
    const low = clamp01(bin.lowShare ?? 0);
    const mid = clamp01(bin.midShare ?? 0);
    const high = clamp01(bin.highShare ?? 0);

    // Low frequencies bias warm, mids green/cyan and highs blue/violet.
    // Brightness follows local energy, so breakdowns remain visibly sparse.
    const red = Math.round(34 + intensity * (205 * low + 55 * mid + 35 * high));
    const green = Math.round(42 + intensity * (45 * low + 205 * mid + 70 * high));
    const blue = Math.round(58 + intensity * (35 * low + 95 * mid + 205 * high));
    const alpha = 0.42 + intensity * 0.56;
    context.strokeStyle = `rgba(${clampByte(red)}, ${clampByte(green)}, ${clampByte(blue)}, ${alpha.toFixed(3)})`;

    const min = Number.isFinite(bin.min) ? bin.min : 0;
    const max = Number.isFinite(bin.max) ? bin.max : 0;
    const energyFloor = Math.max(0.06, intensity * 0.18);
    const top = middle - Math.max(Math.abs(max), energyFloor) * amplitude;
    const bottom = middle + Math.max(Math.abs(min), energyFloor) * amplitude;
    context.beginPath();
    context.moveTo(x + 0.5, top);
    context.lineTo(x + 0.5, bottom);
    context.stroke();
  }

  context.restore();
}

function onePoleAlpha(cutoffHz, sampleRate) {
  const cutoff = Math.max(1, Math.min(cutoffHz, sampleRate * 0.45));
  return 1 - Math.exp((-2 * Math.PI * cutoff) / sampleRate);
}

function finiteSample(value) {
  return Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}
