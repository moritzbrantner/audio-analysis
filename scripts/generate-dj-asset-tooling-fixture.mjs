#!/usr/bin/env bun

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

function parseArgs(argv) {
  const values = {};
  for (let index = 2; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!flag?.startsWith("--") || value === undefined) {
      throw new Error(
        "usage: generate-dj-asset-tooling-fixture.mjs --asset-tooling <path> --recipe <json> --output <wav> --manifest <json>",
      );
    }
    values[flag.slice(2)] = value;
  }
  for (const name of ["asset-tooling", "recipe", "output", "manifest"]) {
    if (!values[name]) throw new Error(`missing --${name}`);
  }
  return values;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function assertInteger(value, name, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}`);
  }
  return value;
}

function assertArrayOfGridOffsets(value, name) {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value.map((offset, index) => assertInteger(offset, `${name}[${index}]`, 0));
}

function validateRecipe(recipe) {
  if (recipe?.schemaVersion !== "audio-analysis-dj-generated-fixture/v1") {
    throw new Error("unsupported generated DJ fixture schemaVersion");
  }
  const audio = recipe.audio;
  assertInteger(audio?.sampleRate, "audio.sampleRate", 8_000);
  assertInteger(audio?.channels, "audio.channels", 1);
  if (audio.channels !== 1) throw new Error("the v1 generated fixture must be mono");
  assertInteger(audio?.bpm, "audio.bpm", 1);
  assertInteger(audio?.beatsPerBar, "audio.beatsPerBar", 1);
  assertInteger(audio?.bars, "audio.bars", 1);
  if (!Array.isArray(recipe.arrangement) || recipe.arrangement.length !== audio.bars) {
    throw new Error("arrangement must contain exactly one entry per bar");
  }
  if (!recipe.generator?.revision || !recipe.generator?.repository) {
    throw new Error("generator repository and revision are required");
  }
  if (!recipe.instruments || !recipe.chords || !recipe.patterns) {
    throw new Error("instruments, chords, and patterns are required");
  }
  for (const [name, pattern] of Object.entries(recipe.patterns)) {
    for (const field of ["kickHalfBeats", "snareHalfBeats", "hatHalfBeats", "bassHalfBeats"]) {
      assertArrayOfGridOffsets(pattern[field], `patterns.${name}.${field}`);
    }
  }
  return recipe;
}

function gitHead(repositoryPath) {
  return execFileSync("git", ["-C", repositoryPath, "rev-parse", "HEAD"], {
    encoding: "utf8",
  }).trim();
}

const args = parseArgs(process.argv);
const recipePath = path.resolve(args.recipe);
const assetToolingRoot = path.resolve(args["asset-tooling"]);
const outputPath = path.resolve(args.output);
const manifestPath = path.resolve(args.manifest);
const recipeBytes = await readFile(recipePath);
const recipe = validateRecipe(JSON.parse(recipeBytes.toString("utf8")));
const actualRevision = gitHead(assetToolingRoot);
if (actualRevision !== recipe.generator.revision) {
  throw new Error(
    `asset-tooling revision mismatch: ${actualRevision} != ${recipe.generator.revision}`,
  );
}

const operations = await import(
  pathToFileURL(path.join(assetToolingRoot, "src", "audio-operations.js")).href
);
const assetStore = await import(
  pathToFileURL(path.join(assetToolingRoot, "src", "asset-store.js")).href
);

const {
  executeAudioFadeOperation,
  executeAudioMixOperation,
  executeAudioSynthesizeOperation,
} = operations;
const { resolveAssetObject } = assetStore;

const sampleRate = recipe.audio.sampleRate;
const framesPerBeat = (sampleRate * 60) / recipe.audio.bpm;
if (!Number.isSafeInteger(framesPerBeat)) {
  throw new Error("audio.sampleRate * 60 / audio.bpm must resolve to an exact frame count");
}
if (framesPerBeat % 2 !== 0) {
  throw new Error("framesPerBeat must be divisible by two for the v1 half-beat grid");
}
const halfBeatFrames = framesPerBeat / 2;
const barFrames = framesPerBeat * recipe.audio.beatsPerBar;
const expectedFrames = barFrames * recipe.audio.bars;
const workspace = await mkdtemp(path.join(os.tmpdir(), "audio-analysis-dj-fixture-"));

try {
  const sourceCache = new Map();

  async function sourceAsset({ waveform, amplitude, durationFrames, frequencyHz, seed, fadeInFrames, fadeOutFrames }) {
    const descriptor = {
      waveform,
      amplitude,
      durationFrames,
      frequencyHz: frequencyHz ?? null,
      seed: seed ?? null,
      fadeInFrames: fadeInFrames ?? 0,
      fadeOutFrames: fadeOutFrames ?? 0,
    };
    const cacheKey = JSON.stringify(descriptor);
    if (sourceCache.has(cacheKey)) return sourceCache.get(cacheKey);

    const parameters = {
      waveform,
      sampleRate,
      channels: recipe.audio.channels,
      frameCount: assertInteger(durationFrames, "source durationFrames", 1),
      amplitude: assertInteger(amplitude, "source amplitude", 0),
    };
    if (frequencyHz !== undefined) parameters.frequencyHz = assertInteger(frequencyHz, "source frequencyHz", 1);
    if (seed !== undefined) parameters.seed = String(seed);

    const synthesized = await executeAudioSynthesizeOperation(workspace, {
      parameters,
      inputs: {},
    });
    let asset = synthesized.outputs.output;
    const fadeIn = assertInteger(fadeInFrames ?? 0, "source fadeInFrames", 0);
    const fadeOut = assertInteger(fadeOutFrames ?? 0, "source fadeOutFrames", 0);
    if (fadeIn > 0 || fadeOut > 0) {
      if (fadeIn + fadeOut > durationFrames) {
        throw new Error("source fades must not overlap beyond the source duration");
      }
      const faded = await executeAudioFadeOperation(workspace, {
        parameters: { fadeInFrames: fadeIn, fadeOutFrames: fadeOut },
        inputs: { source: asset },
      });
      asset = faded.outputs.output;
    }
    sourceCache.set(cacheKey, asset);
    return asset;
  }

  async function mixPlacements(placements) {
    if (placements.length === 0) throw new Error("cannot mix an empty placement list");
    if (placements.length <= 32) {
      const mixed = await executeAudioMixOperation(workspace, {
        parameters: {
          tracks: placements.map((placement) => ({
            startFrame: placement.startFrame,
            gainNumerator: 1,
            gainDenominator: 1,
          })),
        },
        inputs: { sources: placements.map((placement) => placement.asset) },
      });
      return mixed.outputs.output;
    }

    const partials = [];
    for (let index = 0; index < placements.length; index += 32) {
      const asset = await mixPlacements(placements.slice(index, index + 32));
      partials.push({ asset, startFrame: 0 });
    }
    return mixPlacements(partials);
  }

  const placements = [];
  const pad = recipe.instruments.pad;
  const bass = recipe.instruments.bass;
  const kick = recipe.instruments.kick;
  const downbeat = recipe.instruments.downbeat;
  const snare = recipe.instruments.snare;
  const hat = recipe.instruments.hat;

  const kickAsset = await sourceAsset({
    ...kick,
    durationFrames: kick.durationFrames,
    frequencyHz: kick.frequencyHz,
  });
  const downbeatAsset = await sourceAsset({
    ...downbeat,
    durationFrames: downbeat.durationFrames,
    frequencyHz: downbeat.frequencyHz,
  });
  const snareAsset = await sourceAsset({ ...snare, durationFrames: snare.durationFrames });
  const hatAsset = await sourceAsset({ ...hat, durationFrames: hat.durationFrames });

  for (let barIndex = 0; barIndex < recipe.arrangement.length; barIndex += 1) {
    const bar = recipe.arrangement[barIndex];
    const chord = recipe.chords[bar.chord];
    const pattern = recipe.patterns[bar.pattern];
    if (!chord) throw new Error(`unknown chord '${bar.chord}' at bar ${barIndex}`);
    if (!pattern) throw new Error(`unknown pattern '${bar.pattern}' at bar ${barIndex}`);
    const barStart = barIndex * barFrames;

    placements.push({ asset: downbeatAsset, startFrame: barStart });

    for (const frequencyHz of chord.frequenciesHz) {
      const asset = await sourceAsset({
        ...pad,
        durationFrames: barFrames,
        frequencyHz,
      });
      placements.push({ asset, startFrame: barStart });
    }

    const bassAsset = await sourceAsset({
      ...bass,
      durationFrames: bass.durationFrames,
      frequencyHz: chord.rootFrequencyHz,
    });
    for (const offset of pattern.bassHalfBeats) {
      placements.push({ asset: bassAsset, startFrame: barStart + offset * halfBeatFrames });
    }
    for (const offset of pattern.kickHalfBeats) {
      placements.push({ asset: kickAsset, startFrame: barStart + offset * halfBeatFrames });
    }
    for (const offset of pattern.snareHalfBeats) {
      placements.push({ asset: snareAsset, startFrame: barStart + offset * halfBeatFrames });
    }
    for (const offset of pattern.hatHalfBeats) {
      placements.push({ asset: hatAsset, startFrame: barStart + offset * halfBeatFrames });
    }
  }

  const outputAsset = await mixPlacements(placements);
  if (outputAsset.metadata.audio.frameCount !== expectedFrames) {
    throw new Error(
      `generated frame count ${outputAsset.metadata.audio.frameCount} != expected ${expectedFrames}`,
    );
  }
  const outputBytes = await resolveAssetObject(workspace, outputAsset);
  const outputDigest = sha256(outputBytes);
  if (outputDigest !== outputAsset.sha256) {
    throw new Error(`content hash mismatch: ${outputDigest} != ${outputAsset.sha256}`);
  }

  await mkdir(path.dirname(outputPath), { recursive: true });
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await writeFile(outputPath, outputBytes);
  const manifest = {
    schemaVersion: "audio-analysis-dj-generated-evidence/v1",
    fixture: recipe.name,
    recipeSha256: sha256(recipeBytes),
    assetTooling: {
      repository: recipe.generator.repository,
      revision: actualRevision,
    },
    outputSha256: outputDigest,
    outputAsset,
    frameCount: expectedFrames,
    durationSeconds: expectedFrames / sampleRate,
    sourceCount: sourceCache.size,
    placementCount: placements.length,
  };
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify(manifest)}\n`);
} finally {
  await rm(workspace, { recursive: true, force: true });
}
