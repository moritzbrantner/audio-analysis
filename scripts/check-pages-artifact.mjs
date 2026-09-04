#!/usr/bin/env node

import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve, join } from "node:path";

const outputDir = resolve(process.argv[2] ?? "_site");

function requireFile(relativePath) {
  const path = join(outputDir, relativePath);
  assert.ok(existsSync(path), `missing Pages artifact file: ${relativePath}`);
  return path;
}

for (const relativePath of [
  "index.html",
  "app.js",
  "styles.css",
  "analysis-capabilities.json",
  ".nojekyll",
]) {
  requireFile(relativePath);
}

const manifest = JSON.parse(readFileSync(requireFile("analysis-capabilities.json"), "utf8"));
assert.equal(manifest.schemaVersion, "audio-analysis-pages-capabilities/v1");
assert.equal(manifest.defaultRuntime, "client-wasm");
assert.equal(manifest.privacy.githubPages, "browser-only");
assert.equal(manifest.privacy.automaticUpload, false);
assert.equal(manifest.backend.githubPagesEnabled, false);

const html = readFileSync(requireFile("index.html"), "utf8");
assert.match(html, /<script type="module" src="\.\/app\.js"><\/script>/);
assert.match(html, /id="file-input"/);
assert.match(html, /data-example="tone"/);
assert.match(html, /data-example="clicks"/);
assert.match(html, /id="export-json"/);

const appSource = readFileSync(requireFile("app.js"), "utf8");
assert.match(appSource, /if \(!baseUrl \|\| isGitHubPages\(\)\)/);
assert.match(appSource, /automaticUpload: false/);

for (const entry of manifest.browserPackages) {
  const packageRoot = join(outputDir, "wasm", entry.package);
  assert.ok(existsSync(packageRoot), `missing packaged WASM adapter: ${entry.package}`);
  const wrapperPath = join(packageRoot, "index.js");
  assert.ok(existsSync(wrapperPath), `missing ${entry.package}/index.js`);

  const pkgRoot = join(packageRoot, "pkg");
  assert.ok(existsSync(pkgRoot), `missing ${entry.package}/pkg`);
  const packageFiles = readdirSync(pkgRoot);
  assert.ok(
    packageFiles.some((file) => file.endsWith(".wasm")),
    `missing compiled .wasm payload for ${entry.package}`,
  );

  const wrapperSource = readFileSync(wrapperPath, "utf8");
  const importMatch = wrapperSource.match(/const wasmEntry = "\.\/pkg\/([^"\n]+\.js)"/);
  assert.ok(importMatch, `could not resolve generated JS import from ${entry.package}/index.js`);
  assert.ok(
    packageFiles.includes(importMatch[1]),
    `${entry.package}/index.js imports missing generated entry ${importMatch[1]}`,
  );
  assert.match(
    wrapperSource,
    /value instanceof Map/,
    `${entry.package}/index.js must normalize serde-wasm-bindgen Map values`,
  );
  assert.match(
    wrapperSource,
    /const result = surfaceValue\?\.result;/,
    `${entry.package}/index.js must normalize the structured surface result`,
  );
  assert.match(
    wrapperSource,
    /return \{ \.\.\.response, value: result, surfaceValue \};/,
    `${entry.package}/index.js must preserve the structured surface envelope`,
  );

  for (const operation of entry.operations) {
    assert.ok(appSource.includes(`"${operation}"`), `app.js does not invoke declared operation ${operation}`);
  }
}

console.log(`Audio Inspector Pages artifact verified: ${outputDir}`);
