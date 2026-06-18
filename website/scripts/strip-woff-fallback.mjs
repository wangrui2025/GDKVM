#!/usr/bin/env node
// Post-build: Strip legacy .woff KaTeX font fallback (keep woff2 only).
//
// Mirrors the pattern from main site (mykcs.github.io build-pipeline.mjs
// removeLegacyWoff). woff2 is universally supported (>97%); the legacy
// woff fallback was for IE/old Android which no current browser needs.
// Saves ~250KB per build for GDKVM (~20 woff files).
//
// We MUST also strip the woff source from CSS — otherwise browsers will
// 404 on the missing fallback url. The font-face src list has 2 entries:
// woff2 (keep) + woff (drop).
import fs from 'node:fs';
import path from 'node:path';

const DIST_DIR = path.join(process.cwd(), 'dist');
const _astro = path.join(DIST_DIR, '_astro');

if (!fs.existsSync(_astro)) {
  console.log('[strip-woff-fallback] No dist/_astro dir, skipping');
  process.exit(0);
}

// Phase 1: Remove .woff files
let removedBytes = 0;
let removedCount = 0;
for (const f of fs.readdirSync(_astro)) {
  if (f.endsWith('.woff')) {
    const p = path.join(_astro, f);
    const size = fs.statSync(p).size;
    fs.unlinkSync(p);
    removedBytes += size;
    removedCount++;
  }
}
console.log(
  `[strip-woff-fallback] Removed ${removedCount} woff files (${(removedBytes / 1024).toFixed(0)}KB)`
);

// Phase 2: Strip woff source URLs from CSS files
function stripWoffFromFile(filePath) {
  if (!fs.existsSync(filePath)) return false;
  const original = fs.readFileSync(filePath, 'utf-8');
  // Remove woff source list item, e.g. ",url(/GDKVM/_astro/x.woff) format(\"woff\")"
  const stripped = original.replace(
    /,url\([^)]+\.woff[^)]*\) format\("woff"\)/g,
    ''
  );
  if (stripped !== original) {
    fs.writeFileSync(filePath, stripped, 'utf-8');
    return true;
  }
  return false;
}

let cssCleaned = 0;
for (const f of fs.readdirSync(_astro)) {
  if (f.endsWith('.css') && stripWoffFromFile(path.join(_astro, f))) {
    cssCleaned++;
  }
}

// Phase 3: Strip woff source from inlined <style> in HTML files
let htmlCleaned = 0;
function walkHtml(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkHtml(full);
    } else if (entry.name.endsWith('.html')) {
      if (stripWoffFromFile(full)) htmlCleaned++;
    }
  }
}
walkHtml(DIST_DIR);

console.log(
  `[strip-woff-fallback] Stripped woff src from ${cssCleaned} CSS + ${htmlCleaned} HTML files`
);
console.log('[strip-woff-fallback] Done');