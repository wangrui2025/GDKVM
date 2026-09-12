#!/usr/bin/env node
// Post-build: Strip legacy KaTeX font fallbacks (keep woff2 only) and
// relax `font-display` from `block` to `swap`.
//
// Mirrors the pattern from main site (mykcs.github.io build-pipeline.mjs
// removeLegacyWoff). woff2 is universally supported (>97%); the legacy
// woff/ttf fallbacks were for IE/old Android which no current browser needs.
// KaTeX's own CSS ships a 3-entry src list — woff2 (keep) + woff + ttf
// (both dropped). Stripping woff alone left ~540KB of .ttf in dist.
//
// We MUST also strip the fallback sources from CSS — otherwise browsers
// will 404 on the missing files.
//
// font-display: KaTeX ships `block`, which hides math glyphs for up to 3s
// while the font loads. `swap` renders fallback glyphs immediately and
// swaps in KaTeX faces when ready — strictly better for FCP/LCP.
import fs from 'node:fs';
import path from 'node:path';

const DIST_DIR = path.join(process.cwd(), 'dist');
const _astro = path.join(DIST_DIR, '_astro');

if (!fs.existsSync(_astro)) {
  console.log('[strip-woff-fallback] No dist/_astro dir, skipping');
  process.exit(0);
}

// Phase 1: Remove legacy .woff / .ttf font files (woff2 is kept)
const LEGACY_FONT_EXT = ['.woff', '.ttf'];
let removedBytes = 0;
let removedCount = 0;
for (const f of fs.readdirSync(_astro)) {
  if (LEGACY_FONT_EXT.some((ext) => f.endsWith(ext))) {
    const p = path.join(_astro, f);
    const size = fs.statSync(p).size;
    fs.unlinkSync(p);
    removedBytes += size;
    removedCount++;
  }
}
console.log(
  `[strip-woff-fallback] Removed ${removedCount} legacy font files (${(removedBytes / 1024).toFixed(0)}KB)`
);

// Phase 2: Strip legacy font source URLs from CSS files + relax font-display
function stripWoffFromFile(filePath) {
  if (!fs.existsSync(filePath)) return false;
  const original = fs.readFileSync(filePath, 'utf-8');
  // Remove woff/ttf source list items, e.g.
  //   ,url(/GDKVM/_astro/x.woff) format("woff")
  //   ,url(/GDKVM/_astro/x.ttf) format("truetype")
  let stripped = original
    .replace(/,url\([^)]+\.woff[^)]*\) format\("woff"\)/g, '')
    .replace(/,url\([^)]+\.ttf[^)]*\) format\("truetype"\)/g, '');
  // KaTeX ships font-display:block — swap avoids the 3s invisible-text window.
  stripped = stripped.replace(/font-display\s*:\s*block/g, 'font-display:swap');
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

// Phase 3: Strip legacy font src from inlined <style> in HTML files
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
  `[strip-woff-fallback] Stripped legacy font src + font-display:swap in ${cssCleaned} CSS + ${htmlCleaned} HTML files`
);
console.log('[strip-woff-fallback] Done');