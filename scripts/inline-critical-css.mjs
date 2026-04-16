#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

function findHtmlFiles(dir) {
  const results = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) results.push(...findHtmlFiles(full));
    else if (entry.name.endsWith('.html')) results.push(full);
  }
  return results;
}

const DIST_DIR = path.join(process.cwd(), 'dist');
const htmlFiles = findHtmlFiles(DIST_DIR);
console.log(`[inline-critical-css] Found ${htmlFiles.length} HTML files`);

for (const htmlPath of htmlFiles) {
  let html = fs.readFileSync(htmlPath, 'utf-8');
  const cssLinks = [
    ...html.matchAll(/<link([^>]*)href="(\/[^"]*_astro\/[^"]+\.css[^"]*)"([^>]*)>/g),
  ];

  if (cssLinks.length === 0) {
    console.log(
      `[inline-critical-css] No CSS links in ${path.relative(DIST_DIR, htmlPath)}`
    );
    continue;
  }

  const seenFiles = new Set();
  let allCssContent = '';
  const linkTags = [];

  for (const match of cssLinks) {
    const href = match[2];
    const fullTag = match[0];
    linkTags.push(fullTag);
    if (seenFiles.has(href)) continue;
    seenFiles.add(href);
    let cssFilePath = path.join(DIST_DIR, href.replace(/^\//, ''));
    // Astro with base path emits href=/GDKVM/_astro/*.css but files live in dist/_astro/
    if (!fs.existsSync(cssFilePath) && href.startsWith('/GDKVM/')) {
      cssFilePath = path.join(DIST_DIR, href.replace(/^\/GDKVM\//, ''));
    }
    if (fs.existsSync(cssFilePath)) {
      allCssContent += fs.readFileSync(cssFilePath, 'utf-8') + '\n';
    }
  }

  if (!allCssContent) continue;

  for (const linkTag of linkTags) {
    html = html.replace(linkTag, '');
  }

  html = html.replace(
    /<!-- critical-inlined-css -->\s*<style is:inline>\s*[\s\S]*?<\/style>\s*/g,
    ''
  );

  const headEnd = html.lastIndexOf('</head>');
  if (headEnd < 0) continue;

  const styleTag =
    `\n<!-- critical-inlined-css (${allCssContent.length} bytes) -->\n` +
    `<style is:inline>\n${allCssContent}\n</style>\n`;
  html = html.slice(0, headEnd) + styleTag + html.slice(headEnd);

  fs.writeFileSync(htmlPath, html, 'utf-8');
  console.log(
    `[inline-critical-css] Inlined ${(allCssContent.length / 1024).toFixed(
      0
    )}KB CSS into ${path.relative(DIST_DIR, htmlPath)}`
  );
}

console.log('[inline-critical-css] Done');
