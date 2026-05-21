#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Astro Integration: inline-critical-css
 * Inlines all <link rel="stylesheet"> CSS from _astro/ into each HTML file's <head>.
 * This eliminates render-blocking CSS requests and improves LCP.
 */
function inlineCriticalCss() {
  return {
    name: 'inline-critical-css',
    hooks: {
      'astro:build:done': ({ dir }) => {
        const distDir = fileURLToPath(dir);
        console.log('[inline-critical-css] Running on dir:', distDir);

        function findHtmlFiles(dir) {
          const results = [];
          try {
            for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
              const full = path.join(dir, entry.name);
              if (entry.isDirectory()) results.push(...findHtmlFiles(full));
              else if (entry.name.endsWith('.html')) results.push(full);
            }
          } catch {}
          return results;
        }

        const htmlFiles = findHtmlFiles(distDir);
        console.log(`[inline-critical-css] Found ${htmlFiles.length} HTML files`);

        for (const htmlPath of htmlFiles) {
          let html = fs.readFileSync(htmlPath, 'utf-8');
          const cssLinks = [
            ...html.matchAll(/<link([^>]*)href="(\/[^"]*_astro\/[^"]+\.css[^"]*)"([^>]*)>/g),
          ];

          if (cssLinks.length === 0) {
            console.log(
              `[inline-critical-css] No CSS links in ${path.relative(distDir, htmlPath)}`
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
            let cssFilePath = path.join(distDir, href.replace(/^\//, ''));
            // Astro with base path emits href=/GDKVM/_astro/*.css but files live in dist/_astro/
            if (!fs.existsSync(cssFilePath) && href.startsWith('/GDKVM/')) {
              cssFilePath = path.join(distDir, href.replace(/^\/GDKVM\//, ''));
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
            )}KB CSS into ${path.relative(distDir, htmlPath)}`
          );
        }

        console.log('[inline-critical-css] Done');
      },
    },
  };
}

export default inlineCriticalCss;
