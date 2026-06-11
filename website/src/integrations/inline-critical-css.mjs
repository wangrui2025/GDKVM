#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Astro Integration: inline-critical-css
 * Inlines all <link rel="stylesheet"> CSS from _astro/ into each HTML file's <head>.
 * This eliminates render-blocking CSS requests and improves LCP.
 *
 * Also post-processes dist/sitemap-0.xml to remove legacy redirect stubs
 * (/reprod/, /) that @astrojs/sitemap can't filter out via its `filter`
 * option (3.7.x zod validation rejects both arrow and function forms).
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

        // ======== Sitemap post-process: remove legacy redirect stubs ========
        // @astrojs/sitemap 3.7.x validates its `filter` option via zod and
        // rejects every form we've tried. As a post-process workaround,
        // re-write the generated dist/sitemap-0.xml to drop only the bare
        // meta-refresh stubs (root /GDKVM/ and /GDKVM/reprod/).
        // NOTE: /GDKVM/en/reprod/ and /GDKVM/zh/reprod/ are REAL pages
        // (100-line HTML), NOT stubs — they MUST remain in the sitemap.
        const sitemapFiles = ['sitemap-0.xml', 'sitemap-index.xml']
          .map(f => path.join(distDir, f))
          .filter(p => fs.existsSync(p));
        for (const sf of sitemapFiles) {
          let xml = fs.readFileSync(sf, 'utf-8');
          const before = (xml.match(/<loc>/g) || []).length;
          // Drop <url>...</url> blocks whose <loc> is the bare /GDKVM/reprod/
          // or /GDKVM/ root (both are single-line meta-refresh stubs).
          // DO NOT touch locale-prefixed variants /en/reprod/ and /zh/reprod/
          // — those are real content pages.
          xml = xml.replace(
            /<url>\s*<loc>https:\/\/wangrui2025\.github\.io\/GDKVM\/reprod\/<\/loc>\s*<\/url>/g,
            ''
          );
          xml = xml.replace(
            /<url>\s*<loc>https:\/\/wangrui2025\.github\.io\/GDKVM\/<\/loc>\s*<\/url>/g,
            ''
          );
          // Tidy stray blank lines left after removals
          xml = xml.replace(/^\s*\n/gm, '\n');
          const after = (xml.match(/<loc>/g) || []).length;
          fs.writeFileSync(sf, xml, 'utf-8');
          if (before !== after) {
            console.log(
              `[sitemap-filter] ${path.basename(sf)}: ${before} → ${after} URLs (removed ${before - after} stubs)`
            );
          }
        }

        console.log('[inline-critical-css] Done');
      },
    },
  };
}

export default inlineCriticalCss;
