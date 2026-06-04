#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Astro Integration: strip-woff-fallback
 *
 * Strips the woff (non-woff2) fallback from Noto Sans @font-face declarations
 * in inlined CSS, then removes the corresponding woff files from dist/_astro/.
 *
 * Why: KaTeX ships woff (used for math) and Noto Sans ships woff+woff2.
 * Modern browsers (>97%) support woff2, so the woff fallback only adds ~464K
 * of download for users who can't use woff2 anyway. KaTeX woff MUST stay
 * (it's the only font format KaTeX ships).
 *
 * Run order: must run AFTER inline-critical-css (which inlines CSS into HTML),
 * so the woff src() removals apply to the inlined <style> blocks.
 */
function stripWoffFallback() {
  return {
    name: 'strip-woff-fallback',
    hooks: {
      'astro:build:done': ({ dir }) => {
        const distDir = fileURLToPath(dir);
        console.log('[strip-woff-fallback] Running on dir:', distDir);

        // 1. Remove noto-sans-*.woff files from dist/_astro/
        //    KaTeX_*.woff files are preserved (used for math rendering).
        const astroDir = path.join(distDir, '_astro');
        let removedFiles = 0;
        let bytesRemoved = 0;
        if (fs.existsSync(astroDir)) {
          for (const entry of fs.readdirSync(astroDir)) {
            if (entry.startsWith('noto-sans-') && entry.endsWith('.woff')) {
              const full = path.join(astroDir, entry);
              const size = fs.statSync(full).size;
              fs.unlinkSync(full);
              removedFiles++;
              bytesRemoved += size;
            }
          }
        }
        console.log(
          `[strip-woff-fallback] Removed ${removedFiles} noto-sans-*.woff files (${(bytesRemoved / 1024).toFixed(0)}KB)`
        );

        // 2. Strip `,url(.../noto-sans-...woff) format("woff")` from any
        //    inlined CSS inside <style is:inline> blocks. The woff url() in
        //    @fontsource CSS always comes AFTER the woff2 url() and is
        //    preceded by a comma, so we can safely remove the tail.
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
        let htmlFilesTouched = 0;
        let totalCssBytesRemoved = 0;

        for (const htmlPath of htmlFiles) {
          let html = fs.readFileSync(htmlPath, 'utf-8');
          const before = html.length;

          // Match the woff fallback as ",url(PATH.woff) format("woff")" or
          // ",url(PATH.woff) format('woff')". Keep the woff2 part.
          html = html.replace(
            /,url\(\/[^)"']*noto-sans-[^)"']*\.woff\)\s*format\((['"])woff\1\)/g,
            ''
          );

          if (html.length !== before) {
            fs.writeFileSync(htmlPath, html, 'utf-8');
            htmlFilesTouched++;
            totalCssBytesRemoved += before - html.length;
          }
        }

        console.log(
          `[strip-woff-fallback] Stripped woff fallback from ${htmlFilesTouched} HTML files (${(totalCssBytesRemoved / 1024).toFixed(1)}KB CSS removed)`
        );
        console.log(
          `[strip-woff-fallback] Total savings: ${((bytesRemoved + totalCssBytesRemoved) / 1024).toFixed(0)}KB`
        );
        console.log('[strip-woff-fallback] Done');
      },
    },
  };
}

export default stripWoffFallback;
