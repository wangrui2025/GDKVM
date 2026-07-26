// src/integrations/sitemap-seo.mjs
// Round 22 P2 SEO: alias dist/sitemap.xml → dist/sitemap-index.xml
//
// @astrojs/sitemap v3 emits `sitemap-index.xml` by default, not the
// conventional `sitemap.xml` that many crawlers and SEO tooling
// hardcode. Copy the index file to the conventional name so both
// URLs return the same content. robots.txt keeps pointing at the
// canonical sitemap-index.xml name; this is a compatibility alias.

import { copyFileSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export default function sitemapSeo() {
  return {
    name: 'sitemap-seo',
    hooks: {
      'astro:build:done': ({ dir }) => {
        const distDir = fileURLToPath(dir);
        const srcPath = join(distDir, 'sitemap-index.xml');
        const dstPath = join(distDir, 'sitemap.xml');
        if (!existsSync(srcPath)) return;
        if (existsSync(dstPath)) return; // idempotent: don't overwrite if
        // a future Astro version starts emitting sitemap.xml natively.
        const content = readFileSync(srcPath, 'utf-8');
        copyFileSync(srcPath, dstPath);
        console.log(`[sitemap-alias] Copied sitemap-index.xml → sitemap.xml (${content.length} bytes)`);
      },
    },
  };
}
