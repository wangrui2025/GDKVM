import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import astroIcon from 'astro-icon';
import tailwindcss from '@tailwindcss/vite';
import inlineCriticalCss from './src/integrations/inline-critical-css.mjs';
import sitemapSeo from './src/integrations/sitemap-seo.mjs';

export default defineConfig({
  site: 'https://wangrui2025.github.io',
  base: '/GDKVM',
  outDir: 'dist',
  prefetch: { prefetchAll: true },
  image: {
    remotePatterns: [
      { protocol: 'https', hostname: 'mykcs.github.io' },
      { protocol: 'https', hostname: 'raw.githubusercontent.com' },
    ],
  },
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh'],
    routing: {
      // With `prefixDefaultLocale: true`, every page lives under /en/ or /zh/.
      // `redirectToDefaultLocale: true` makes Astro emit a built-in root
      // redirector stub at build time — no need for a hand-written
      // src/pages/index.astro (removed Round 22, P1: `route '/' conflicts
      // with higher priority route '/'` build warning).
      prefixDefaultLocale: true,
      redirectToDefaultLocale: false,
    },
  },
  integrations: [
    sitemap({
      lastmod: new Date(),
      // Round 22 P1 (cross-site audit 2026-07-26): emit <xhtml:link
      // rel="alternate" hreflang="…"> for every localized URL. Without
      // this, Google Search Console sees the sitemap as monolingual and
      // per-page <link rel="alternate"> tags are the only hreflang signal.
      i18n: {
        defaultLocale: 'en',
        locales: { en: 'en', zh: 'zh' },
      },
      // Round 22 P1: append x-default to every multi-locale path group.
      // @astrojs/sitemap v3.7.3 emits en+zh from the i18n config but never
      // auto-emits x-default even though every per-page <link rel="alternate">
      // tag in the site does (see Layout.astro).
      serialize: (item) => {
        if (item.links && Array.isArray(item.links) && item.links.length > 1) {
          const en = item.links.find((l) => l.lang === 'en');
          if (en && !item.links.some((l) => l.lang === 'x-default')) {
            item.links.push({ url: en.url, lang: 'x-default' });
          }
        }
        return item;
      },
      // Exclude 404 pages and the built-in root redirector stub.
      // The bare host (`/GDKVM/`) is the i18n auto-generated stub that
      // carries <meta name="robots" content="noindex">.
      filter: (page) => !page.endsWith('/404/') && !page.endsWith('/GDKVM/'),
    }),
    astroIcon(),
    sitemapSeo(),
    inlineCriticalCss(),
  ],
  vite: {
    plugins: [tailwindcss()],
    resolve: {
      preserveSymlinks: true,
    },
  },
  experimental: {
    rustCompiler: true,
  },
});
