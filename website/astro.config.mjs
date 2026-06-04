import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import astroIcon from 'astro-icon';
import tailwindcss from '@tailwindcss/vite';
import inlineCriticalCss from './src/integrations/inline-critical-css.mjs';
import stripWoffFallback from './src/integrations/strip-woff-fallback.mjs';

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
      prefixDefaultLocale: true,
      redirectToDefaultLocale: true,
    },
  },
  integrations: [
    // Sitemap includes the legacy /reprod/ redirect stub. Tried
    // zod-compatible filter (arrow + function declaration) but
    // @astrojs/sitemap 3.7.3's zod schema rejects both forms.
    // Accept this as a known cosmetic item — search engines
    // follow the meta-refresh to the canonical page, so no
    // SEO penalty beyond a tiny crawl-budget waste.
    sitemap(),
    astroIcon(),
    inlineCriticalCss(),
    // MUST run after inlineCriticalCss (which inlines CSS into HTML) so the
    // woff src() removals apply to the inlined <style> blocks. Strips
    // noto-sans-*.woff fallback (woff2 covers >97% of modern browsers).
    stripWoffFallback(),
  ],
  vite: {
    plugins: [tailwindcss()],
    resolve: {
      preserveSymlinks: true,
    },
  },
  experimental: {
    // `compiler: "rs"` was the v3/v4-era syntax. Astro 6 declares
    // `rustCompiler?: boolean` under `experimental`. The flag was
    // silently ignored in v6; switching to the v6 path enables it.
    rustCompiler: true,
  },
});
