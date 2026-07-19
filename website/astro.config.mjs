import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import astroIcon from 'astro-icon';
import tailwindcss from '@tailwindcss/vite';
import inlineCriticalCss from './src/integrations/inline-critical-css.mjs';

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
      // `redirectToDefaultLocale: true` makes the root path / auto-redirect
      // to /en/. We must keep `src/pages/index.astro` (empty, meta-refresh)
      // as a stub — see CONTEXT.md "已知限制" for why.
      prefixDefaultLocale: true,
      redirectToDefaultLocale: true,
    },
  },
  integrations: [
    // Sitemap output contains only the 6 canonical locale pages
    // (/en/, /zh/, + reprod/tool). The legacy /reprod/ redirect stub
    // and the / root stub are excluded automatically (they carry
    // <meta name="robots" content="noindex">), which is what we want.
    sitemap(),
    astroIcon(),
    inlineCriticalCss(),
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
