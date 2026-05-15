import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import astroIcon from 'astro-icon';
import expressiveCode from 'astro-expressive-code';

export default defineConfig({
  site: 'https://wangrui2025.github.io',
  base: '/GDKVM',
  outDir: 'dist',
  prefetch: true,
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh'],
    routing: {
      prefixDefaultLocale: true,
      redirectToDefaultLocale: true,
    },
  },
  integrations: [
    sitemap(),
    astroIcon(),
    expressiveCode({
      themes: ['github-dark', 'github-light'],
      styleOverrides: {
        borderRadius: '0.5rem',
      },
    }),
  ],
  experimental: {
    rustCompiler: true,
  },
});
