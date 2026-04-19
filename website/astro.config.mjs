import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://wangrui2025.github.io',
  base: '/GDKVM',
  integrations: [sitemap()]
});
