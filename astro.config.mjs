import { defineConfig } from 'astro/config';

export default defineConfig({
  output: 'static',
  base: '/GDKVM',
  build: {
    format: 'file',
  },
});
