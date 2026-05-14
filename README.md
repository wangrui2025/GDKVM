# GDKVM

[![Astro](https://img.shields.io/badge/Astro-6.x-BC52EE?logo=astro)](https://astro.build)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4.x-06B6D4?logo=tailwindcss)](https://tailwindcss.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![Playwright](https://img.shields.io/badge/Playwright-E2E-2EAD33?logo=playwright)](https://playwright.dev)

> Project page for **GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule** (ICCV 2025).

Live site: [https://wangrui2025.github.io/GDKVM/](https://wangrui2025.github.io/GDKVM/)

## Overview

GDKVM is a project page built with Astro 6.x, Tailwind CSS v4, and TypeScript. It presents research on echocardiography video segmentation using a novel spatiotemporal key-value memory architecture with gated delta rule.

## Tech Stack

- **Framework**: [Astro](https://astro.build) v6.1.8
- **Styling**: [Tailwind CSS](https://tailwindcss.com) v4.2.2
- **Language**: TypeScript 5.9
- **Testing**: Playwright (E2E)
- **Deployment**: GitHub Pages

## Project Structure

```
GDKVM/
├── src/
│   ├── pages/[lang]/         # i18n routes (en, zh)
│   ├── components/           # Reusable Astro components
│   ├── layouts/              # Page layouts
│   ├── content/              # JSON translation files
│   ├── styles/               # Page-specific CSS
│   └── scripts/              # Client-side scripts
├── e2e/                      # Playwright E2E tests
├── public/                   # Static assets
├── .astro/                   # Astro generated types
└── dist/                     # Build output
```

## Development

### Prerequisites

- Node.js 20+
- npm

### Install Dependencies

```bash
npm install
npx playwright install
```

### Local Development

```bash
npm run dev
```

### Build

```bash
npm run build
```

### Run Tests

```bash
npx playwright test
```

## i18n

The site supports English (`en`) and Chinese (`zh`) with Astro's built-in i18n routing:

- `prefixDefaultLocale: true` — all URLs include locale prefix
- `redirectToDefaultLocale: true` — root `/` redirects to `/en/`

Content translations are stored in `src/content/` as JSON files.

## License

This project page is released under the MIT License.
