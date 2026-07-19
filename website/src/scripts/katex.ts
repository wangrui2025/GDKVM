/**
 * Initialize KaTeX auto-render on math delimiters ($$...$$ / $...$).
 * Loaded as a dynamic ESM import from HomePage.astro's `<script>` tag
 * (extracted to keep the page body small and the rendering pipeline
 * testable in isolation).
 *
 * The auto-render bundle is bundled locally via Astro's module graph
 * (`import 'katex/dist/contrib/auto-render.min.js'` in HomePage.astro),
 * which registers `renderMathInElement` on globalThis. No CDN involved —
 * KaTeX CSS is likewise bundled from the npm package in Layout.astro.
 */

const KAATEX_OPTIONS = {
  delimiters: [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false },
  ],
  throwOnError: false,
} as const;

export function initKatex() {
  // The side-effect import exposes `renderMathInElement` on the
  // global. Guard against late loads (e.g. during view transitions).
  const render = (
    globalThis as unknown as { renderMathInElement?: (el: Element, opts: object) => void }
  ).renderMathInElement;
  if (typeof render === 'function') {
    render(document.body, KAATEX_OPTIONS);
  }
}
