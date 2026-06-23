/**
 * Initialize KaTeX auto-render on math delimiters ($$...$$ / $...$).
 * Loaded as a dynamic ESM import from HomePage.astro's `<script>` tag
 * (extracted to keep the page body small and the rendering pipeline
 * testable in isolation).
 *
 * The CDN auto-render bundle is loaded via the <script is:inline> tag
 * in HomePage.astro — see the SOP comment in that file for why CDN
 * is acceptable for a 39KB math renderer that isn't on the critical path.
 */

const KAATEX_OPTIONS = {
  delimiters: [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false },
  ],
  throwOnError: false,
} as const;

export function initKatex() {
  // The CDN auto-render bundle exposes `renderMathInElement` on the
  // global. Guard against late loads (e.g. during view transitions).
  const render = (
    globalThis as unknown as { renderMathInElement?: (el: Element, opts: object) => void }
  ).renderMathInElement;
  if (typeof render === 'function') {
    render(document.body, KAATEX_OPTIONS);
  }
}
