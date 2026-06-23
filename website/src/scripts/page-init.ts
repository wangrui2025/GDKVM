/**
 * Run `init` exactly once per page load, plus after every Astro view
 * transition (`astro:page-load`). Also fires immediately if the document
 * is already past the loading state. Use this instead of duplicating the
 * `astro:page-load` + `DOMContentLoaded` + `readyState` triplet in
 * every page-level `<script>`.
 */
export function bindPageLifecycle(init: () => void): void {
  const fire = () => {
    try {
      init();
    } catch (err) {
      // Swallow + log so a single broken init never breaks the page.
      // Pages should test the DOM presence inside init() themselves.
      console.error('[bindPageLifecycle] init failed:', err);
    }
  };
  document.addEventListener('astro:page-load', fire);
  if (document.readyState !== 'loading') {
    fire();
  } else {
    document.addEventListener('DOMContentLoaded', fire);
  }
}
