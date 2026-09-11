/**
 * Run `init` exactly once per page load, plus after every Astro view
 * transition (`astro:page-load`). Also fires immediately if the document
 * is already past the loading state. Use this instead of duplicating the
 * `astro:page-load` + `DOMContentLoaded` + `readyState` triplet in
 * every page-level `<script>`.
 *
 * IMPORTANT: `astro:page-load` ALSO fires on the initial page load when
 * ClientRouter is mounted, so the readyState fallback and that event can
 * target the same route. Dedupe by route, not forever: ClientRouter keeps
 * this module alive while replacing page DOM, so revisiting `/tool/` must
 * initialize the newly swapped-in form again.
 */
export function bindPageLifecycle(init: () => void): void {
  let lastRoute: string | null = null;
  const fire = () => {
    const route = `${window.location.pathname}${window.location.search}`;
    if (lastRoute === route) return;
    lastRoute = route;
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
