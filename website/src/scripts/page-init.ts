/**
 * Run `init` exactly once per page load, plus after every Astro view
 * transition (`astro:page-load`). Also fires immediately if the document
 * is already past the loading state. Use this instead of duplicating the
 * `astro:page-load` + `DOMContentLoaded` + `readyState` triplet in
 * every page-level `<script>`.
 *
 * IMPORTANT: `astro:page-load` ALSO fires on the initial page load when
 * ClientRouter is mounted, so without deduping we'd call `init` twice on
 * the first paint (once via the readyState branch, once via the event).
 * For a click handler attach that's a no-op the second time, but for any
 * `init` that registers a delegated listener on `document`, double-fire
 * means double listeners → one user click fires the handler twice →
 * toggle-style effects cancel out. See CASE-GDKVM-THEME-TOGGLE-DOUBLE-FIRE-20260623.
 */
export function bindPageLifecycle(init: () => void): void {
  let hasFired = false;
  const fire = () => {
    if (hasFired) return;
    hasFired = true;
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
