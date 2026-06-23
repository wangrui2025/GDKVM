/**
 * Register the production service worker.
 *
 * Best-effort registration: failures (e.g. third-party cookie blocking,
 * insecure context) are intentionally swallowed because the SW is a
 * progressive enhancement, not a hard dependency.
 *
 * Replaces the previous inline `<script>` block in
 * `src/layouts/Layout.astro`.
 */
export function registerServiceWorker(): void {
  if (!('serviceWorker' in navigator)) return;
  const baseUrl = import.meta.env.BASE_URL || '';
  navigator.serviceWorker
    .register(`${baseUrl}/sw.js`)
    .catch(() => {
      /* SW is best-effort; see file header. */
    });
}
