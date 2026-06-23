/**
 * Apply persisted theme before paint to avoid FOUC.
 *
 * Runs synchronously at script execution time (not via astro:page-load)
 * to set the .dark class on <html> before the first paint. Re-applies on
 * every Astro view transition (`astro:after-swap`).
 *
 * Replaces the previous inline `<script is:inline>` block in
 * `src/layouts/Layout.astro`. Kept as a module so consumers can import
 * the function for tests or future page-level hooks.
 */
export function applyTheme(): void {
  const theme =
    localStorage.getItem('theme')
    || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.classList.toggle('dark', theme === 'dark');
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (meta) {
    meta.setAttribute('content', theme === 'dark' ? '#111827' : '#ffffff');
  }
}

export function initThemeBootstrap(): void {
  applyTheme();
  document.addEventListener('astro:after-swap', applyTheme);
}
