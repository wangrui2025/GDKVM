/**
 * Shared delegated click handlers for theme toggle and copy-to-clipboard
 * buttons. Bound to `document` via Layout.astro's inline `<script>` so the
 * handlers work after every view transition (Astro <ClientRouter />).
 *
 * Why a module script (not inline): the inline body is large enough that
 * extracting it keeps `Layout.astro` focused on layout/SEO concerns and
 * makes the handler unit-testable in isolation.
 */

export function initDelegatedClickHandlers() {
  document.addEventListener('click', (e: Event) => {
    const target = e.target as HTMLElement;

    const themeBtn = target.closest('[data-action="theme-toggle"]');
    if (themeBtn) {
      const html = document.documentElement;
      const isDark = html.classList.toggle('dark');
      try {
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
      } catch {}
      // Update both light and dark theme-color meta tags so the browser
      // chrome (Android, iOS Safari) reflects the new scheme.
      document
        .querySelectorAll('meta[name="theme-color"]')
        .forEach((m) => m.setAttribute('content', isDark ? '#111827' : '#ffffff'));
      return;
    }

    const copyBtn = target.closest('button[data-copy-target]');
    if (copyBtn) {
      const targetId = copyBtn.getAttribute('data-copy-target');
      const label = copyBtn.getAttribute('data-copy-label') || 'Copy';
      const successLabel = copyBtn.getAttribute('data-copy-success') || 'Copied!';
      const el = targetId ? document.getElementById(targetId) : null;
      if (!el) return;
      const text = el.innerText;

      const showSuccess = () => {
        const icon = copyBtn.querySelector('.copy-icon');
        const textSpan = copyBtn.querySelector('span');
        if (icon) icon.setAttribute('name', 'lucide:check');
        if (textSpan) textSpan.textContent = successLabel;
        setTimeout(() => {
          if (icon) icon.setAttribute('name', 'lucide:copy');
          if (textSpan) textSpan.textContent = label;
        }, 2000);
      };

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(showSuccess).catch(() => {});
      }
    }
  });
}
