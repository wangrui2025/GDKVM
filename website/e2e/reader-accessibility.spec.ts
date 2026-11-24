import { expect, test } from '@playwright/test';

for (const mobile of [false, true]) {
  test.describe(mobile ? 'phone-sized' : 'desktop', () => {
    test.use({ viewport: mobile ? { width: 440, height: 956 } : { width: 1440, height: 900 }, deviceScaleFactor: mobile ? 3 : 1, hasTouch: mobile, isMobile: mobile });
    test('paper action links and BibTeX remain readable and keyboard accessible', async ({ page }) => {
      await page.goto('en/');
      await page.locator('#bibtex-content').waitFor();
      const pre = page.locator('pre:has(#bibtex-content)');
      await expect(pre).toHaveAttribute('tabindex', '0');
      await pre.focus();
      await expect(pre).toBeFocused();
      for (const dark of [false, true]) {
        await page.evaluate(d => document.documentElement.classList.toggle('dark', d), dark);
        await page.waitForTimeout(250);
        const results = await page.locator('a.bg-gray-800, .copy-btn').evaluateAll(nodes => {
          const canvas = document.createElement('canvas');
          canvas.width = canvas.height = 1;
          const ctx = canvas.getContext('2d')!;
          function luminance(color: string) {
            ctx.clearRect(0, 0, 1, 1);
            ctx.fillStyle = color;
            ctx.fillRect(0, 0, 1, 1);
            return Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3))
              .map(c => c / 255)
              .map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
              .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
          }
          return nodes.filter(n => n.getClientRects().length > 0).map(n => {
            const s = getComputedStyle(n);
            const fg = luminance(s.color), bg = luminance(s.backgroundColor);
            return { contrast: (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05), opacity: s.opacity, foreground:s.color, background:s.backgroundColor, target:n.className, label:n.textContent?.trim().slice(0,60) };
          });
        });
        expect(results.length).toBeGreaterThan(1);
        for (const result of results) {
          expect(result.opacity).toBe('1');
          expect(result.contrast, JSON.stringify(result)).toBeGreaterThanOrEqual(4.5);
        }
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2)).toBe(true);
      }
    });
  });
}
