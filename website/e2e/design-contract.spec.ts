import { test, expect, type Page } from '@playwright/test';
import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const dist = join(process.cwd(), 'dist');
const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? walk(join(dir, entry.name)) : entry.name.endsWith('.html') ? [join(dir, entry.name)] : []);
const routes = walk(dist).map(file => ({
  file,
  path: relative(dist, file).replace(/index\.html$/, ''),
  html: readFileSync(file, 'utf8'),
}));

async function openDisclosures(page: Page) {
  for (const summary of await page.locator('details > summary').all()) {
    if (!await summary.evaluate(element => (element.parentElement as HTMLDetailsElement).open)) await summary.click();
  }
}

async function probeSharedDesign(page: Page) {
  return page.evaluate(() => {
    const root = document.documentElement;
    const saved = root.getAttribute('style');
    root.style.setProperty('--size-root', '18px');
    root.style.setProperty('--leading-body', '2');
    root.style.setProperty('--font-body', 'monospace');
    root.style.setProperty('--space-unit', '6px');
    const body = getComputedStyle(document.body);
    const target = document.querySelector('#main-content') || document.body;
    const result = {
      family: body.fontFamily,
      size: parseFloat(body.fontSize),
      leading: parseFloat(body.lineHeight),
      spacing: parseFloat(getComputedStyle(target).paddingBottom),
      expectedSpacing: target === document.body ? 48 : 36,
    };
    if (saved === null) root.removeAttribute('style'); else root.setAttribute('style', saved);
    return result;
  });
}

for (const width of [440, 1440]) for (const colorScheme of ['light', 'dark'] as const) {
  for (const route of routes) test(`whole-site design ${route.path || '/'} ${width} ${colorScheme}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: width === 440 ? 956 : 1000 });
    await page.emulateMedia({ colorScheme });
    // Render the redirect's actual fallback instead of replacing it with its
    // destination. A separate test below preserves live redirect behaviour.
    if (/http-equiv=["']refresh/i.test(route.html)) {
      await page.route(`**/GDKVM/${route.path}`, async intercepted => {
        const response = await intercepted.fetch();
        await intercepted.fulfill({ response, body: (await response.text()).replace(/<meta\b[^>]*http-equiv=["']refresh["'][^>]*>/gi, '') });
      });
    }
    const response = await page.goto(route.path || './', { waitUntil: 'domcontentloaded' });
    expect(route.path === '404.html' ? [200, 404] : [200]).toContain(response?.status());
    await openDisclosures(page);
    const actual = await page.evaluate(() => {
      const style = getComputedStyle(document.body);
      return { font: style.fontFamily, size: parseFloat(style.fontSize), leading: parseFloat(style.lineHeight), width: document.documentElement.clientWidth, overflow: document.documentElement.scrollWidth };
    });
    expect(actual.font).toContain('Times');
    expect(actual.size).toBe(16);
    expect(actual.leading).toBeCloseTo(26.4, 1);
    expect(actual.overflow).toBeLessThanOrEqual(actual.width + 1);
    const probe = await probeSharedDesign(page);
    expect(probe.family).toBe('monospace');
    expect(probe.size).toBe(18);
    expect(probe.leading).toBe(36);
    expect(probe.spacing).toBe(probe.expectedSpacing);
    if (route.path.includes('reprod/') && !/http-equiv=["']refresh/i.test(route.html)) {
      await expect(page.locator('footer')).toContainText(route.path.startsWith('zh/') ? '2025年12月15日' : '2025-12-15');
    } else if (route.path.startsWith('zh/') && !route.path.includes('tool')) {
      await expect(page.locator('footer')).toContainText('2026年6月4日');
    }
    if (['en/', 'zh/reprod/', 'en/tool/', '404.html'].includes(route.path) && colorScheme === 'light') {
      await testInfo.attach('rendered-page', { body: await page.screenshot({ fullPage: true }), contentType: 'image/png' });
    }
  });
}

test('token presence without actual consumers fails the runtime contract', async ({ page }) => {
  await page.setContent('<style>:root{--size-root:16px;--font-body:serif;--leading-body:1.65;--space-unit:4px}.never-matches{font-family:var(--font-body);font-size:var(--size-root);line-height:var(--leading-body);padding:var(--space-unit)}body{font:16px/24px serif}</style><main>Unbound page</main>');
  const probe = await probeSharedDesign(page);
  expect(probe.family).not.toBe('monospace');
  expect(probe.size).not.toBe(18);
  expect(probe.spacing).not.toBe(probe.expectedSpacing);
});

test('root and legacy redirects retain destinations', async ({ page }) => {
  await page.goto('./');
  await expect(page).toHaveURL(/\/GDKVM\/en\/$/);
  await page.goto('reprod/');
  await expect(page).toHaveURL(/\/GDKVM\/en\/reprod\/$/);
});

for (const route of ['en/', 'zh/reprod/', 'en/tool/']) {
  test(`real utility roles follow their owners: ${route}`, async ({ page }) => {
    await page.goto(route);
    await openDisclosures(page);
    const actual = await page.evaluate(() => {
      const root = document.documentElement;
      root.style.setProperty('--space-unit', '6px');
      root.style.setProperty('--size-caption', '23px');
      root.style.setProperty('--size-section', '31px');
      root.style.setProperty('--size-subheading', '29px');
      root.style.setProperty('--leading-title', '1.8');
      root.style.setProperty('--width-content', '620px');
      const sample = (selector: string, property: string) => {
        const el = document.querySelector(selector);
        return el ? getComputedStyle(el).getPropertyValue(property) : null;
      };
      return {
        utilityPadding: sample('.px-4', 'padding-left'),
        width: sample('.max-w-content', 'max-width'),
        caption: sample('.text-sm', 'font-size'),
        section: sample('h2.text-section', 'font-size'),
        subheading: sample('h3.text-subheading', 'font-size'),
        h1Size: sample('h1', 'font-size'),
        h1Leading: sample('h1', 'line-height'),
      };
    });
    expect(actual.utilityPadding).toBe('24px');
    expect(actual.width).toBe('620px');
    if (route !== 'en/tool/') {
      expect(actual.caption).toBe('23px');
      expect(actual.section).toBe('31px');
    } else expect(actual.subheading).toBe('29px');
    expect(parseFloat(actual.h1Leading!)).toBeCloseTo(parseFloat(actual.h1Size!) * 1.8, 1);
  });
}

test('paper metadata, code and physical print roles retain their meaning', async ({ page }) => {
  await page.goto('en/');
  await expect(page.locator('meta[name="citation_publication_date"]')).toHaveAttribute('content', '2025/10/01');
  await page.emulateMedia({ media: 'print' });
  const print = await page.evaluate(() => ({
    body: parseFloat(getComputedStyle(document.body).fontSize),
    title: parseFloat(getComputedStyle(document.querySelector('.publication-title')!).fontSize),
    code: getComputedStyle(document.querySelector('pre code')!).fontFamily,
  }));
  expect(print.body).toBeCloseTo(11 * 96 / 72, 1);
  expect(print.title).toBe(24);
  expect(print.code).toContain('Consolas');
  await page.emulateMedia({ media: 'screen' });
  await page.goto('zh/reprod/');
  await openDisclosures(page);
  await expect(page.locator('pre').filter({ hasText: 'offline-run-20251215_123456-abcdef1gh' })).toBeVisible();
});
