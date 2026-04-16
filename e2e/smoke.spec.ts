import { test, expect } from '@playwright/test';

test.describe('GDKVM smoke tests', () => {
  test('homepage has title and theme toggle', async ({ page }) => {
    await page.goto('en/');
    await expect(page).toHaveTitle(/GDKVM/i);

    const themeBtn = page.locator('button[aria-label="Toggle Theme"]').first();
    await expect(themeBtn).toBeVisible();

    // Toggle dark mode
    await themeBtn.click();
    const html = page.locator('html');
    await expect(html).toHaveClass(/dark/);

    // Toggle back to light
    await themeBtn.click();
    await expect(html).not.toHaveClass(/dark/);
  });

  test('language switcher works', async ({ page }) => {
    await page.goto('en/');
    const zhLink = page.locator('a[href="/GDKVM/zh/"]').first();
    await expect(zhLink).toBeVisible();
    await zhLink.click();
    await expect(page).toHaveURL(/\/zh\/$/);
  });

  test('reproduction page renders', async ({ page }) => {
    await page.goto('en/reprod/');
    await expect(page).toHaveTitle(/GDKVM/i);
    await expect(page.locator('text=Reproduction').first()).toBeVisible();
  });

  test('client-router navigation via language switcher', async ({ page }) => {
    await page.goto('en/');

    const zhLink = page.locator('a[href="/GDKVM/zh/"]').first();
    await zhLink.click();

    // Wait for View Transition navigation
    await expect(page).toHaveURL(/\/zh\/$/);

    // Verify astro-route-announcer is present and has appropriate role
    const announcer = page.locator('.astro-route-announcer');
    await expect(announcer).toHaveAttribute('aria-live', 'assertive');
    await expect(announcer).toHaveAttribute('aria-atomic', 'true');
  });
});
