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

  test.describe('tool page', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('en/tool/');
      await expect(page).toHaveTitle(/Predictable Scale/i);
    });

    test('tab switching works', async ({ page }) => {
      const tab1 = page.locator('button[data-tab="tab1"]').first();
      const tab2 = page.locator('button[data-tab="tab2"]').first();
      const pane1 = page.locator('#tab1').first();
      const pane2 = page.locator('#tab2').first();

      await expect(tab1).toBeVisible();
      await expect(tab2).toBeVisible();

      // Tab1 should be active by default
      await expect(pane1).toHaveClass(/active/);

      // Click tab2
      await tab2.click();
      await expect(pane2).toHaveClass(/active/);
      await expect(pane1).not.toHaveClass(/active/);

      // Click back to tab1
      await tab1.click();
      await expect(pane1).toHaveClass(/active/);
      await expect(pane2).not.toHaveClass(/active/);
    });

    test('form calculation produces results', async ({ page }) => {
      // Ensure tab1 is active
      const tab1 = page.locator('button[data-tab="tab1"]').first();
      await tab1.click();

      const modelSizeInput = page.locator('#modelSize').first();
      const trainingTokensInput = page.locator('#trainingTokens').first();
      const submitBtn = page.locator('#modelForm button[type="submit"]').first();

      await modelSizeInput.fill('1e8');
      await trainingTokensInput.fill('1e9');
      await submitBtn.click();

      // Result should show computed values instead of defaults
      const bsValue = page.locator('#bsValue').first();
      const lrValue = page.locator('#lrValue').first();
      await expect(bsValue).not.toContainText('BS: -');
      await expect(lrValue).not.toContainText('LR: -');
    });

    test('cascading selectors update options', async ({ page }) => {
      // Switch to tab2 (Loss Heatmap)
      const tab2 = page.locator('button[data-tab="tab2"]').first();
      await tab2.click();

      const modelType = page.locator('#modelType').first();
      const nValue = page.locator('#nValue').first();
      const naValue = page.locator('#naValue').first();
      const dValue = page.locator('#dValue').first();

      // Dense model: nValue should have options, naValue hidden
      await modelType.selectOption('Dense');
      await expect(nValue).toBeVisible();
      const denseNOptions = await nValue.locator('option').count();
      expect(denseNOptions).toBeGreaterThan(0);

      // dValue should also have options for Dense
      await page.waitForTimeout(100); // allow cascade update
      const denseDOptions = await dValue.locator('option').count();
      expect(denseDOptions).toBeGreaterThan(0);

      // Switch to Moe: naValue should appear
      await modelType.selectOption('Moe');
      await expect(nValue).toBeVisible();
      await page.waitForTimeout(100);
      const moeNOptions = await nValue.locator('option').count();
      expect(moeNOptions).toBeGreaterThan(0);

      // naValue should be visible for Moe (naItem uses 'hidden' class, not inline style, in JS runtime)
      const naItem = page.locator('#naItem');
      // Wait for the JS class toggle to take effect
      await page.waitForTimeout(200);
      await expect(naItem).not.toHaveClass(/hidden/);
      const moeNaOptions = await naValue.locator('option').count();
      expect(moeNaOptions).toBeGreaterThan(0);

      // dValue should have options for Moe
      const moeDOptions = await dValue.locator('option').count();
      expect(moeDOptions).toBeGreaterThan(0);
    });
  });
});
