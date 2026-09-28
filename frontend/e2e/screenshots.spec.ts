import { test, expect } from "@playwright/test";

/** Visual check (spec §Testing): landing, project, interview in both themes.
 * Needs the `samples` project loaded (plan Task 9 Step 1). Output in
 * test-results/screens/ — look at every image before calling the UI done. */
for (const colorScheme of ["light", "dark"] as const) {
  test.describe(`${colorScheme} theme`, () => {
    test.use({ colorScheme, viewport: { width: 1440, height: 900 } });

    test(`screens (${colorScheme})`, async ({ page }) => {
      await page.goto("/");
      await expect(page.getByRole("link", { name: /Samples/ })).toBeVisible();
      await page.screenshot({ path: `test-results/screens/landing-${colorScheme}.png`, fullPage: true });

      await page.getByRole("link", { name: /Samples/ }).click();
      await expect(page.getByRole("link", { name: /Weekly Planning Sync/ })).toBeVisible();
      await page.screenshot({ path: `test-results/screens/project-${colorScheme}.png`, fullPage: true });

      await page.getByRole("link", { name: /Weekly Planning Sync/ }).click();
      await expect(page.getByRole("heading", { name: /Decisions/ })).toBeVisible();
      await page.screenshot({ path: `test-results/screens/interview-${colorScheme}.png` });
    });
  });
}
