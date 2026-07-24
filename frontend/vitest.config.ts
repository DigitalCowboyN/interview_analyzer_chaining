import { configDefaults, defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Deliberately separate from next.config.ts — Vitest runs component/unit
// tests in jsdom with zero network; Next's build config is irrelevant here.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    css: false,
    // e2e/ holds the Playwright smoke (frontend/e2e/smoke.spec.ts) — it needs
    // the live dev stack (docker + uvicorn + next dev) and is run via
    // `make ui-smoke`, never by vitest. Without this exclude, vitest's
    // default *.spec.ts glob would pick it up and fail on `npm test`.
    exclude: [...configDefaults.exclude, "e2e/**"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
