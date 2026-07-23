import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import os from "node:os";

/**
 * Playwright config for the M5.0 UI smoke (frontend/e2e/smoke.spec.ts) —
 * see that file's header for the full required-services list and how to run
 * it (`make ui-smoke`). Env-gated behind UI_SMOKE=1, mirroring the backend's
 * DEPLOYED_SMOKE=1 pytest gate (tests/integration/test_deployed_projection_smoke.py):
 * this config MUST NOT start real servers or run anything when UI_SMOKE isn't
 * set, so a bare `npx playwright test` never surprises anyone.
 */
const UI_SMOKE = process.env.UI_SMOKE === "1";

const FRONTEND_DIR = __dirname;
const REPO_ROOT = path.resolve(FRONTEND_DIR, "..");
const PYTHON_BIN = process.env.PYTHON_BIN ?? path.join(os.homedir(), ".pyenv/versions/3.10.7/bin/python");

// Backend command: mirrors scripts/test.sh's ".env + pyenv interpreter" idiom,
// but overrides ESDB_CONNECTION_STRING — the committed .env points ESDB at
// the docker-internal "eventstore" hostname (correct for the dockerized
// app/worker/projection-service containers), which a host-run uvicorn can't
// resolve. localhost:2113 is the same eventstore container's host-exposed
// port (docker-compose.yml `eventstore` service: "2113:2113").
const BACKEND_COMMAND = [
  `cd '${REPO_ROOT}'`,
  "set -a",
  "source .env 2>/dev/null || true",
  "set +a",
  "export ESDB_CONNECTION_STRING='esdb://localhost:2113?tls=false'",
  `'${PYTHON_BIN}' -m uvicorn src.main:app --host 0.0.0.0 --port 8000`,
].join(" && ");

export default defineConfig({
  testDir: "./e2e",
  testMatch: "smoke.spec.ts",
  // Belt-and-suspenders alongside the spec's own test.skip: if UI_SMOKE isn't
  // set, ignore the whole e2e dir so an accidental bare `npx playwright test`
  // never attempts test discovery (and therefore never starts webServer)
  // against the live dev stack.
  testIgnore: UI_SMOKE ? undefined : ["**/*"],
  fullyParallel: false,
  retries: 0,
  workers: 1,
  reporter: "list",
  // Generous overall test/hook timeout: beforeAll shells out to seed_smoke.py
  // (spaCy load + agent construction + real ingest, ~10s observed) and the
  // settle assertion itself uses a 40s expect timeout (see smoke.spec.ts) —
  // both need headroom well past Playwright's 30s default.
  timeout: 120_000,
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  // Only wired up when UI_SMOKE=1 — never start real servers otherwise.
  webServer: UI_SMOKE
    ? [
        {
          command: BACKEND_COMMAND,
          url: "http://localhost:8000/docs",
          reuseExistingServer: true,
          timeout: 60_000,
        },
        {
          command: "npm run dev",
          cwd: FRONTEND_DIR,
          url: "http://localhost:3000",
          reuseExistingServer: true,
          timeout: 60_000,
        },
      ]
    : undefined,
});
