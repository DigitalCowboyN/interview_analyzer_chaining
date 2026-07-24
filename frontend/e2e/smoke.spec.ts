import { execFileSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { expect, test } from "@playwright/test";

/**
 * M5.0 Task 9 — UI Playwright smoke: the first end-to-end proof that a real
 * ingest is navigable in the workbench AND that a correction made through
 * the UI actually round-trips through the real event-sourced write path
 * (command -> ESDB -> dockerized projection-service -> Neo4j -> refetch).
 *
 * Journey: workbench nav (projects -> interviews -> transcript) -> transcript
 * renders the seeded lines -> edit one line's text through the UI -> the
 * "edited" badge settles (proves the projection consumer delivered the
 * SentenceEdited event, not just that the POST returned 202).
 *
 * M5.1 Task 6 adds a second journey below: the "money shot" — with the
 * transcript page already OPEN, a brand-new line is appended SERVER-SIDE
 * (no UI interaction) and must appear on the page with NO user action at
 * all (no click, no reload) — proof that the SSE bridge
 * (src/ui/notifications.py's EsdbWatcher -> src/api/routers/ui.py's
 * `/ui/streams/events` -> the frontend's `useLiveInvalidation` hook) is
 * wired end to end, not just that a plain refetch-on-navigate works (the
 * M5.0 journey above never proves that — it reloads the page fresh each
 * time via `page.goto`/nav clicks).
 *
 * REQUIRED SERVICES (all must be up before running; `make ui-smoke` handles
 * all of this):
 *   1. Dockerized dev stack: `docker compose up -d --build neo4j eventstore
 *      projection-service` — the test Neo4j/ESDB from `make test-infra-up`
 *      do NOT work here: only the dockerized projection-service consumes and
 *      delivers events to the DEV Neo4j. See
 *      tests/integration/test_deployed_projection_smoke.py for the same
 *      requirement on the backend side.
 *   2. Backend: `uvicorn src.main:app` on :8000, with ESDB_CONNECTION_STRING
 *      overridden to esdb://localhost:2113?tls=false (the committed .env
 *      points at the docker-internal "eventstore" hostname, unresolvable
 *      from a host-run process) — started automatically by this file's
 *      `webServer` config in playwright.config.ts.
 *   3. Frontend: `npm run dev` on :3000 (its next.config.ts `/api/*` rewrite
 *      proxies to :8000) — also started by the `webServer` config.
 *
 * ONE-TIME SETUP: on a machine that has never run Playwright, install its
 * browser binary first — `npx playwright install chromium` (from `frontend/`)
 * — or this spec fails before reaching any of the above.
 *
 * Seeding: frontend/e2e/seed_smoke.py ingests one small LABELED transcript
 * through the real command path (IngestionOrchestrator — same idiom as
 * test_deployed_projection_smoke.py), shelled out to from this spec's
 * beforeAll/afterAll (simpler than wiring ids through Makefile env vars for
 * one spec). LABELED format means speaker assignment is parsed from labels,
 * not LLM-inferred, so seeding needs no working enrichment/LLM calls.
 *
 * Gating: this whole file only runs behind UI_SMOKE=1 (belt-and-suspenders
 * with playwright.config.ts's testIgnore) — mirrors the backend's
 * DEPLOYED_SMOKE=1 pytest gate. Run via `make ui-smoke`, never `npm test`
 * (vitest.config.ts excludes `e2e/**`) and never a bare `npx playwright test`.
 */

test.skip(process.env.UI_SMOKE !== "1", "UI smoke: run via `make ui-smoke`");

const PYTHON_BIN =
  process.env.PYTHON_BIN ?? path.join(os.homedir(), ".pyenv/versions/3.10.7/bin/python");
const SEED_SCRIPT = path.join(__dirname, "seed_smoke.py");
// Run from the repo root: src/ modules resolve config/prompt paths (e.g.
// prompts/ingestion_prompts.yaml) relative to the process cwd, not to this
// file's location.
const REPO_ROOT = path.resolve(__dirname, "..", "..");

interface SeedResult {
  project_id: string;
  interview_id: string;
  title: string;
  first_line_text: string;
  fragment_count: number;
}

/** Shells out to seed_smoke.py's `append-line` subcommand — appends ONE new
 * line to the ALREADY-seeded interview via the real command path
 * (CreateSentenceCommand), independent of the UI. Used by the live-append
 * journey below; see seed_smoke.py's header for why this needs a separate
 * subcommand from `seed` (that one only ever creates a brand-new
 * interview_id, never appends to an existing one). */
function appendLine(interviewId: string, index: number, text: string): void {
  execFileSync(
    PYTHON_BIN,
    [SEED_SCRIPT, "append-line", "--interview-id", interviewId, "--index", String(index), "--text", text],
    { encoding: "utf-8", cwd: REPO_ROOT },
  );
}

let seeded: SeedResult | undefined;

test.beforeAll(() => {
  const output = execFileSync(PYTHON_BIN, [SEED_SCRIPT, "seed"], {
    encoding: "utf-8",
    cwd: REPO_ROOT,
  });
  seeded = JSON.parse(output.trim().split("\n").pop()!) as SeedResult;
});

test.afterAll(() => {
  if (!seeded) return;
  execFileSync(
    PYTHON_BIN,
    [SEED_SCRIPT, "cleanup", "--project-id", seeded.project_id, "--interview-id", seeded.interview_id],
    { encoding: "utf-8", cwd: REPO_ROOT },
  );
});

test("workbench nav renders seeded transcript, and a text edit settles", async ({ page }) => {
  const data = seeded!;

  // Workbench nav: projects -> this project -> its one interview.
  await page.goto("/workbench");
  await page.getByRole("link", { name: data.project_id }).click();
  await expect(page).toHaveURL(new RegExp(`/workbench/${encodeURIComponent(data.project_id)}$`));

  await page.getByRole("link", { name: data.title }).click();
  await expect(page).toHaveURL(
    new RegExp(`/workbench/${encodeURIComponent(data.project_id)}/${data.interview_id}$`),
  );

  // Transcript renders the seeded line.
  const lineButton = page.getByRole("button", { name: new RegExp(data.first_line_text) });
  await expect(lineButton).toBeVisible();
  await expect(lineButton.getByText("edited", { exact: true })).toHaveCount(0);

  // Perform a text edit through the UI.
  await lineButton.click();
  const detailPanel = page.getByRole("dialog", { name: "Line detail" });
  await expect(detailPanel).toBeVisible();

  await detailPanel.getByRole("button", { name: "Edit text" }).click();
  const textarea = detailPanel.getByRole("textbox", { name: "Edit sentence text" });
  // Append rather than replace: `lineButton` is a live locator matched
  // against the current accessible name via `data.first_line_text` — it
  // must still match after the edit lands, so the original text has to
  // survive as a substring (a full replacement would make `lineButton`
  // resolve to nothing the instant the edit settles). "updated" (not
  // "edited") in the suffix so it doesn't collide with the exact-text
  // "edited" badge assertion below.
  const editedText = `${data.first_line_text} (updated via UI smoke)`;
  await textarea.fill(editedText);
  await detailPanel.getByRole("button", { name: "Save" }).click();

  // Assert the edited badge settles. Generous timeout in the spirit of the
  // deployed smoke's 90s consumer-group poll bound: the intent hook's own
  // client-side poll (2s x 10 attempts, ~20s) must finish inside this window
  // for the badge to ever appear, so 40s comfortably covers that plus
  // projection-delivery latency and page-render overhead. Exact match: the
  // badge's own text node is exactly "edited" (the surrounding paragraph
  // text is a separate node, so this doesn't need exact to disambiguate
  // from it, but pins the intent precisely).
  await expect(lineButton.getByText("edited", { exact: true })).toBeVisible({ timeout: 40_000 });
});

test("a server-side line append appears live on an open transcript page with no user action", async ({
  page,
}) => {
  const data = seeded!;

  await page.goto(`/workbench/${encodeURIComponent(data.project_id)}/${data.interview_id}`);

  // Baseline: the seeded lines are up, and the live indicator has connected
  // (its EventSource onopen fired) -- both must be true before the append,
  // or a later false-positive ("line appeared") could just be an ordinary
  // slow initial render rather than a live update.
  await expect(page.getByRole("button", { name: new RegExp(data.first_line_text) })).toBeVisible();
  await expect(page.getByText("Live updates on")).toBeVisible({ timeout: 10_000 });

  const liveText = "This line arrived live via the M5.1 UI smoke -- no reload, no click.";
  const newLineButton = page.getByRole("button", { name: new RegExp(liveText) });
  await expect(newLineButton).toHaveCount(0); // not present before the append

  // Append server-side, through the real command path -- NOT through the
  // page. No Playwright action (click/reload/goto) follows this call: the
  // whole point is that the page updates itself.
  appendLine(data.interview_id, data.fragment_count, liveText);

  // Generous timeout in the same spirit as the text-edit journey above:
  // command -> ESDB -> dockerized projection-service -> Neo4j -> the SSE
  // bridge's ESDB catch-up subscription -> the browser's EventSource ->
  // useLiveInvalidation's debounced invalidate -> refetch -> render. Each
  // hop adds real latency; none of it involves a client-side poll loop
  // (unlike the edit journey), so this is the only timeout budget covering
  // the whole live path end to end.
  await expect(newLineButton).toBeVisible({ timeout: 40_000 });
});
