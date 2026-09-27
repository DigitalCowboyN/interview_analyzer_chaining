"use client";

import { useParams, usePathname, useRouter } from "next/navigation";
import { useProjects } from "@/hooks/useProjects";
import { displayProjectName } from "@/lib/projectName";
import { routes, TEST_RUNS_ID } from "@/lib/routes";

const TABS = ["personas", "people", "review"] as const;

/** The tab segment right after `/projects/<id>/` in `pathname`, if it's one
 * we carry across projects (never an interview id — see lead ruling). */
function currentTab(pathname: string): (typeof TABS)[number] | undefined {
  const segment = pathname.split("/")[3];
  return TABS.find((tab) => tab === segment);
}

/** Header project dropdown. The URL is the only source of truth for the
 * current project (ADR-0030) — no component state, so Back always agrees. */
export function ProjectSwitcher() {
  const router = useRouter();
  const pathname = usePathname() ?? "";
  const params = useParams<{ projectId?: string }>();
  const { data: projects } = useProjects();

  const real = projects?.filter((p) => p.kind === "real") ?? [];
  const hasTests = projects?.some((p) => p.kind === "test") ?? false;
  const urlProject = params?.projectId ?? (pathname === routes.testRuns() ? TEST_RUNS_ID : "");
  const urlKind = projects?.find((p) => p.project_id === urlProject)?.kind;
  const current = urlKind === "test" ? TEST_RUNS_ID : urlProject;
  const unlisted = current !== "" && current !== TEST_RUNS_ID && !real.some((p) => p.project_id === current);

  function onChange(value: string) {
    if (value === "") router.push(routes.home());
    else if (value === TEST_RUNS_ID) router.push(routes.testRuns());
    else {
      const tab = currentTab(pathname);
      router.push(tab ? routes[tab](value) : routes.project(value));
    }
  }

  return (
    <label className="flex items-center gap-2 text-sm text-fg-muted">
      <span className="sr-only">Project</span>
      <select
        aria-label="Project"
        value={current}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-border bg-surface px-2 py-1 text-sm text-fg"
      >
        <option value="">All projects</option>
        {real.map((p) => (
          <option key={p.project_id} value={p.project_id}>
            {displayProjectName(p)}
          </option>
        ))}
        {unlisted && <option value={current}>{current}</option>}
        {(hasTests || current === TEST_RUNS_ID) && <option value={TEST_RUNS_ID}>Test runs</option>}
      </select>
    </label>
  );
}
