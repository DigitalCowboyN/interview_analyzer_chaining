"use client";

import { useProjects } from "@/hooks/useProjects";
import { StateGate } from "@/components/StateGate";
import { ProjectCard } from "@/components/ProjectCard";
import { displayProjectName } from "@/lib/projectName";
import { routes } from "@/lib/routes";

/** Landing: real projects as cards, all test runs as one bucket card. */
export default function HomePage() {
  const { data: projects, isLoading, isError, error } = useProjects();
  const real = projects?.filter((p) => p.kind === "real") ?? [];
  const testInterviews = (projects ?? [])
    .filter((p) => p.kind === "test")
    .reduce((sum, p) => sum + p.interview_count, 0);

  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="text-xl font-semibold text-fg">Projects</h1>
      <div className="mt-4">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={projects?.length === 0}
          emptyFallback={<p className="p-4 text-sm text-fg-muted">No projects yet.</p>}
        >
          <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {real.map((p) => (
              <li key={p.project_id}>
                <ProjectCard
                  href={routes.project(p.project_id)}
                  name={displayProjectName(p)}
                  interviewCount={p.interview_count}
                />
              </li>
            ))}
            {testInterviews > 0 && (
              <li>
                <ProjectCard href={routes.testRuns()} name="Test runs" interviewCount={testInterviews} subtle />
              </li>
            )}
          </ul>
        </StateGate>
      </div>
    </div>
  );
}
