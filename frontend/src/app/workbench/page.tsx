"use client";

import { useProjects } from "@/hooks/useProjects";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { ProjectList } from "@/components/ProjectList";

/** Workbench nav root: projects list, click-through to each project's interviews. */
export default function WorkbenchPage() {
  const { data: projects, isLoading, isError, error } = useProjects();

  return (
    <div className="p-6">
      <Breadcrumbs items={[{ label: "Workbench" }]} />
      <h1 className="text-lg font-semibold">Projects</h1>
      <div className="mt-4">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={projects?.length === 0}
          emptyFallback={
            <div className="p-4 text-sm text-neutral-500">
              No projects yet.
            </div>
          }
        >
          <ProjectList projects={projects ?? []} />
        </StateGate>
      </div>
    </div>
  );
}
