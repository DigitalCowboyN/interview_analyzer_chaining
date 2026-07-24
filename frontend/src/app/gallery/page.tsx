"use client";

import { useState } from "react";
import Link from "next/link";
import { useProjects } from "@/hooks/useProjects";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";

/**
 * Gallery home: project selector → Personas / Persons areas + Worklist
 * entry (Task 8's route — will 404 until Task 8 ships, expected).
 */
export default function GalleryPage() {
  const { data: projects, isLoading, isError, error } = useProjects();
  const [selectedProjectId, setSelectedProjectId] = useState("");

  return (
    <div className="p-6">
      <Breadcrumbs items={[{ label: "Gallery" }]} />
      <h1 className="text-lg font-semibold">Gallery</h1>

      <div className="mt-4">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={projects?.length === 0}
          emptyFallback={
            <div className="p-4 text-sm text-neutral-500">No projects yet.</div>
          }
        >
          <label
            htmlFor="gallery-project-select"
            className="text-xs font-semibold uppercase text-neutral-500"
          >
            Project
          </label>
          <select
            id="gallery-project-select"
            value={selectedProjectId}
            onChange={(e) => setSelectedProjectId(e.target.value)}
            className="mt-1 block rounded border border-neutral-300 p-1 text-sm"
          >
            <option value="">Select a project…</option>
            {projects?.map((project) => (
              <option key={project.project_id} value={project.project_id}>
                {project.project_id}
              </option>
            ))}
          </select>

          {selectedProjectId && (
            <ul className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
              <li>
                <Link
                  href={`/gallery/personas/${encodeURIComponent(selectedProjectId)}`}
                  className="block rounded border border-neutral-200 p-4 hover:bg-neutral-50"
                >
                  <h2 className="font-medium text-neutral-900">Personas</h2>
                  <p className="mt-1 text-sm text-neutral-500">
                    Persona profiles seeded from interview contributions.
                  </p>
                </Link>
              </li>
              <li>
                <Link
                  href={`/gallery/persons/${encodeURIComponent(selectedProjectId)}`}
                  className="block rounded border border-neutral-200 p-4 hover:bg-neutral-50"
                >
                  <h2 className="font-medium text-neutral-900">Persons</h2>
                  <p className="mt-1 text-sm text-neutral-500">
                    Identity facts — linked speakers per interview.
                  </p>
                </Link>
              </li>
              <li>
                <Link
                  href={`/gallery/worklist?project=${encodeURIComponent(selectedProjectId)}`}
                  className="block rounded border border-neutral-200 p-4 hover:bg-neutral-50"
                >
                  <h2 className="font-medium text-neutral-900">Worklist</h2>
                  <p className="mt-1 text-sm text-neutral-500">
                    Review queue and suggestions.
                  </p>
                </Link>
              </li>
            </ul>
          )}
        </StateGate>
      </div>
    </div>
  );
}
