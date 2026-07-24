"use client";

import { useParams } from "next/navigation";
import { usePersons } from "@/hooks/usePersons";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { PersonCardGrid } from "@/components/PersonCardGrid";

/** Project's person cards (discovery grid) — separate route tree from personas. */
export default function ProjectPersonsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: persons, isLoading, isError, error } = usePersons(projectId);

  return (
    <div className="p-6">
      <Breadcrumbs
        items={[
          { label: "Gallery", href: "/gallery" },
          { label: projectId },
          { label: "Persons" },
        ]}
      />
      <h1 className="text-lg font-semibold">Persons</h1>

      <div className="mt-4">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={persons?.length === 0}
          emptyFallback={
            <div className="p-4 text-sm text-neutral-500">No persons yet.</div>
          }
        >
          <PersonCardGrid projectId={projectId} persons={persons ?? []} />
        </StateGate>
      </div>
    </div>
  );
}
