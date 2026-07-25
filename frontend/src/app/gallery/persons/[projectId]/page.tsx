"use client";

import { useParams } from "next/navigation";
import { usePersons } from "@/hooks/usePersons";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { LiveIndicator } from "@/components/LiveIndicator";
import { PersonCardGrid } from "@/components/PersonCardGrid";

/** Project's person cards (discovery grid) — separate route tree from personas. */
export default function ProjectPersonsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: persons, isLoading, isError, error } = usePersons(projectId);
  const liveStatus = useLiveInvalidation({ projectId });

  return (
    <div className="p-6">
      <div className="flex items-center justify-between">
        <Breadcrumbs
          items={[
            { label: "Gallery", href: "/gallery" },
            { label: projectId },
            { label: "Persons" },
          ]}
        />
        <LiveIndicator status={liveStatus} />
      </div>
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
