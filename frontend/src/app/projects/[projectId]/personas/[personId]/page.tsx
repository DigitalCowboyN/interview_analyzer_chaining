"use client";

import { useParams } from "next/navigation";
import { usePersonaDetail } from "@/hooks/usePersonaDetail";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { LiveIndicator } from "@/components/LiveIndicator";
import { PersonaCoreView } from "@/components/PersonaCoreView";

/** Persona CORE view: dimension-grouped items with per-interview provenance. */
export default function PersonaDetailPage() {
  const { projectId, personId } = useParams<{ projectId: string; personId: string }>();
  const { data: persona, isLoading, isError, error } = usePersonaDetail(projectId, personId);
  const liveStatus = useLiveInvalidation({ projectId, personId });

  return (
    <div className="p-6">
      <div className="flex items-center justify-end">
        <LiveIndicator status={liveStatus} />
      </div>
      <h1 className="text-lg font-semibold">{persona?.display_name ?? personId}</h1>
      <p className="mt-1 text-xs text-fg-muted">
        Persona profiles are currently seeded from per-person contributions.
      </p>

      <div className="mt-4">
        <StateGate isLoading={isLoading} isError={isError} error={error}>
          {persona && <PersonaCoreView dimensions={persona.dimensions} />}
        </StateGate>
      </div>
    </div>
  );
}
