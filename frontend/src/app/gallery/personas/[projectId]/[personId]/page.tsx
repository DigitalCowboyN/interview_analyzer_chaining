"use client";

import { useParams } from "next/navigation";
import { usePersonaDetail } from "@/hooks/usePersonaDetail";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { PersonaCoreView } from "@/components/PersonaCoreView";

/** Persona CORE view: dimension-grouped items with per-interview provenance. */
export default function PersonaDetailPage() {
  const { projectId, personId } = useParams<{ projectId: string; personId: string }>();
  const { data: persona, isLoading, isError, error } = usePersonaDetail(projectId, personId);

  return (
    <div className="p-6">
      <Breadcrumbs
        items={[
          { label: "Gallery", href: "/gallery" },
          { label: projectId, href: `/gallery/personas/${encodeURIComponent(projectId)}` },
          { label: persona?.display_name ?? personId },
        ]}
      />
      <h1 className="text-lg font-semibold">{persona?.display_name ?? personId}</h1>
      <p className="mt-1 text-xs text-neutral-400">
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
