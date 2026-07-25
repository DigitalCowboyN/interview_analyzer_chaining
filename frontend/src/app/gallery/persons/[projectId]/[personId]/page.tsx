"use client";

import { useParams } from "next/navigation";
import { usePersonDetail } from "@/hooks/usePersonDetail";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { LiveIndicator } from "@/components/LiveIndicator";
import { PersonCoreView } from "@/components/PersonCoreView";

/**
 * Person CORE view: identity facts (linked speakers per interview) with a
 * loose link to the persona profile — navigation only, never an embed
 * (domain-model rule).
 */
export default function PersonDetailPage() {
  const { projectId, personId } = useParams<{ projectId: string; personId: string }>();
  const { data: person, isLoading, isError, error } = usePersonDetail(projectId, personId);
  const liveStatus = useLiveInvalidation({ projectId, personId });

  return (
    <div className="p-6">
      <div className="flex items-center justify-between">
        <Breadcrumbs
          items={[
            { label: "Gallery", href: "/gallery" },
            { label: projectId, href: `/gallery/persons/${encodeURIComponent(projectId)}` },
            { label: person?.display_name ?? personId },
          ]}
        />
        <LiveIndicator status={liveStatus} />
      </div>
      <h1 className="text-lg font-semibold">{person?.display_name ?? personId}</h1>

      <div className="mt-4">
        <StateGate isLoading={isLoading} isError={isError} error={error}>
          {person && (
            <PersonCoreView
              projectId={projectId}
              personId={personId}
              links={person.links}
              contributesToPersona={person.contributes_to_persona}
            />
          )}
        </StateGate>
      </div>
    </div>
  );
}
