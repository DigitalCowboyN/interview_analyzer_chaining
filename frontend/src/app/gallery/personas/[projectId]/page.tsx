"use client";

import { useParams } from "next/navigation";
import { useState } from "react";
import { usePersonaCards } from "@/hooks/usePersonaCards";
import { useInterviews } from "@/hooks/useInterviews";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { PersonaInterviewFilter } from "@/components/PersonaInterviewFilter";

/**
 * Project's persona cards (discovery grid) + a per-interview filter toggle.
 * v1-honesty: persona profiles are seeded per-person — the graph has no
 * archetype Persona entity yet (kept subtle in the header copy below).
 */
export default function ProjectPersonasPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: personas, isLoading, isError, error } = usePersonaCards(projectId);
  const { data: interviews } = useInterviews(projectId);
  const [selectedInterviewId, setSelectedInterviewId] = useState("");

  return (
    <div className="p-6">
      <Breadcrumbs
        items={[
          { label: "Gallery", href: "/gallery" },
          { label: projectId },
          { label: "Personas" },
        ]}
      />
      <h1 className="text-lg font-semibold">Personas</h1>
      <p className="mt-1 text-xs text-neutral-400">
        Persona profiles are currently seeded from per-person contributions.
      </p>

      <div className="mt-4">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={personas?.length === 0}
          emptyFallback={
            <div className="p-4 text-sm text-neutral-500">
              No persona profiles yet. Run the persona lens for an interview:{" "}
              <code className="rounded bg-neutral-100 px-1 py-0.5 text-neutral-700">
                python -m src.lens &lt;interview_id&gt; persona
              </code>
            </div>
          }
        >
          {personas && personas.length > 0 && (
            <PersonaInterviewFilter
              projectId={projectId}
              interviews={interviews ?? []}
              personas={personas}
              selectedInterviewId={selectedInterviewId}
              onSelectInterview={setSelectedInterviewId}
            />
          )}
        </StateGate>
      </div>
    </div>
  );
}
