"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useWorklist } from "@/hooks/useWorklist";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { LiveIndicator } from "@/components/LiveIndicator";
import { WorklistRows } from "@/components/WorklistRows";

/**
 * Gallery worklist (M5.0 Task 8): review queue for a project — reads
 * `project` from the query string (the gallery home links here as
 * `/gallery/worklist?project=<id>`, not a path segment), so `useSearchParams`
 * needs a Suspense boundary for the CSR bailout.
 */
function WorklistContent() {
  const searchParams = useSearchParams();
  const projectId = searchParams.get("project") ?? "";
  const { data, isLoading, isError, error } = useWorklist(projectId);
  const liveStatus = useLiveInvalidation({ projectId });

  const isEmpty =
    Boolean(data) &&
    data!.lens_items.length === 0 &&
    data!.claims.length === 0 &&
    data!.entity_merge_suggestions.length === 0 &&
    data!.person_link_suggestions.length === 0 &&
    data!.flags.length === 0;

  return (
    <div className="p-6">
      <div className="flex items-center justify-between">
        <Breadcrumbs
          items={[
            { label: "Gallery", href: "/gallery" },
            ...(projectId ? [{ label: projectId }] : []),
            { label: "Worklist" },
          ]}
        />
        <LiveIndicator status={liveStatus} />
      </div>
      <h1 className="text-lg font-semibold">Worklist</h1>

      <div className="mt-4">
        {!projectId ? (
          <div className="p-4 text-sm text-neutral-500">
            Select a project from the Gallery to view its worklist.
          </div>
        ) : (
          <StateGate
            isLoading={isLoading}
            isError={isError}
            error={error}
            isEmpty={isEmpty}
            emptyFallback={
              <div className="p-4 text-sm text-neutral-500">Nothing to review.</div>
            }
          >
            {data && <WorklistRows projectId={projectId} data={data} />}
          </StateGate>
        )}
      </div>
    </div>
  );
}

export default function WorklistPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-neutral-500">Loading…</div>}>
      <WorklistContent />
    </Suspense>
  );
}
