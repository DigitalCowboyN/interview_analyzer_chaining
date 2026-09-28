"use client";

import { useParams } from "next/navigation";
import { useWorklist } from "@/hooks/useWorklist";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { LiveIndicator } from "@/components/LiveIndicator";
import { WorklistRows } from "@/components/WorklistRows";

/** Project review queue: low-confidence lens items, claims, and merge/link suggestions. */
export default function ReviewPage() {
  const { projectId } = useParams<{ projectId: string }>();
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
      <div className="flex items-center justify-end">
        <LiveIndicator status={liveStatus} />
      </div>
      <h1 className="text-lg font-semibold">Review</h1>

      <div className="mt-4">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={isEmpty}
          emptyFallback={
            <div className="p-4 text-sm text-fg-muted">Nothing to review.</div>
          }
        >
          {data && <WorklistRows projectId={projectId} data={data} />}
        </StateGate>
      </div>
    </div>
  );
}
