"use client";

import { useParams } from "next/navigation";
import { useInterviews } from "@/hooks/useInterviews";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { LiveIndicator } from "@/components/LiveIndicator";
import { InterviewList } from "@/components/InterviewList";

/** Project's interviews: title, created, fragment count; click-through to transcript. */
export default function ProjectInterviewsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: interviews, isLoading, isError, error } = useInterviews(projectId);
  const liveStatus = useLiveInvalidation({ projectId });

  return (
    <div className="p-6">
      <div className="flex items-center justify-between">
        <Breadcrumbs
          items={[
            { label: "Workbench", href: "/workbench" },
            { label: projectId },
          ]}
        />
        <LiveIndicator status={liveStatus} />
      </div>
      <h1 className="text-lg font-semibold">Interviews</h1>
      <div className="mt-4">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={interviews?.length === 0}
          emptyFallback={
            <div className="p-4 text-sm text-neutral-500">
              No interviews yet.
            </div>
          }
        >
          <InterviewList projectId={projectId} interviews={interviews ?? []} />
        </StateGate>
      </div>
    </div>
  );
}
