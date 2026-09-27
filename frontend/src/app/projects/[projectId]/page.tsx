"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { useInterviews } from "@/hooks/useInterviews";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { LiveIndicator } from "@/components/LiveIndicator";
import { InterviewList } from "@/components/InterviewList";
import { ApiError } from "@/api/client";
import { routes } from "@/lib/routes";

/** Project's interviews: title, created, fragment count; click-through to transcript. */
export default function ProjectInterviewsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: interviews, isLoading, isError, error } = useInterviews(projectId);
  const liveStatus = useLiveInvalidation({ projectId });

  if (isError && error instanceof ApiError && error.status === 404) {
    return (
      <p className="p-4 text-sm text-fg-muted">
        Project not found. <Link href={routes.home()} className="text-accent hover:underline">All projects</Link>
      </p>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-end">
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
            <div className="p-4 text-sm text-fg-muted">
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
