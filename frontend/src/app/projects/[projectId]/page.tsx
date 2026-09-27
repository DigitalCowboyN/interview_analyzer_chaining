"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { Suspense } from "react";
import { useInterviews } from "@/hooks/useInterviews";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { useShowEmptyParam } from "@/hooks/useShowEmptyParam";
import { StateGate } from "@/components/StateGate";
import { LiveIndicator } from "@/components/LiveIndicator";
import { InterviewRow } from "@/components/InterviewRow";
import { EmptyInterviewsToggle, splitEmpty } from "@/components/EmptyInterviewsToggle";
import { ApiError } from "@/api/client";
import { routes } from "@/lib/routes";

/** Project's interviews: title, created, participants, lines, insight chips; click-through to transcript. */
function ProjectInterviewsPageContent() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: interviews, isLoading, isError, error } = useInterviews(projectId);
  const liveStatus = useLiveInvalidation({ projectId });
  const { showEmpty, toggleShowEmpty } = useShowEmptyParam();

  if (isError && error instanceof ApiError && error.status === 404) {
    return (
      <p className="p-4 text-sm text-fg-muted">
        Project not found. <Link href={routes.home()} className="text-accent hover:underline">All projects</Link>
      </p>
    );
  }

  const { withLines, emptyCount } = splitEmpty(interviews ?? []);
  const allEmpty = (interviews?.length ?? 0) > 0 && withLines.length === 0;
  const visible = showEmpty ? interviews ?? [] : withLines;

  return (
    <div className="mx-auto max-w-5xl p-6">
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
          {allEmpty && !showEmpty ? null : (
            <ul className="space-y-3">
              {visible.map((i) => (
                <li key={i.interview_id}>
                  <InterviewRow href={routes.interview(projectId, i.interview_id)} interview={i} />
                </li>
              ))}
            </ul>
          )}
          <EmptyInterviewsToggle
            emptyCount={emptyCount}
            allEmpty={allEmpty}
            showEmpty={showEmpty}
            onToggle={toggleShowEmpty}
          />
        </StateGate>
      </div>
    </div>
  );
}

export default function ProjectInterviewsPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-fg-muted">Loading…</div>}>
      <ProjectInterviewsPageContent />
    </Suspense>
  );
}
