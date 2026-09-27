"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { useState } from "react";
import { useInterviews } from "@/hooks/useInterviews";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { LiveIndicator } from "@/components/LiveIndicator";
import { InterviewRow } from "@/components/InterviewRow";
import { ApiError } from "@/api/client";
import { routes } from "@/lib/routes";

/** Project's interviews: title, created, participants, lines, insight chips; click-through to transcript. */
export default function ProjectInterviewsPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: interviews, isLoading, isError, error } = useInterviews(projectId);
  const liveStatus = useLiveInvalidation({ projectId });
  const [showEmpty, setShowEmpty] = useState(false);

  if (isError && error instanceof ApiError && error.status === 404) {
    return (
      <p className="p-4 text-sm text-fg-muted">
        Project not found. <Link href={routes.home()} className="text-accent hover:underline">All projects</Link>
      </p>
    );
  }

  const withLines = interviews?.filter((i) => i.fragment_count > 0) ?? [];
  const empty = (interviews?.length ?? 0) - withLines.length;
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
          <ul className="space-y-3">
            {visible.map((i) => (
              <li key={i.interview_id}>
                <InterviewRow href={routes.interview(projectId, i.interview_id)} interview={i} />
              </li>
            ))}
          </ul>
          {empty > 0 && (
            <button
              type="button"
              onClick={() => setShowEmpty((v) => !v)}
              className="mt-3 text-sm text-accent"
            >
              {showEmpty ? "Hide empty interviews" : `Show ${empty} empty`}
            </button>
          )}
        </StateGate>
      </div>
    </div>
  );
}
