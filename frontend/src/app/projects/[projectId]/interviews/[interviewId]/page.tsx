"use client";

import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef } from "react";
import { useTranscript } from "@/hooks/useTranscript";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { StateGate } from "@/components/StateGate";
import { LiveIndicator } from "@/components/LiveIndicator";
import { SegmentHeading } from "@/components/SegmentHeading";
import { TranscriptLine } from "@/components/TranscriptLine";
import { LineDetailPanel } from "@/components/LineDetailPanel";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { InterviewHeader } from "@/components/InterviewHeader";
import { InsightsPanel } from "@/components/InsightsPanel";
import { useInsights } from "@/hooks/useInsights";
import { useInterviews } from "@/hooks/useInterviews";
import { useProject } from "@/hooks/useProjects";
import { displayProjectName } from "@/lib/projectName";
import { formatDate } from "@/lib/formatDate";
import { routes } from "@/lib/routes";
import type { Insight } from "@/lib/insights";

function TranscriptPageContent() {
  const { projectId, interviewId } = useParams<{ projectId: string; interviewId: string }>();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { data: transcript, isLoading, isError, error } = useTranscript(interviewId);
  const insightsQuery = useInsights(interviewId);
  const { data: interviews } = useInterviews(projectId);
  const { project } = useProject(projectId);
  const liveStatus = useLiveInvalidation({ interviewId, projectId });

  // URL is the state (reload/Back restore it); replace() so line clicks don't
  // pile up history entries — Back leaves the interview.
  function setParam(key: "line" | "insight", value: string | null) {
    const next = new URLSearchParams(searchParams.toString());
    if (value) next.set(key, value);
    else next.delete(key);
    const qs = next.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }

  const selectedLine =
    transcript?.lines.find((l) => l.fragment_id === searchParams.get("line")) ?? null;
  const insights = insightsQuery.data ?? [];
  const selectedInsight = insights.find((i) => i.item_id === searchParams.get("insight")) ?? null;
  const highlighted = useMemo(
    () => new Set(selectedInsight?.supporting_fragment_ids ?? []),
    [selectedInsight],
  );
  const participants = useMemo(
    () => [...new Set(transcript?.lines.map((l) => l.speaker?.display_name).filter(Boolean) as string[])],
    [transcript],
  );
  const summary = interviews?.find((i) => i.interview_id === interviewId);

  // Scroll to the first supporting line of an insight that's present in the
  // transcript. Returns whether it found one and scrolled.
  const scrollToFirstPresentLine = useCallback(
    (insight: Insight): boolean => {
      if (!transcript) return false;
      const lineIds = new Set(transcript.lines.map((l) => l.fragment_id));
      const id = insight.supporting_fragment_ids.find((fid) => lineIds.has(fid));
      if (!id) return false;
      document.getElementById(`line-${id}`)?.scrollIntoView?.({ block: "center", behavior: "smooth" });
      return true;
    },
    [transcript],
  );

  // URL-restore scroll: when ?insight= arrives from a reload/Back rather
  // than a click, there's no click handler to scroll for it, so this effect
  // does it. Guarded by a ref so it fires once per insight, not on every
  // unrelated rerender (e.g. a transcript refetch) — clicks set this ref
  // themselves (see onSelectInsight) so this effect doesn't double-scroll
  // once the URL catches up.
  const lastScrolledInsightId = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedInsight) {
      lastScrolledInsightId.current = null;
      return;
    }
    if (selectedInsight.item_id === lastScrolledInsightId.current) return;
    if (scrollToFirstPresentLine(selectedInsight)) {
      lastScrolledInsightId.current = selectedInsight.item_id;
    }
  }, [selectedInsight, scrollToFirstPresentLine]);

  function onSelectInsight(insight: Insight) {
    if (scrollToFirstPresentLine(insight)) lastScrolledInsightId.current = insight.item_id;
    setParam("insight", insight.item_id);
  }

  return (
    <div className="mx-auto grid max-w-7xl gap-6 p-6 lg:grid-cols-[minmax(0,1fr)_24rem]">
      <div className="min-w-0">
        <div className="flex items-center justify-between">
          <Breadcrumbs
            items={
              project?.kind === "test"
                ? [
                    { label: "Test runs", href: routes.testRuns() },
                    { label: displayProjectName(project), href: routes.project(projectId) },
                    { label: transcript?.title ?? "Interview" },
                  ]
                : [
                    { label: project ? displayProjectName(project) : projectId, href: routes.project(projectId) },
                    { label: transcript?.title ?? "Interview" },
                  ]
            }
          />
          <LiveIndicator status={liveStatus} />
        </div>
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={transcript?.lines.length === 0}
          emptyFallback={
            <div className="p-4 text-sm text-fg-muted">
              This interview has no transcript lines yet.
            </div>
          }
        >
          {transcript && (
            <>
              <InterviewHeader
                title={transcript.title}
                participants={participants}
                date={formatDate(summary?.created_at)}
                lineCount={transcript.lines.length}
                metadata={transcript.metadata}
              />
              <div className="mt-4">
                {transcript.lines.map((line, index) => {
                  const previous = transcript.lines[index - 1];
                  // Segment heading precedes the first line of each segment —
                  // derived from the segment field changing between
                  // consecutive lines (a null->non-null or id change).
                  const startsNewSegment =
                    line.segment !== null &&
                    (!previous || previous.segment?.segment_id !== line.segment.segment_id);
                  const continuesUtterance = Boolean(
                    line.utterance_id &&
                      previous &&
                      previous.utterance_id === line.utterance_id,
                  );

                  return (
                    <div key={line.fragment_id}>
                      {startsNewSegment && (
                        <SegmentHeading topic={line.segment?.topic ?? null} />
                      )}
                      <TranscriptLine
                        line={line}
                        continuesUtterance={continuesUtterance}
                        highlighted={highlighted.has(line.fragment_id)}
                        onSelect={(l) => setParam("line", l.fragment_id)}
                      />
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </StateGate>
      </div>
      <aside className="lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto">
        {selectedLine ? (
          <LineDetailPanel
            projectId={projectId}
            interviewId={interviewId}
            line={selectedLine}
            onClose={() => setParam("line", null)}
          />
        ) : (
          <div className="rounded-lg border border-border bg-surface p-4">
            <h2 className="mb-3 font-semibold text-fg">Insights</h2>
            {insightsQuery.isError ? (
              <p className="text-sm text-danger">Couldn&rsquo;t load insights.</p>
            ) : (
              <InsightsPanel
                insights={insights}
                selectedId={selectedInsight?.item_id ?? null}
                onSelect={onSelectInsight}
                isLoading={insightsQuery.isLoading}
              />
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

export default function TranscriptPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-fg-muted">Loading…</div>}>
      <TranscriptPageContent />
    </Suspense>
  );
}
