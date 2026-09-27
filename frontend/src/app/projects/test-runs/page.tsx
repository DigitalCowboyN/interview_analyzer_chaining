"use client";

import { Suspense } from "react";
import { useTestRunInterviews } from "@/hooks/useTestRunInterviews";
import { useShowEmptyParam } from "@/hooks/useShowEmptyParam";
import { StateGate } from "@/components/StateGate";
import { InterviewRow } from "@/components/InterviewRow";
import { EmptyInterviewsToggle, splitEmpty } from "@/components/EmptyInterviewsToggle";
import { routes } from "@/lib/routes";

/** ADR-0034 bucket: every test-run interview, grouped by suite. */
function TestRunsPageContent() {
  const { data, isLoading, isError, error } = useTestRunInterviews();
  const { showEmpty, toggleShowEmpty } = useShowEmptyParam();

  const { withLines, emptyCount } = splitEmpty(data ?? []);
  const allEmpty = (data?.length ?? 0) > 0 && withLines.length === 0;
  const visible = showEmpty ? data ?? [] : withLines;

  const bySuite = new Map<string, NonNullable<typeof data>>();
  for (const row of visible) bySuite.set(row.suite, [...(bySuite.get(row.suite) ?? []), row]);

  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="text-xl font-semibold text-fg">Test runs</h1>
      <p className="mt-1 text-sm text-fg-muted">
        Interviews created by integration and smoke tests, grouped by suite.
      </p>
      <div className="mt-4 space-y-3">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={data?.length === 0}
          emptyFallback={<p className="p-4 text-sm text-fg-muted">No test runs.</p>}
        >
          {[...bySuite.entries()].map(([suite, rows]) => (
            <details key={suite} className="rounded-lg border border-border bg-surface">
              <summary className="cursor-pointer px-4 py-3 font-medium text-fg">
                {suite} ({rows.length})
              </summary>
              <ul className="space-y-3 p-4 pt-0">
                {rows.map((row) => (
                  <li key={row.interview_id}>
                    <InterviewRow href={routes.interview(row.project_id, row.interview_id)} interview={row} />
                  </li>
                ))}
              </ul>
            </details>
          ))}
          <EmptyInterviewsToggle
            emptyCount={emptyCount}
            allEmpty={allEmpty}
            showEmpty={showEmpty}
            onToggle={toggleShowEmpty}
            scope="test runs"
          />
        </StateGate>
      </div>
    </div>
  );
}

export default function TestRunsPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-fg-muted">Loading…</div>}>
      <TestRunsPageContent />
    </Suspense>
  );
}
