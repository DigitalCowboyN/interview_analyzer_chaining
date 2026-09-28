import Link from "next/link";
import type { InterviewSummary } from "@/hooks/useInterviews";
import { formatDate } from "@/lib/formatDate";
import { insightChip, orderNodeTypes } from "@/lib/insights";

/** One interview in a project list: identity, size, and what was extracted. */
export function InterviewRow({ href, interview }: { href: string; interview: InterviewSummary }) {
  const counts = interview.insight_counts ?? {};
  const types = orderNodeTypes(Object.keys(counts));

  return (
    <Link
      href={href}
      className="block rounded-lg border border-border bg-surface p-4 hover:border-accent"
    >
      <div className="flex items-baseline justify-between gap-4">
        <span className="font-medium text-fg">{interview.title}</span>
        <span className="shrink-0 text-sm text-fg-muted">{formatDate(interview.created_at)}</span>
      </div>
      <div className="mt-1 text-sm text-fg-muted">
        {interview.participants.length > 0 && <span>{interview.participants.join(", ")} · </span>}
        <span>
          {interview.fragment_count} {interview.fragment_count === 1 ? "line" : "lines"}
        </span>
      </div>
      {types.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-2">
          {types.map((t) => (
            <li
              key={t}
              data-testid="insight-chip"
              className="rounded-full bg-accent-subtle px-2 py-0.5 text-xs text-accent"
            >
              {counts[t]} {insightChip(t)}
            </li>
          ))}
        </ul>
      )}
    </Link>
  );
}
