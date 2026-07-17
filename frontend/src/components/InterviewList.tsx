import Link from "next/link";
import type { InterviewSummary } from "@/hooks/useInterviews";

/** Renders interview rows (title, created, fragment count); each navigates to its transcript. */
export function InterviewList({
  projectId,
  interviews,
}: {
  projectId: string;
  interviews: InterviewSummary[];
}) {
  return (
    <ul className="divide-y divide-neutral-200">
      {interviews.map((interview) => (
        <li key={interview.interview_id}>
          <Link
            href={`/workbench/${encodeURIComponent(projectId)}/${encodeURIComponent(
              interview.interview_id,
            )}`}
            className="flex items-center justify-between py-3 hover:bg-neutral-50"
          >
            <span className="font-medium text-neutral-900">
              {interview.title}
            </span>
            <span className="flex items-center gap-4 text-sm text-neutral-500">
              <span>{interview.created_at}</span>
              <span>
                {interview.fragment_count}{" "}
                {interview.fragment_count === 1 ? "fragment" : "fragments"}
              </span>
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
