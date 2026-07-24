import type { InterviewSummary } from "@/hooks/useInterviews";
import type { PersonaCardData } from "@/hooks/usePersonaCards";
import { PersonaCardGrid } from "@/components/PersonaCardGrid";

/**
 * Per-interview persona list (v1): interview selector → persona profiles
 * present in that interview → same core view. Derived client-side from the
 * persona cards' `interview_ids` field — no extra endpoint (per the plan).
 */
export function PersonaInterviewFilter({
  projectId,
  interviews,
  personas,
  selectedInterviewId,
  onSelectInterview,
}: {
  projectId: string;
  interviews: InterviewSummary[];
  personas: PersonaCardData[];
  selectedInterviewId: string;
  onSelectInterview: (interviewId: string) => void;
}) {
  const filtered = selectedInterviewId
    ? personas.filter((p) => p.interview_ids.includes(selectedInterviewId))
    : personas;

  return (
    <div>
      <label
        htmlFor="persona-interview-filter"
        className="text-xs font-semibold uppercase text-neutral-500"
      >
        Filter by interview
      </label>
      <select
        id="persona-interview-filter"
        value={selectedInterviewId}
        onChange={(e) => onSelectInterview(e.target.value)}
        className="mt-1 block rounded border border-neutral-300 p-1 text-sm"
      >
        <option value="">All interviews</option>
        {interviews.map((interview) => (
          <option key={interview.interview_id} value={interview.interview_id}>
            {interview.title}
          </option>
        ))}
      </select>

      <div className="mt-4">
        {filtered.length === 0 ? (
          <p className="text-sm text-neutral-400">
            No persona profiles for this interview.
          </p>
        ) : (
          <PersonaCardGrid projectId={projectId} personas={filtered} />
        )}
      </div>
    </div>
  );
}
