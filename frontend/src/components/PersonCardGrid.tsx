import Link from "next/link";
import type { PersonSummary } from "@/hooks/usePersons";

/**
 * Person cards: a discovery grid, never the core display (domain-model
 * rule). Each card links through to the person core view — a distinct
 * route tree from personas (separate entity type; m:n future).
 */
export function PersonCardGrid({
  projectId,
  persons,
}: {
  projectId: string;
  persons: PersonSummary[];
}) {
  return (
    <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {persons.map((person) => (
        <li key={person.person_id}>
          <Link
            href={`/gallery/persons/${encodeURIComponent(projectId)}/${encodeURIComponent(
              person.person_id,
            )}`}
            className="block rounded border border-neutral-200 p-4 hover:bg-neutral-50"
          >
            <h3 className="font-medium text-neutral-900">{person.display_name}</h3>
            <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-500">
              <div className="flex items-center gap-1">
                <dt>Speakers</dt>
                <dd className="font-medium text-neutral-700">{person.speaker_count}</dd>
              </div>
              <div className="flex items-center gap-1">
                <dt>Interviews</dt>
                <dd className="font-medium text-neutral-700">{person.interview_count}</dd>
              </div>
            </dl>
          </Link>
        </li>
      ))}
    </ul>
  );
}
