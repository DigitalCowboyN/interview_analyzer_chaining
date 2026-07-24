import Link from "next/link";
import type { PersonLink } from "@/hooks/usePersonDetail";

/**
 * Person CORE view (not a card): identity facts (linked speakers per
 * interview) plus a loose link to the persona profile. This is a link
 * BETWEEN views (navigates to the persona core view) — never an embed of
 * one entity's display inside the other's (domain-model rule; the m:n
 * future means a person may one day link to several persona profiles).
 */
export function PersonCoreView({
  projectId,
  personId,
  links,
  contributesToPersona,
}: {
  projectId: string;
  personId: string;
  links: PersonLink[];
  contributesToPersona: boolean;
}) {
  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-sm font-semibold uppercase text-neutral-500">
          Linked speakers ({links.length})
        </h2>
        {links.length === 0 ? (
          <p className="mt-2 text-sm text-neutral-400">No linked speakers.</p>
        ) : (
          <ul className="mt-2 space-y-2">
            {links.map((link) => (
              <li
                key={`${link.interview_id}-${link.speaker_id}`}
                className="flex flex-wrap items-center justify-between gap-2 rounded border border-neutral-200 p-3"
              >
                <span className="text-sm text-neutral-900">
                  {link.speaker_display_name}
                </span>
                <span className="rounded bg-blue-50 px-1.5 py-0.5 text-xs text-blue-700">
                  {link.interview_title}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="text-sm font-semibold uppercase text-neutral-500">Persona profile</h2>
        {contributesToPersona ? (
          <Link
            href={`/gallery/personas/${encodeURIComponent(projectId)}/${encodeURIComponent(
              personId,
            )}`}
            className="mt-2 inline-block text-sm text-blue-700 hover:underline"
          >
            View persona profile →
          </Link>
        ) : (
          <p className="mt-2 text-sm text-neutral-400">
            No persona profile yet for this person.
          </p>
        )}
      </section>
    </div>
  );
}
